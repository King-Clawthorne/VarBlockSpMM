"""Numerical and input-provenance regressions for the replacement campaign."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_transport import (prepare, convergence_study, require_identity_rejection,
                               INPUT_DIRECTORY, translation_block)
from run_relevance import design, validate_inputs, expected_inputs, sha, analyze
import run_final_validation


class TransportEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        payloads = cls.root / INPUT_DIRECTORY
        prepare(256, 8, 8, payloads)
        convergence_study(payloads)
        cls.jobs = [dict(kind='transport', args=[
            (INPUT_DIRECTORY / 'transport_e256_n8').as_posix(), 8, 8, 6, 1])]
        names = expected_inputs(cls.jobs)
        with zipfile.ZipFile(cls.root/'inputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names: archive.write(cls.root/name, name)
        cls.manifest = dict(jobs=cls.jobs, inputs={name:sha(cls.root/name) for name in names},
                            input_archive_sha256=sha(cls.root/'inputs.zip'))

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def test_unchanged_state_is_rejected(self):
        base = self.root / INPUT_DIRECTORY / 'transport_e256_n8'
        initial=np.fromfile(str(base)+'.input', dtype='<f4')
        reference=np.fromfile(str(base)+'.reference', dtype='<f4')
        require_identity_rejection(initial, reference)
        with self.assertRaisesRegex(ValueError, 'negative control'):
            require_identity_rejection(initial, initial)

    def test_constant_field_is_preserved_by_projection(self):
        for m,n in ((8,64),(16,8),(32,16),(64,32)):
            result=translation_block(m,m,False)[:,0]+translation_block(m,n,True)[:,0]
            expected=np.zeros(m); expected[0]=1
            np.testing.assert_allclose(result,expected,atol=2e-12,rtol=0)

    def test_valid_archive(self): validate_inputs(self.root,self.manifest)

    def test_missing_input_hashes_are_rejected(self):
        for inputs in ({},dict(list(self.manifest['inputs'].items())[1:])):
            broken=dict(self.manifest,inputs=inputs)
            with self.assertRaisesRegex(ValueError,'Input manifest coverage'):
                validate_inputs(self.root,broken)

    def test_altered_payload_digest_is_rejected(self):
        broken=copy.deepcopy(self.manifest)
        name=next(n for n in broken['inputs'] if n.endswith('.bin'))
        broken['inputs'][name]='0'*64
        with self.assertRaisesRegex(ValueError,'Input payload checksum'):
            validate_inputs(self.root,broken)

    def test_rehashed_corrupt_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with zipfile.ZipFile(self.root/'inputs.zip') as source, zipfile.ZipFile(root/'inputs.zip','w') as target:
                for name in source.namelist():
                    payload=source.read(name)
                    if name.endswith('.reference'): payload=b'bad'+payload[3:]
                    target.writestr(name,payload)
            broken=dict(self.manifest,input_archive_sha256=sha(root/'inputs.zip'))
            with self.assertRaisesRegex(ValueError,'Input payload checksum'):
                validate_inputs(root,broken)

    def test_command_metadata_mismatch_is_rejected(self):
        broken=copy.deepcopy(self.manifest); broken['jobs'][0]['args'][2]=32
        with self.assertRaisesRegex(ValueError,'metadata differs'):
            validate_inputs(self.root,broken)

    def test_legacy_archive_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError,'Legacy campaign'):
            analyze(Path(__file__).resolve().parents[1]/'data/relevance')

    def test_core_crosses_seed_process_and_queue_mode(self):
        jobs=[j for j in design() if j['kind']=='core']
        self.assertEqual(len(jobs),2304)
        cells={}
        for j in jobs:
            key=tuple(j['args'][:4])+(j['args'][8],)
            cells.setdefault(key,set()).add((j['args'][4],j['process'],j['batch']))
            self.assertEqual(j['args'][5],20)
        self.assertEqual(len(cells),128)
        self.assertTrue(all(len(v)==18 for v in cells.values()))

    def test_final_runner_collects_required_relevance_directory(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(sys,'argv',['run_final_validation.py','--output-root',directory]), \
             patch.object(run_final_validation,'verified_build',return_value=(Path('test.exe'),{})), \
             patch.object(run_final_validation,'find_sanitizer',return_value=Path('sanitizer.exe')), \
             patch.object(run_final_validation.subprocess,'run') as run:
            run.return_value.returncode=0
            run_final_validation.main()
            commands=[call.args[0] for call in run.call_args_list]
            scripts={Path(c[1]).name:c for c in commands if c[0]==sys.executable}
            self.assertIn('prepare_magma.py',scripts)
            self.assertIn('--all',scripts['prepare_transport.py'])
            self.assertIn('build_relevance.py',scripts)
            self.assertEqual(scripts['run_relevance.py'][-2:],['--output',str(Path(directory)/'relevance')])


if __name__=='__main__': unittest.main()
