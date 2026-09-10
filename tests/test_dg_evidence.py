"""Regression checks for the specialized transport comparison archive."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_dg import analyze, structure

SOURCE = ROOT / 'data/dg'


@unittest.skipUnless((SOURCE / 'manifest.json').exists(), 'campaign archive not present')
class DgEvidenceTests(unittest.TestCase):
    def copy(self, destination):
        for name in ('manifest.json', 'source_snapshot.zip', 'runs.zip'):
            shutil.copyfile(SOURCE / name, destination / name)

    def check_rejected(self, alter, error):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.copy(destination)
            manifest = json.loads((destination / 'manifest.json').read_text())
            alter(manifest)
            (destination / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, error):
                analyze(destination)

    def test_accepts_the_recorded_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.copy(destination)
            data = analyze(destination)
        self.assertEqual(data['processes'], 18)
        # Every process must agree that direct execution won its trace.
        self.assertTrue(all(row['wins'] == 3 for row in data['summary']))

    def test_changed_design_is_rejected(self):
        def alter(manifest):
            manifest['reps'] += 1
        self.check_rejected(alter, 'design mismatch')

    def test_changed_comparator_set_is_rejected(self):
        def alter(manifest):
            manifest['methods'] = ['direct']
        self.check_rejected(alter, 'Comparator set')

    def test_stale_build_receipt_is_rejected(self):
        def alter(manifest):
            manifest['build_receipt']['sources']['bench/dg_specialized.cpp'] = '0' * 64
        self.check_rejected(alter, 'compiled inputs')

    def test_missing_structure_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing specialized structure'):
            structure({'environment': 'identity_negative_control,rejected\n'})

    def test_implausible_structure_record_is_rejected(self):
        # Operators as large as the assembled matrix would mean no reuse.
        line = 'specialized_structure,8,10,4096,4096,8192,16384\n'
        with self.assertRaisesRegex(ValueError, 'Implausible specialized structure'):
            structure({'environment': line})

    def test_truncated_structure_record_is_rejected(self):
        line = 'specialized_structure,8,10,4096\n'
        with self.assertRaisesRegex(ValueError, 'Missing specialized structure'):
            structure({'environment': line})

    def test_plan_storage_below_operator_bytes_is_rejected(self):
        line = 'specialized_structure,8,10,34560,35389440,1024,2048\n'
        with self.assertRaisesRegex(ValueError, 'Implausible specialized structure'):
            structure({'environment': line})

    def test_trace_allocation_must_cover_both_plans(self):
        # The timed trace builds one plan per direction, so a record claiming a
        # single plan's bytes would understate what the comparator retains.
        line = 'specialized_structure,8,10,34560,35389440,231168,231168\n'
        with self.assertRaisesRegex(ValueError, 'Implausible specialized structure'):
            structure({'environment': line})

    def test_recorded_trace_allocation_is_two_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.copy(destination)
            data = analyze(destination)
        for row in data['summary']:
            self.assertEqual(row['trace_bytes'], 2 * row['storage_bytes'])

    def test_corrupt_source_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.copy(destination)
            (destination / 'source_snapshot.zip').write_bytes(b'not a zip archive')
            with self.assertRaises(Exception):
                analyze(destination)

    def test_altered_snapshot_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.copy(destination)
            with zipfile.ZipFile(destination / 'source_snapshot.zip') as archive:
                contents = {name: archive.read(name) for name in archive.namelist()}
            target = next(n for n in contents if n.endswith('bench/dg_specialized.cpp'))
            contents[target] += b'\n// tampered\n'
            with zipfile.ZipFile(destination / 'source_snapshot.zip', 'w') as archive:
                for name, payload in contents.items():
                    archive.writestr(name, payload)
            with self.assertRaisesRegex(ValueError, 'Snapshot checksum mismatch'):
                analyze(destination)

    def test_missing_snapshot_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.copy(destination)
            with zipfile.ZipFile(destination / 'source_snapshot.zip') as archive:
                contents = {name: archive.read(name) for name in archive.namelist()}
            contents.pop(next(n for n in contents if n.endswith('scripts/run_dg.py')))
            with zipfile.ZipFile(destination / 'source_snapshot.zip', 'w') as archive:
                for name, payload in contents.items():
                    archive.writestr(name, payload)
            with self.assertRaisesRegex(ValueError, 'Snapshot coverage mismatch'):
                analyze(destination)

    def test_missing_input_hashes_are_rejected(self):
        def alter(manifest):
            manifest.pop('inputs')
        self.check_rejected(alter, 'Input coverage mismatch')

    def test_incomplete_input_coverage_is_rejected(self):
        def alter(manifest):
            manifest['inputs'].pop(sorted(manifest['inputs'])[0])
        self.check_rejected(alter, 'Input coverage mismatch')

    def test_changed_input_payload_is_rejected(self):
        def alter(manifest):
            name = next(n for n in sorted(manifest['inputs']) if n.endswith('.bin'))
            manifest['inputs'][name] = '0' * 64
        self.check_rejected(alter, 'checksum mismatch')

    def test_inputs_match_the_archived_transport_campaign(self):
        """The reuse claim is checked against the canonical archive, not asserted."""
        manifest = json.loads((SOURCE / 'manifest.json').read_text())
        canonical = ROOT / 'data/relevance-v2/manifest.json'
        if not canonical.is_file():
            self.skipTest('canonical transport archive not present')
        published = json.loads(canonical.read_text())['inputs']
        self.assertTrue(manifest['inputs'])
        for name, digest in manifest['inputs'].items():
            self.assertIn(name, published, name)
            self.assertEqual(published[name], digest, name)

    def test_archive_checksums_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.copy(destination)
            with zipfile.ZipFile(destination / 'runs.zip') as archive:
                contents = {name: archive.read(name) for name in archive.namelist()}
            target = next(n for n in contents if n.endswith('.csv'))
            contents[target] = contents[target] + b'direct,0,99,1.0,1.0\n'
            with zipfile.ZipFile(destination / 'runs.zip', 'w') as archive:
                for name, payload in contents.items():
                    archive.writestr(name, payload)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                analyze(destination)


if __name__ == '__main__':
    unittest.main()
