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
        # Operators larger than the assembled matrix would mean no reuse.
        line = 'specialized_structure,8,10,4096,4096\n'
        with self.assertRaisesRegex(ValueError, 'Implausible specialized structure'):
            structure({'environment': line})

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
