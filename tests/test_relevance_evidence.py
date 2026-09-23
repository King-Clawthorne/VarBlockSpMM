"""Regression checks for rejecting altered focused-campaign provenance."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_relevance import analyze

class RelevanceEvidenceTests(unittest.TestCase):
  def check_rejected(self, alter, error):
    with tempfile.TemporaryDirectory() as directory:
      dest=Path(directory)
      for name in ('manifest.json','sources.zip','runs.zip'):
        shutil.copyfile(ROOT/'data/relevance'/name,dest/name)
      manifest=json.loads((dest/'manifest.json').read_text())
      alter(manifest)
      (dest/'manifest.json').write_text(json.dumps(manifest))
      with self.assertRaisesRegex(ValueError,error): analyze(dest, allow_legacy=True)

  def test_changed_command_is_rejected(self):
    self.check_rejected(lambda m:m['records'][0]['command'].append('unexpected'), 'command')

  def test_missing_process_is_rejected(self):
    self.check_rejected(lambda m:m['records'].pop(), 'Missing process')

  def test_stale_build_receipt_is_rejected(self):
    def alter(m): m['build_receipt']['sources']['src/kernels/row_owned.cu']='0'*64
    self.check_rejected(alter,'Build inputs')

  def test_changed_raw_hash_is_rejected(self):
    def alter(m): m['records'][0]['stdout_sha256']='0'*64
    self.check_rejected(alter,'Raw checksum')

if __name__=='__main__': unittest.main()
