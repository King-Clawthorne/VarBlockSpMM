"""Regression checks for evidence validation, without CUDA or timing assertions."""
import copy
import csv
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_runs import RunRecord, validate_run, CSV_FIELDS
from build_verified import validate_build_receipt


class EvidenceTests(unittest.TestCase):
  def record(self, transform=lambda x: x, returncode=0):
    rows = [dict(method=m, position=str(p), iteration=str(i), gpu_ms='1', host_ms='2')
            for p, m in enumerate(('direct', 'baseline')) for i in range(2)]
    rows = transform(rows)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return RunRecord('test', text.getvalue(), dict(returncode=returncode))

  def test_valid(self):
    validate_run(self.record(), ['direct', 'baseline'], 2)

  def test_failed_process(self):
    with self.assertRaises(ValueError): validate_run(self.record(returncode=1), ['direct', 'baseline'], 2)

  def test_corrupt_trials(self):
    mutations = [lambda r: r[:-1], lambda r: [r[0], r[0], *r[2:]],
                 lambda r: [dict(x, gpu_ms='nan') for x in r],
                 lambda r: [dict(x, host_ms='0') for x in r],
                 lambda r: [dict(x, position='0') for x in r]]
    for mutation in mutations:
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        validate_run(self.record(mutation), ['direct', 'baseline'], 2)

  def test_stale_binary_and_source(self):
    valid = dict(sources={'src/a.cpp': 'source'}, executable_sha256='binary',
                 build_receipt=dict(sources={'src/a.cpp': 'source'}, executables={'test.exe': 'binary'}))
    validate_build_receipt(valid)
    for key, replacement in [('sources', {'src/a.cpp': 'changed'}), ('executable_sha256', 'stale')]:
      broken = copy.deepcopy(valid)
      broken[key] = replacement
      with self.assertRaises(ValueError): validate_build_receipt(broken)
    with self.assertRaises(ValueError): validate_build_receipt({})


if __name__ == '__main__': unittest.main()
