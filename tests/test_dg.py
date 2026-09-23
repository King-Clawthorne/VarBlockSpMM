"""The specialized DG comparison archive supports the numbers the paper reports."""
import csv
import json
import io
import math
import re
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_dg import analyze

SOURCE = ROOT / 'data/dg'


@unittest.skipUnless((SOURCE / 'manifest.json').exists(), 'campaign archive not present')
class DgEvidenceTests(unittest.TestCase):
  def test_process_aggregation_matches_raw_trial_products(self):
    """Independent reduction: trial medians, then products and roots.

        Deliberately do not reuse analyzer records or its geometric-mean
        helper. The retained asymmetric processes distinguish a median of
        ratios from equal process weighting in log space.
        """
    grouped = {}
    with zipfile.ZipFile(SOURCE / 'runs.zip') as archive:
      for name in archive.namelist():
        match = re.fullmatch(r'dg_e(\d+)_n(\d+)_p\d+\.csv', name)
        if not match:
          continue
        trials = {}
        for row in csv.DictReader(io.StringIO(archive.read(name).decode())):
          trials.setdefault(row['method'], []).append(float(row['gpu_ms']))
        medians = {}
        for method, values in trials.items():
          values.sort()
          middle = len(values) // 2
          medians[method] = (values[(len(values)-1)//2] + values[middle]) / 2
        grouped.setdefault(tuple(map(int, match.groups())), []).append(medians)
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      self.copy(destination)
      data = analyze(destination)
    self.assertEqual(len(grouped), 6)
    for row in data['summary']:
      processes = grouped[row['elements'], row['rhs']]
      self.assertEqual(len(processes), 3)
      expected = {
          'direct_ms': [p['direct'] for p in processes],
          'specialized_ms': [p['dg_specialized'] for p in processes],
          'fused_ms': [p['dg_fused'] for p in processes],
          'specialized': [p['dg_specialized']/p['direct'] for p in processes],
          'fused': [p['dg_fused']/p['direct'] for p in processes],
          'fused_copies': [p['dg_fused_copies']/p['direct'] for p in processes],
          'sharing': [p['dg_fused_copies']/p['dg_fused'] for p in processes],
          'best_generic': [min(p['grouped'],p['bsr8'])/p['direct'] for p in processes],
      }
      for field, values in expected.items():
        with self.subTest(elements=row['elements'], rhs=row['rhs'], field=field):
          self.assertAlmostEqual(row[field], math.prod(values)**(1/len(values)), places=11)

  def copy(self, destination):
    for name in ('manifest.json', 'source_snapshot.zip', 'runs.zip'):
      shutil.copyfile(SOURCE / name, destination / name)

  def test_accepts_the_recorded_campaign(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      self.copy(destination)
      data = analyze(destination)
    self.assertEqual(data['processes'], 18)
    # Every process must agree that direct execution won its trace.
    self.assertTrue(all(row['wins'] == 3 for row in data['summary']))

  def test_recorded_trace_allocation_is_two_plans(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      self.copy(destination)
      data = analyze(destination)
    for row in data['summary']:
      self.assertEqual(row['trace_bytes'], 2 * row['storage_bytes'])

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


if __name__ == '__main__':
  unittest.main()
