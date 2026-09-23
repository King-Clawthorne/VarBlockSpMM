"""The documented full rerun must produce every campaign the paper build reads.

`build_paper.ps1 -ResultsRoot <root>` consumes a fixed set of subdirectories
under that root. `run_final_validation.py` is the documented way to fill it.
A campaign added to one and not the other leaves the README workflow broken,
which is only discovered after hours of GPU time, so check it here.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'scripts/build_paper.ps1'
RUNNER = ROOT / 'scripts/run_final_validation.py'


def consumed():
  """Subdirectories the paper build reads from the results root."""
  text = BUILD.read_text(encoding='utf-8')
  return set(re.findall(r"Join-Path \$ResultsRoot '([^']+)'", text))


def produced():
  """Subdirectories the documented runner writes under the results root."""
  text = RUNNER.read_text(encoding='utf-8')
  return set(re.findall(r"str\(root / '([^']+)'\)", text))


class WorkflowCoverageTests(unittest.TestCase):
  def test_build_consumes_at_least_one_campaign(self):
    # Guards the regexes above against silently matching nothing.
    self.assertTrue(consumed())
    self.assertTrue(produced())

  def test_runner_produces_every_consumed_campaign(self):
    missing = consumed() - produced()
    self.assertEqual(missing, set(),
                     'run_final_validation.py does not produce: ' + ', '.join(sorted(missing)))

  def test_every_export_has_an_input_override(self):
    """Each export invoked with a results root must accept that root."""
    text = BUILD.read_text(encoding='utf-8')
    for script, folder in re.findall(
            r'python "\$PSScriptRoot/(\w+\.py)"[^\n]*Join-Path \$ResultsRoot \'([^\']+)\'', text):
      self.assertTrue((ROOT / 'scripts' / script).is_file(), script)
      self.assertIn(folder, produced(), f'{script} reads {folder}, which no runner produces')

  def test_dg_campaign_is_covered(self):
    # The specialized transport comparison is the most recent addition.
    self.assertIn('dg', consumed())
    self.assertIn('dg', produced())


if __name__ == '__main__':
  unittest.main()
