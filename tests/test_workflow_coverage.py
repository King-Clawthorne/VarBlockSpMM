"""The documented full rerun must produce every campaign the paper build reads.

`build_paper.ps1 -ResultsRoot <root>` consumes a fixed set of subdirectories
under that root. `run_final_validation.py` is the documented way to fill it.
A campaign added to one and not the other leaves the README workflow broken,
which is only discovered after hours of GPU time, so check it here.
"""
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'scripts/build_paper.ps1'
RUNNER = ROOT / 'scripts/run_final_validation.py'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_final_validation


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

  def test_runner_prepares_and_collects_relevance(self):
    with tempfile.TemporaryDirectory() as directory,          patch.object(sys, 'argv', ['run_final_validation.py', '--output-root', directory]),          patch.object(run_final_validation, 'verified_build', return_value=(Path('test.exe'), {})),          patch.object(run_final_validation, 'find_sanitizer', return_value=Path('sanitizer.exe')),          patch.object(run_final_validation.subprocess, 'run') as run:
      run.return_value.returncode = 0
      run_final_validation.main()
      commands = [call.args[0] for call in run.call_args_list]
      scripts = {Path(c[1]).name: c for c in commands if c[0] == sys.executable}
      self.assertIn('prepare_magma.py', scripts)
      self.assertIn('--all', scripts['prepare_transport.py'])
      self.assertIn('build_relevance.py', scripts)
      self.assertEqual(scripts['run_relevance.py'][-2:], ['--output', str(Path(directory) / 'relevance')])


if __name__ == '__main__':
  unittest.main()
