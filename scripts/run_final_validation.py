"""Build, sanitize, and collect all final and robustness campaigns without GPU overlap."""
from pathlib import Path
import json
import subprocess
import sys
import argparse
from benchmark_runs import ROOT
from build_verified import verified_build, find_sanitizer


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--output-root', type=Path, required=True,
                      help='Fresh root for all paper campaigns, including MAGMA and transport, plus validation records')
  root = parser.parse_args().output_root.resolve()
  executable, receipt = verified_build('vbsr_tests')
  logs = root / 'validation'
  logs.mkdir(parents=True, exist_ok=True)
  sanitizer = find_sanitizer()
  commands = [
      ('memcheck', [str(sanitizer), '--tool', 'memcheck', '--error-exitcode', '9', str(executable)]),
      ('racecheck', [str(sanitizer), '--tool', 'racecheck', '--error-exitcode', '9',
                     '--kernel-name', 'regex=row_owned|refresh_grouped', str(executable)]),
  ]
  for name, command in commands:
    print('Running ' + name, flush=True)
    with (logs / (name + '.log')).open('w') as log:
      result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (logs / (name + '.json')).write_text(json.dumps(dict(command=command, returncode=result.returncode,
                                                       build_receipt=receipt), indent=2))
    result.check_returncode()
  for script, args in [
      ('prepare_application.py', []),
      ('prepare_native.py', []),
      ('run_revision.py', ['--output', str(root / 'synthetic')]),
      ('run_application.py', ['--output', str(root / 'published')]),
      ('run_supplement.py', ['native', '--output', str(root / 'native')]),
      ('run_supplement.py', ['ablation', '--output', str(root / 'ablation')]),
      ('run_robustness.py', ['--output', str(root / 'robustness')]),
      ('prepare_transport.py', ['--all']),
      ('build_relevance.py', []),
      ('run_relevance.py', ['--output', str(root / 'relevance')]),
      ('run_dg.py', ['--output', str(root / 'dg')]),
  ]:
    print('Running ' + script + ' ' + ' '.join(args), flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args], cwd=ROOT, check=True)


if __name__ == '__main__': main()
