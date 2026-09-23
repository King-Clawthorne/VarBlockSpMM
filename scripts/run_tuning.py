"""Compare narrow-panel candidates on development seed 4 with full references."""
import hashlib
import argparse
import itertools
import random
import time
from pathlib import Path
from benchmark_runs import ROOT, archive_runs, run_process, save_manifest, source_hashes, source_paths
from build_verified import verified_build


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--phase', type=int, choices=(1, 2), default=1)
  parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
  args = parser.parse_args()
  phase = args.phase
  executable, receipt = verified_build('vbsr_ablation')
  output = args.output
  output.mkdir(parents=True, exist_ok=True)
  jobs = list(itertools.product((256, 1024, 4096), (2, 16), (8, 16),
                                ('uniform', 'low', 'high', 'bimodal')))
  rng = random.Random(20260908)
  rng.shuffle(jobs)
  paths = source_paths('run_tuning.py')
  save_manifest(output, dict(jobs=jobs, matrix_seed=4, reps=20, warmup=5,
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                sources=source_hashes(paths), build_receipt=receipt), paths)
  for index, (rows, degree, rhs, distribution) in enumerate(jobs):
    stem = f'{rows}_{degree}_{rhs}_{distribution}'
    command = [str(executable), str(rows), str(degree), str(rhs), distribution,
               '4', str(rng.randrange(2**31)), '20', 'tuning' if phase == 1 else 'tuning2']
    variants = range(100, 105) if phase == 1 else (102, 103, 105, 106, 107)
    run_process(output, stem, command, ['dispatch', *[f'v{i}' for i in variants]],
                20, 'environment')
    print(f'{index + 1}/{len(jobs)} {stem}', flush=True)
  archive_runs(output)


if __name__ == '__main__': main()
