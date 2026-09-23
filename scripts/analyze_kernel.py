"""Validate the RHS-8 campaign and generate manuscript numbers from raw trials."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics
import zipfile
from build_verified import validate_build_receipt

ROOT = Path(__file__).resolve().parents[1]


def geomean(values):
  return math.exp(statistics.mean(map(math.log, values)))


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--input', type=Path, default=ROOT / 'data/kernel-rhs8-final-20260906')
  parser.add_argument('--output', type=Path, default=ROOT / 'research/generated')
  args = parser.parse_args()
  manifest = json.loads((args.input / 'manifest.json').read_text())
  if 'build_receipt' in manifest:
    validate_build_receipt(manifest)
  with zipfile.ZipFile(args.input / 'source_snapshot.zip') as snapshot:
    for name, digest in manifest['sources'].items():
      if hashlib.sha256(snapshot.read(name.replace('\\', '/'))).hexdigest() != digest:
        raise ValueError('Source checksum mismatch: ' + name)
  cases = set(itertools.product((256, 512, 1024, 4096), (2, 4, 8, 16),
                                ('uniform', 'low', 'high', 'bimodal'), ('local', 'random'), (0,)))
  cases.update(itertools.product((256, 512, 1024, 4096), (16,),
                                 ('high', 'bimodal'), ('local', 'random'), (1,)))
  jobs = {(case, seed) for case in cases for seed in (1, 2, 3)}
  if manifest['reps'] != 30 or {(tuple(c), s) for c, s in manifest['jobs']} != jobs:
    raise ValueError('Unexpected campaign design')
  if len(manifest['jobs']) != len(jobs):
    raise ValueError('Duplicate manifest jobs')
  records = {}
  with zipfile.ZipFile(args.input / 'runs.zip') as archive:
    hashes = json.loads(archive.read('checksums.json'))
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(hashes) | {'checksums.json'}:
      raise ValueError('Invalid archive entries')
    for name, digest in hashes.items():
      if hashlib.sha256(archive.read(name)).hexdigest() != digest:
        raise ValueError('Run checksum mismatch: ' + name)
    expected_names = set()
    for case, seed in sorted(jobs):
      size, degree, distribution, locality, irregular = case
      stem = f'{size}_{degree}_{distribution}_{locality}_i{irregular}_s{seed}'
      expected_names.update((stem + '.csv', stem + '.json'))
      metadata = json.loads(archive.read(stem + '.json'))
      if 'build_receipt' in manifest and metadata.get('executable_sha256') != manifest['executable_sha256']:
        raise ValueError('Kernel executable checksum mismatch')
      command = metadata['command']
      if (metadata['returncode'] != 0 or len(command) != 9 or
          command[1:6] != [str(size), str(degree), distribution, locality, str(seed)] or
          command[7:] != [str(irregular), '30']):
        raise ValueError('Process metadata mismatch: ' + stem)
      rows = list(csv.DictReader(io.StringIO(archive.read(stem + '.csv').decode())))
      if len(rows) != 60 or {r['method'] for r in rows} != {'legacy', 'tiled'}:
        raise ValueError('Unexpected method coverage')
      medians = {}
      positions = set()
      for method in ('legacy', 'tiled'):
        trials = [r for r in rows if r['method'] == method]
        if sorted(int(r['iteration']) for r in trials) != list(range(30)):
          raise ValueError('Unexpected trial coverage')
        order = {int(r['position']) for r in trials}
        if len(order) != 1:
          raise ValueError('Inconsistent method position')
        positions.update(order)
        if not all(math.isfinite(float(r[k])) and float(r[k]) > 0
                   for r in trials for k in ('gpu_ms', 'host_ms')):
          raise ValueError('Invalid duration')
        medians[method] = statistics.median(float(r['gpu_ms']) for r in trials)
      if positions != {0, 1}:
        raise ValueError('Invalid method order')
      records[case, seed] = medians
    if set(hashes) != expected_names:
      raise ValueError('Unexpected run coverage')
  ratios = {c: geomean(records[c, s]['legacy'] / records[c, s]['tiled'] for s in (1, 2, 3))
            for c in cases}
  active = {c: r for c, r in ratios.items() if c[0] >= 512}
  summary = dict(configurations=len(cases), processes=len(jobs), raw_trials=60 * len(jobs),
                 overall=geomean(ratios.values()), active=geomean(active.values()),
                 active_minimum=min(active.values()), active_wins=sum(r > 1 for r in active.values()),
                 by_size={n: geomean(r for c, r in ratios.items() if c[0] == n)
                          for n in (256, 512, 1024, 4096)})
  args.output.mkdir(parents=True, exist_ok=True)
  table = ['\\begin{tabular}{rlr}', '\\toprule',
           r'Block rows & RHS-8 mapping & Speedup \\', '\\midrule']
  for n, ratio in summary['by_size'].items():
    mapping = 'Original scalar' if n < 512 else 'Eight-row tiles'
    table.append(f'{n:,} & {mapping} & {ratio:.3f}' + r' \\')
  table += ['\\bottomrule', '\\end{tabular}']
  (args.output / 'kernel.tex').write_text('\n'.join(table) + '\n')
  macros = [('KernelSpeedup', summary['active']), ('KernelMinimum', summary['active_minimum']),
            ('KernelOverall', summary['overall'])]
  (args.output / 'kernel-macros.tex').write_text(''.join(
      '\\newcommand{\\' + name + '}{' + f'{value:.3f}' + '}\n' for name, value in macros))
  (args.output / 'kernel-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
