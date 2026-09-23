"""Application-specific transport comparison.

The generic library controls in the main campaign do not answer whether a
specialized transport implementation would beat direct execution. This campaign
adds one: a plan that keeps only the distinct dense operators the repeating
element orders produce and runs each shape group as one batched SGEMM whose
operator pointer array repeats that cached block. It reuses the transport
inputs, physical horizon, and numerical checks of the main campaign, so only
the comparator set differs. The archive records the content hashes of those
inputs, so the reuse is checkable rather than asserted.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import time
import zipfile

from benchmark_runs import (ROOT, archive_runs, read_runs, run_process, save_manifest,
                            source_hashes, source_paths, validate_run)
from build_verified import verified_build, validate_build_receipt
from prepare_transport import INPUT_DIRECTORY, transport_steps

METHODS = ['direct', 'dg_specialized', 'dg_fused', 'dg_fused_copies', 'grouped', 'bsr8']
ELEMENTS = (256, 1024, 4096)
WIDTHS = (8, 64)
PROCESSES = 3
REPS = 6
ORDER_SEED = 20260912


def jobs():
  result = [(elements, rhs, process)
            for elements, rhs, process in itertools.product(ELEMENTS, WIDTHS, range(PROCESSES))]
  rng = random.Random(ORDER_SEED)
  rng.shuffle(result)
  return [(e, n, p, rng.randrange(1, 2**31)) for e, n, p in result]


def stem(elements, rhs, process):
  return f'dg_e{elements}_n{rhs}_p{process}'


def arguments(elements, rhs, order):
  base = (INPUT_DIRECTORY / f'transport_e{elements}_n{rhs}').as_posix()
  return [base, str(rhs), str(transport_steps(elements)), str(REPS), str(order)]


def structure(metadata):
  """Parse the one structure line the executable prints before timing."""
  lines = [l for l in metadata['environment'].splitlines() if l.startswith('specialized_structure,')]
  if len(lines) != 1:
    raise ValueError('Missing specialized structure record')
  fields = [int(x) for x in lines[0].split(',')[1:]]
  if len(fields) != 6:
    raise ValueError('Missing specialized structure record')
  distinct, launches, operator_bytes, assembled_bytes, storage_bytes, trace_bytes = fields
  # The trace keeps one plan per buffer direction, so its allocation is
  # twice a single plan. Anything less would mean the plans share state.
  if (not 0 < distinct < launches or operator_bytes <= 0
          or assembled_bytes <= operator_bytes or storage_bytes < operator_bytes
          or trace_bytes != 2 * storage_bytes):
    raise ValueError('Implausible specialized structure record')
  return dict(distinct=distinct, launches=launches, operator_bytes=operator_bytes,
              assembled_bytes=assembled_bytes, storage_bytes=storage_bytes,
              trace_bytes=trace_bytes)


def input_paths(root=ROOT):
  """Every transport payload the campaign reads, in a stable order."""
  names = set()
  for elements, rhs, _, _ in jobs():
    base = f'transport_e{elements}_n{rhs}'
    for suffix in ('.bin', '.input', '.reference', '.exact', '.weights', '.json'):
      names.add((INPUT_DIRECTORY / (base + suffix)).as_posix())
  return sorted(names)


def input_hashes(root=ROOT):
  return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
          for name in input_paths(root)}


def validate_snapshot(folder, manifest):
  """The archive must carry the sources and inputs it claims."""
  with zipfile.ZipFile(folder / 'source_snapshot.zip') as archive:
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(manifest['sources']):
      raise ValueError('Snapshot coverage mismatch')
    for name, digest in manifest['sources'].items():
      if hashlib.sha256(archive.read(name)).hexdigest() != digest:
        raise ValueError('Snapshot checksum mismatch')
  recorded = manifest.get('inputs')
  if not recorded or set(recorded) != set(input_paths()):
    raise ValueError('Input coverage mismatch')
  for name, digest in recorded.items():
    payload = ROOT / name
    if not payload.is_file():
      raise ValueError('Missing transport input payload: ' + name)
    if hashlib.sha256(payload.read_bytes()).hexdigest() != digest:
      raise ValueError('Transport input payload checksum mismatch: ' + name)
  # Bind this campaign to the inputs the main transport campaign validated,
  # so its claim to reuse them does not rest on the shared path alone.
  canonical = ROOT / 'data/relevance-v2/manifest.json'
  if canonical.is_file():
    published = json.loads(canonical.read_text()).get('inputs', {})
    shared = {name: digest for name, digest in recorded.items() if name in published}
    if len(shared) != len(recorded):
      raise ValueError('Transport inputs are not the archived campaign inputs')
    for name, digest in shared.items():
      if published[name] != digest:
        raise ValueError('Transport input differs from the archived campaign: ' + name)


def analyze(folder):
  manifest = json.loads((folder / 'manifest.json').read_text())
  validate_build_receipt(manifest)
  validate_snapshot(folder, manifest)
  if manifest['jobs'] != [list(job) for job in jobs()] or manifest['reps'] != REPS:
    raise ValueError('Transport comparison design mismatch')
  if manifest['methods'] != METHODS:
    raise ValueError('Comparator set mismatch')
  expected = {stem(e, n, p): (e, n, arguments(e, n, o)) for e, n, p, o in jobs()}
  records = list(read_runs(folder))
  if {r.stem for r in records} != set(expected):
    raise ValueError('Transport comparison coverage mismatch')
  rows = []
  for record in records:
    elements, rhs, command = expected[record.stem]
    validate_run(record, METHODS, REPS)
    if record.metadata['command'][1:] != command:
      raise ValueError('Command differs from design')
    if record.metadata['executable_sha256'] != manifest['executable_sha256']:
      raise ValueError('Executable mismatch')
    log = record.metadata['environment']
    if 'identity_negative_control,rejected' not in log:
      raise ValueError('Missing negative control')
    # The shared harness validates each method three times: after the
    # untimed startup trace, after its warmup, and after its timed samples.
    errors = [float(l.split(',')[1]) for l in log.splitlines()
              if l.startswith('analytic_relative_l2,')]
    if len(errors) != len(METHODS) * 3 or any(not 0 <= e <= 2e-3 for e in errors):
      raise ValueError('Missing or failed GPU analytic checks')
    medians = {m: statistics.median(float(t['gpu_ms']) for t in record.rows()
                                    if t['method'] == m) for m in METHODS}
    rows.append(dict(elements=elements, rhs=rhs, steps=transport_steps(elements),
                     medians=medians, structure=structure(record.metadata),
                     specialized=medians['dg_specialized'] / medians['direct'],
                     fused=medians['dg_fused'] / medians['direct'],
                     fused_copies=medians['dg_fused_copies'] / medians['direct'],
                     sharing=medians['dg_fused_copies'] / medians['dg_fused'],
                     best_generic=min(medians[m] for m in ('grouped', 'bsr8')) / medians['direct']))
  shapes = {(r['elements'], r['rhs']) for r in rows}
  summary = []
  for elements, rhs in sorted(shapes):
    selected = [r for r in rows if r['elements'] == elements and r['rhs'] == rhs]
    if len(selected) != PROCESSES:
      raise ValueError('Process coverage mismatch')
    if len({(r['structure']['distinct'], r['structure']['operator_bytes']) for r in selected}) != 1:
      raise ValueError('Structure disagreement across processes')
    ratios = sorted(r['specialized'] for r in selected)
    fused_ratios = sorted(r['fused'] for r in selected)
    sharing_ratios = sorted(r['sharing'] for r in selected)
    # Reduce trials by median within each process, then give processes
    # equal weight in log space, matching the manuscript methodology.
    summary.append(dict(
        elements=elements, rhs=rhs, steps=selected[0]['steps'],
        direct_ms=statistics.geometric_mean(r['medians']['direct'] for r in selected),
        specialized_ms=statistics.geometric_mean(r['medians']['dg_specialized'] for r in selected),
        specialized=statistics.geometric_mean(ratios), specialized_min=ratios[0],
        specialized_max=ratios[-1],
        wins=sum(r['specialized'] > 1 for r in selected),
        fused_ms=statistics.geometric_mean(r['medians']['dg_fused'] for r in selected),
        fused=statistics.geometric_mean(fused_ratios), fused_min=fused_ratios[0],
        fused_max=fused_ratios[-1],
        fused_wins=sum(r['fused'] > 1 for r in selected),
        fused_copies=statistics.geometric_mean(r['fused_copies'] for r in selected),
        sharing=statistics.geometric_mean(sharing_ratios), sharing_min=sharing_ratios[0],
        sharing_max=sharing_ratios[-1],
        best_generic=statistics.geometric_mean(r['best_generic'] for r in selected),
        **selected[0]['structure']))
  output = dict(processes=len(rows), methods=METHODS, summary=summary, records=rows)
  (folder / 'summary.json').write_text(json.dumps(output, indent=2))
  print(json.dumps(summary, indent=2), flush=True)
  return output


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--output', type=Path, default=ROOT / 'data/dg')
  parser.add_argument('--analyze-only', action='store_true')
  args = parser.parse_args()
  if args.analyze_only:
    analyze(args.output)
    return
  executable, receipt = verified_build('vbsr_dg')
  args.output.mkdir(parents=True, exist_ok=True)
  sources = source_paths('run_dg.py')
  manifest = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  jobs=jobs(), reps=REPS, methods=METHODS,
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  sources=source_hashes(sources), inputs=input_hashes(),
                  build_receipt=receipt)
  save_manifest(args.output, manifest, sources)
  existing = {r.stem: r for r in read_runs(args.output)}
  start = time.monotonic()
  for index, (elements, rhs, process, order) in enumerate(jobs()):
    name = stem(elements, rhs, process)
    command = [str(executable), *arguments(elements, rhs, order)]
    if name in existing:
      validate_run(existing[name], METHODS, REPS)
      if existing[name].metadata['command'] != command:
        raise ValueError('Resume command mismatch')
      continue
    run_process(args.output, name, command, METHODS, REPS, 'environment')
    print(f'{index + 1}/{len(jobs())} processes, {time.monotonic() - start:.0f}s elapsed',
          flush=True)
  if not (args.output / 'runs.zip').exists():
    archive_runs(args.output)
  analyze(args.output)


if __name__ == '__main__':
  main()
