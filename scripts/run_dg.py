"""Application-specific transport comparison.

The generic library controls in the main campaign do not answer whether a
specialized transport implementation would beat direct execution. This campaign
adds one: a plan that keeps only the distinct dense operators the repeating
element orders produce and runs each shape group as a strided batched SGEMM.
It reuses the transport inputs, physical horizon, and numerical checks of the
main campaign, so only the comparator set differs.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import time

from benchmark_runs import (ROOT, archive_runs, read_runs, run_process, save_manifest,
                            source_hashes, source_paths, validate_run)
from build_verified import verified_build, validate_build_receipt
from prepare_transport import INPUT_DIRECTORY, transport_steps

METHODS = ['direct', 'dg_specialized', 'grouped', 'bsr8']
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
    if len(fields) != 5:
        raise ValueError('Missing specialized structure record')
    distinct, launches, operator_bytes, assembled_bytes, storage_bytes = fields
    if (not 0 < distinct < launches or operator_bytes <= 0
            or assembled_bytes <= operator_bytes or storage_bytes < operator_bytes):
        raise ValueError('Implausible specialized structure record')
    return dict(distinct=distinct, launches=launches, operator_bytes=operator_bytes,
                assembled_bytes=assembled_bytes, storage_bytes=storage_bytes)


def analyze(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    validate_build_receipt(manifest)
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
        summary.append(dict(
            elements=elements, rhs=rhs, steps=selected[0]['steps'],
            direct_ms=statistics.median(r['medians']['direct'] for r in selected),
            specialized_ms=statistics.median(r['medians']['dg_specialized'] for r in selected),
            specialized=statistics.median(ratios), specialized_min=ratios[0],
            specialized_max=ratios[-1],
            wins=sum(r['specialized'] > 1 for r in selected),
            best_generic=statistics.median(r['best_generic'] for r in selected),
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
                    sources=source_hashes(sources), build_receipt=receipt)
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
