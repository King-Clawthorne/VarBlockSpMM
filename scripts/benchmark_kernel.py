"""Serial RHS-8 release-versus-tiled validation with retained raw timings."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import zipfile
from build_verified import verified_build
from benchmark_runs import source_paths, source_hashes

ROOT = Path(__file__).resolve().parents[1]


def archive_runs(directory):
    directory = directory.resolve()
    paths = sorted(directory.glob('*_s*.csv')) + sorted(directory.glob('*_s*.json'))
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    archive = directory / 'runs.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            output.write(path, path.name)
        output.writestr('checksums.json', json.dumps(checksums, indent=2))
    with zipfile.ZipFile(archive) as saved:
        for name, digest in checksums.items():
            if hashlib.sha256(saved.read(name)).hexdigest() != digest:
                raise ValueError('Archive roundtrip mismatch: ' + name)
    for path in paths:
        if path.resolve().parent != directory:
            raise ValueError('Run file escaped output directory')
        path.unlink()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=30)
    args = parser.parse_args()
    if args.reps < 2:
        parser.error('at least two repetitions required')
    args.output.mkdir(parents=True, exist_ok=False)
    executable, receipt = verified_build('vbsr_kernel_audit')
    cases = [(n, d, s, loc, 0) for n, d, s, loc in itertools.product(
        (256, 512, 1024, 4096), (2, 4, 8, 16), ('uniform', 'low', 'high', 'bimodal'), ('local', 'random'))]
    cases += [(n, 16, s, loc, 1) for n, s, loc in itertools.product(
        (256, 512, 1024, 4096), ('high', 'bimodal'), ('local', 'random'))]
    jobs = [(case, seed) for case in cases for seed in (1, 2, 3)]
    rng = random.Random(20260906)
    rng.shuffle(jobs)
    sources = source_paths('benchmark_kernel.py')
    manifest = dict(reps=args.reps, jobs=jobs,
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    sources=source_hashes(sources), build_receipt=receipt)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.output / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as snapshot:
        for path in sources:
            snapshot.write(path, str(path.relative_to(ROOT)))
    records = []
    for index, (case, seed) in enumerate(jobs):
        size, degree, distribution, locality, irregular = case
        command = [str(executable), str(size), str(degree), distribution, locality,
                   str(seed), str(rng.randrange(2**31)), str(irregular), str(args.reps)]
        run = subprocess.run(command, capture_output=True, text=True)
        stem = f'{size}_{degree}_{distribution}_{locality}_i{irregular}_s{seed}'
        (args.output / f'{stem}.csv').write_text(run.stdout)
        (args.output / f'{stem}.json').write_text(json.dumps(
            dict(command=command, returncode=run.returncode, environment=run.stderr,
                 executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest()), indent=2))
        run.check_returncode()
        rows = list(csv.DictReader(io.StringIO(run.stdout)))
        if len(rows) != 2 * args.reps or {r['method'] for r in rows} != {'legacy', 'tiled'}:
            raise ValueError('Incomplete method coverage')
        medians = {}
        for method in ('legacy', 'tiled'):
            selected = [r for r in rows if r['method'] == method]
            if sorted(int(r['iteration']) for r in selected) != list(range(args.reps)):
                raise ValueError('Incomplete trial coverage')
            times = [float(r['gpu_ms']) for r in selected]
            if not all(float(r[clock]) > 0 and math.isfinite(float(r[clock]))
                       for r in selected for clock in ('gpu_ms', 'host_ms')):
                raise ValueError('Invalid timing')
            medians[method] = statistics.median(times)
        records.append(dict(case=case, seed=seed, **medians, ratio=medians['legacy']/medians['tiled']))
        if (index + 1) % 36 == 0:
            print(f'{index + 1}/{len(jobs)} processes passed', flush=True)
    geo = lambda values: math.exp(statistics.mean(map(math.log, values)))
    ratios = [geo(r['ratio'] for r in records if tuple(r['case']) == case) for case in cases]
    summary = dict(configurations=len(cases), processes=len(jobs),
                   geometric_speedup=geo(ratios), minimum=min(ratios), wins=sum(r > 1 for r in ratios),
                   by_size={size: geo(r['ratio'] for r in records if r['case'][0] == size) for size in (256, 512, 1024, 4096)},
                   records=records)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    archive_runs(args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
