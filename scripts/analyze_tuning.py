"""Validate the archived seed-4 narrow-kernel development comparisons."""
import hashlib
import itertools
import json
import random
import statistics
import zipfile
from benchmark_runs import ROOT, read_runs, validate_run
from build_verified import validate_build_receipt
from analyze_revision import geomean


def main():
    summary = {}
    for phase, folder in ((1, 'narrow-tuning'), (2, 'narrow-tuning-2')):
        path = ROOT / 'data' / folder
        manifest = json.loads((path / 'manifest.json').read_text())
        validate_build_receipt(manifest)
        with zipfile.ZipFile(path / 'source_snapshot.zip') as archive:
            if set(archive.namelist()) != set(manifest['sources']):
                raise ValueError('Tuning source coverage mismatch')
            for name, digest in manifest['sources'].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError('Tuning source checksum mismatch')
        jobs = list(itertools.product((256, 1024, 4096), (2, 16), (8, 16),
                                      ('uniform', 'low', 'high', 'bimodal')))
        rng = random.Random(20260908)
        rng.shuffle(jobs)
        if manifest['jobs'] != [list(j) for j in jobs] or manifest['matrix_seed'] != 4:
            raise ValueError('Tuning design mismatch')
        variants = list(range(100, 105)) if phase == 1 else [102, 103, 105, 106, 107]
        methods = ['dispatch', *[f'v{i}' for i in variants]]
        records = {r.stem: r for r in read_runs(path)}
        if len(records) != len(jobs):
            raise ValueError('Tuning run coverage mismatch')
        timings = []
        for rows, degree, rhs, dist in jobs:
            record = records[f'{rows}_{degree}_{rhs}_{dist}']
            validate_run(record, methods, 20)
            expected = [str(rows), str(degree), str(rhs), dist, '4', str(rng.randrange(2**31)),
                        '20', 'tuning' if phase == 1 else 'tuning2']
            if (record.metadata['command'][1:] != expected or
                    record.metadata['executable_sha256'] != manifest['executable_sha256']):
                raise ValueError('Tuning command or executable mismatch')
            values = {m: statistics.median(float(r['gpu_ms']) for r in record.rows() if r['method'] == m)
                      for m in methods}
            timings.append(dict(rows=rows, degree=degree, rhs=rhs, distribution=dist, medians=values))
        summary[folder] = dict(records=timings, ratios={str(rhs): {
            method: geomean(t['medians']['dispatch'] / t['medians'][method]
                            for t in timings if t['rhs'] == rhs) for method in methods[1:]}
            for rhs in (8, 16)})
    output = ROOT / 'research/generated/tuning-summary.json'
    output.write_text(json.dumps(summary, indent=2))
    print('Validated 96 development processes on seed 4')


if __name__ == '__main__': main()
