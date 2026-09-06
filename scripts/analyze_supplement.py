"""Validate specified supplementary designs and generate their paper tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import zipfile
from benchmark_runs import ROOT, read_runs, validate_run
from build_verified import validate_build_receipt
from run_supplement import jobs_for, specification
from analyze_revision import library_digest


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def table(output, name, columns, header, rows):
    text = ['\\begin{tabular}{' + columns + '}', '\\toprule', ' & '.join(header) + r' \\', '\\midrule']
    text += [' & '.join(map(str, row)) + r' \\' for row in rows]
    text += ['\\bottomrule', '\\end{tabular}']
    (output / (name + '.tex')).write_text('\n'.join(text) + '\n')


def read_campaign(directory, kind):
    manifest = json.loads((directory / 'manifest.json').read_text())
    validate_build_receipt(manifest)
    with zipfile.ZipFile(directory / 'source_snapshot.zip') as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(manifest['sources']):
            raise ValueError('Source archive coverage mismatch')
        for path, digest in manifest['sources'].items():
            if hashlib.sha256(archive.read(path)).hexdigest() != digest:
                raise ValueError('Source checksum mismatch')
    for path, digest in manifest['inputs'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError('Native input checksum mismatch')
    jobs = jobs_for(kind)
    rng = random.Random(20260907)
    rng.shuffle(jobs)
    if manifest['jobs'] != [list(j) for j in jobs] or manifest['kind'] != kind or manifest['reps'] != 20:
        raise ValueError('Supplementary design mismatch')
    records = {r.stem: r for r in read_runs(directory)}
    expected_stems = set()
    result = {}
    for job in jobs:
        stem, command, methods = specification(kind, job, 'unused', rng.randrange(2**31),
                                                manifest.get('ablation_revision', 1))
        expected_stems.add(stem)
        record = records[stem]
        validate_run(record, methods, 20)
        actual = record.metadata['command']
        if kind == 'native':
            matches = Path(actual[1]).name == Path(command[1]).name and actual[2:] == command[2:]
        else:
            matches = actual[1:] == command[1:]
        if not matches or record.metadata.get('executable_sha256') != manifest['executable_sha256']:
            raise ValueError('Command or executable mismatch')
        timings = {m: {clock: statistics.median(float(r[clock + '_ms']) for r in record.rows() if r['method'] == m)
                       for clock in ('gpu', 'host')} for m in methods}
        setup = {}
        if kind == 'native':
            for line in record.metadata['environment'].splitlines():
                if line.startswith('setup,'):
                    _, method, ms, size = line.split(',')
                    if method in setup or not math.isfinite(float(ms)) or float(ms) <= 0 or int(size) <= 0:
                        raise ValueError('Invalid startup record')
                    setup[method] = dict(ms=float(ms), bytes=int(size))
            if set(setup) != set(methods):
                raise ValueError('Missing startup coverage')
        result[job] = dict(timings=timings, setup=setup)
    if set(records) != expected_stems:
        raise ValueError('Unexpected supplementary runs')
    return result, manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--native', type=Path, default=ROOT / 'data/product-evaluation/native')
    parser.add_argument('--ablation', type=Path, default=ROOT / 'data/product-evaluation/ablation')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/generated')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native, native_manifest = read_campaign(args.native, 'native')
    ablation, ablation_manifest = read_campaign(args.ablation, 'ablation')
    if library_digest(native_manifest) != library_digest(ablation_manifest):
        raise ValueError('Supplementary library source mismatch')
    rows = []
    native_ratios = []
    for size in (2048, 8192):
        for geometry in ('uniform', 'clustered'):
            for rhs in (8, 16, 32, 64):
                subset = [r for (name, width, p), r in native.items() if
                          name.startswith(f'covariance_{size}_{geometry}_') and width == rhs]
                ratios = [[(min(t[m]['gpu'] for m in ('compact_alg1_pre', 'compact_alg2', 'compact_alg3_pre'))
                            if method == 'csr' else t[method]['gpu']) / t['direct']['gpu']
                           for t in (r['timings'] for r in subset)] for method in ('csr', 'bsr8', 'grouped_cached', 'dense')]
                rows.append([size, geometry, rhs, *[f'{geo(v):.3f}' for v in ratios]])
                native_ratios.extend(ratios[1])
    table(args.output, 'native', 'rlrrrrr', ['Points', 'Geometry', 'RHS', 'CSR', 'BSR8', 'Grouped', 'Dense'], rows)
    shape_rows = []
    for size in (2048, 8192):
        for geometry in ('uniform', 'clustered'):
            meta = [json.loads((ROOT / 'data/native' / f'covariance_{size}_{geometry}_s{s}.json').read_text()) for s in (1, 2, 3)]
            heights = sorted({int(h) for m in meta for h in m['shape_counts']})
            shape_rows.append([size, geometry, ', '.join(map(str, heights)),
                               f"{min(m['block_rows'] for m in meta)} to {max(m['block_rows'] for m in meta)}",
                               f"{min(m['mean_degree'] for m in meta):.1f} to {max(m['mean_degree'] for m in meta):.1f}",
                               f"{100*min(m['packed_values']/size**2 for m in meta):.1f} to {100*max(m['packed_values']/size**2 for m in meta):.1f}"])
    table(args.output, 'native-shapes', 'rlrrrr', ['Points', 'Geometry', 'Leaf sizes', 'Block rows', 'Mean degree', r'Density \%'], shape_rows)
    # Startup includes construction, transfer, lazy library setup and one product.
    # k-1 synchronized host medians model subsequent products. No timing is
    # subtracted, and no GPU-event duration is mixed into this host cost model.
    lifecycle = []
    lifecycle_summary = {}
    for products in (1, 10, 100, 1000):
        ratios, with_assembly = [], []
        for (name, rhs, process), record in native.items():
            meta = json.loads((ROOT / 'data/native' / (name + '.json')).read_text())
            totals = {m: s['ms'] + (products - 1) * record['timings'][m]['host'] for m, s in record['setup'].items()}
            ratios.append(min(v for m, v in totals.items() if m != 'direct') / totals['direct'])
            complete = {m: v + (meta['geometry_ms'] + meta['scalar_assembly_ms'] if m.startswith('compact') or m == 'dense'
                                else meta['generation_and_packing_ms']) for m, v in totals.items()}
            with_assembly.append(min(v for m, v in complete.items() if m != 'direct') / complete['direct'])
        lifecycle.append([products, f'{geo(ratios):.3f}', f'{geo(with_assembly):.3f}'])
        lifecycle_summary[str(products)] = dict(prepared=geo(ratios), with_assembly=geo(with_assembly))
    table(args.output, 'lifecycle', 'rrr', ['Products', 'Prepared-host ratio', 'Including assembly ratio'], lifecycle)
    storage_rows = []
    for method in ('direct', 'compact_alg1_pre', 'compact_alg2', 'compact_alg3_pre', 'bsr8', 'grouped_cached', 'dense'):
        values = [r['setup'][method] for r in native.values()]
        storage_rows.append([method.replace('_', r'\_'), f"{min(v['bytes'] for v in values)/2**20:.2f}",
                             f"{max(v['bytes'] for v in values)/2**20:.2f}", f"{geo(v['ms'] for v in values):.3f}"])
    table(args.output, 'storage', 'lrrr', ['Method', 'Min MiB', 'Max MiB', 'Startup ms'], storage_rows)
    rows, contrasts = [], {}
    for rhs in (8, 16, 32, 64):
        subset = [r['timings'] for job, r in ablation.items() if job[2] == rhs]
        if rhs == 8:
            comparisons = [('Eight-row mapping', [(i, i + 4) for i in range(4)]),
                           ('Two accumulators', [(i, i + 2) for i in (0, 1, 4, 5)]),
                           ('One CTA instead of two', [(i + 1, i) for i in (0, 2, 4, 6)])]
        else:
            comparisons = [('Wider accumulation', [(0, 1)]), ('Cooperative staging', [(1, 2)]),
                           ('Double buffering', [(3, 4)])]
        if ablation_manifest.get('ablation_revision', 1) >= 2 and rhs in (8, 16):
            comparisons += [('Full-panel staging', [(8, 9)] if rhs == 8 else [(5, 6)])]
        for label, pairs in comparisons:
            ratios = [geo(t[f'v{a}']['gpu'] / t[f'v{b}']['gpu'] for a, b in pairs) for t in subset]
            contrasts[f'{rhs}:{label}'] = dict(ratio=geo(ratios), minimum=min(ratios), maximum=max(ratios))
            # Ranges aggregate the 3 process observations of each fixed instance.
            instances = {}
            for job, record in ablation.items():
                if job[2] != rhs: continue
                t = record['timings']
                instances.setdefault(job[:-1], []).append(geo(t[f'v{a}']['gpu'] / t[f'v{b}']['gpu'] for a, b in pairs))
            instance_ratios = [geo(v) for v in instances.values()]
            rows.append([rhs, label, f'{geo(ratios):.3f}', f'{min(instance_ratios):.3f}', f'{max(instance_ratios):.3f}'])
    table(args.output, 'matched-ablation', 'rlrrr', ['RHS', 'Change', 'Ratio', 'Min', 'Max'], rows)
    upgrades = {}
    if ablation_manifest.get('ablation_revision', 1) >= 2:
        upgrade_rows = []
        for rhs in (8, 16):
            for size in (256, 1024):
                instances = {}
                for job, record in ablation.items():
                    if job[0] == size and job[2] == rhs:
                        t = record['timings']
                        instances.setdefault(job[:-1], []).append(t['v200']['gpu'] / t['dispatch']['gpu'])
                values = [geo(v) for v in instances.values()]
                upgrades[f'{rhs}:{size}'] = dict(ratio=geo(values), minimum=min(values), maximum=max(values))
                upgrade_rows.append([rhs, size, f'{geo(values):.3f}', f'{min(values):.3f}', f'{max(values):.3f}'])
        table(args.output, 'kernel-upgrade', 'rrrrr', ['RHS', 'Block rows', 'Ratio', 'Min', 'Max'], upgrade_rows)
    # Retain per-instance factorial interactions and every method median, so
    # the marginal effects in the compact table can be checked and disaggregated.
    result = dict(native_processes=len(native), ablation_processes=len(ablation),
                  native_bsr_ratio=geo(native_ratios), contrasts=contrasts, upgrades=upgrades,
                  lifecycle=lifecycle_summary,
                  native=[dict(job=list(j), **r) for j, r in sorted(native.items())],
                  ablation=[dict(job=list(j), **r) for j, r in sorted(ablation.items())],
                  kernel_sha256=native_manifest['sources']['src/kernels/row_owned.cu'],
                  library_sha256=library_digest(native_manifest))
    (args.output / 'supplement-summary.json').write_text(json.dumps(result, indent=2))
    macro_values = dict(NativeBsrRatio=geo(native_ratios),
                        NativeStartupRatio=lifecycle_summary['1']['prepared'],
                        NativeTenProductsRatio=lifecycle_summary['10']['prepared'])
    macro_values['NativeCsrStorageRatio'] = geo(
        min(r['setup'][m]['bytes'] for m in ('compact_alg1_pre', 'compact_alg2', 'compact_alg3_pre')) /
        r['setup']['direct']['bytes'] for r in native.values())
    if upgrades:
        for rhs, name in ((8, 'RhsEightUpgrade'), (16, 'RhsSixteenUpgrade')):
            macro_values[name] = geo(v['ratio'] for key, v in upgrades.items() if key.startswith(f'{rhs}:'))
    (args.output / 'supplement-macros.tex').write_text(''.join(
        '\\newcommand{\\' + name + '}{' + f'{value:.3f}' + '}\n' for name, value in macro_values.items()))
    print(json.dumps({k: v for k, v in result.items() if k not in ('native', 'ablation')}, indent=2))


if __name__ == '__main__':
    main()
