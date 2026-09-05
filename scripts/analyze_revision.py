"""Validate raw measurements and generate every revised result table."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

from benchmark_runs import read_runs, validate_run

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / 'research/generated'
PATTERN = re.compile(r'(\d+)_(\d+)_(\d+)_(uniform|low|high|bimodal)_(local|random)_s(\d+)_i([01])_p(\d+)')


def geomean(values):
    values = list(values)
    if not values or not all(x > 0 and math.isfinite(x) for x in values):
        raise ValueError('Invalid ratio population')
    return math.exp(statistics.mean(map(math.log, values)))


def read_timings(path, methods):
    validate_run(path, methods, 20)
    rows = path.rows()
    return {
        method: {
            clock: statistics.median(float(row[clock + '_ms']) for row in rows if row['method'] == method)
            for clock in ('gpu', 'host')
        }
        for method in methods
    }


def table(filename, header, rows, columns):
    text = '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += ' & '.join(header) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (GENERATED / filename).write_text(text, encoding='utf-8', newline='\n')


def ratio(methods, comparator, clock='gpu'):
    direct = 'direct_changing' if comparator == 'grouped_changing' else 'direct'
    if comparator == 'csr_best':
        numerator = min(methods[x][clock] for x in ('csr_default', 'csr_alg1_pre', 'csr_alg2', 'csr_alg3_pre'))
    elif comparator == 'compact_best':
        numerator = min(methods[x][clock] for x in ('compact_alg1_pre', 'compact_alg2', 'compact_alg3_pre'))
    else:
        numerator = methods[comparator][clock]
    return numerator / methods[direct][clock]


def summarize(records, comparator, clock='gpu'):
    # Equal configuration weights, then equal weights for the three process medians.
    per_case = {key: geomean(ratio(methods, comparator, clock) for methods in processes.values())
                for key, processes in records.items()}
    per_process = [geomean(ratio(processes[p], comparator, clock) for processes in records.values()) for p in range(3)]
    return dict(geomean=geomean(per_case.values()), minimum=min(per_case.values()),
                wins=sum(x > 1 for x in per_case.values()), count=len(per_case),
                process_range=[min(per_process), max(per_process)])


def main():
    global GENERATED
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=Path, default=ROOT / 'data/optimized-revision')
    parser.add_argument('--application', type=Path, default=ROOT / 'data/application/optimized-results')
    parser.add_argument('--output', type=Path, default=GENERATED)
    args = parser.parse_args()
    GENERATED = args.output
    GENERATED.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.revision / 'manifest.json').read_text())
    manifest['sources'] = {path.replace('\\', '/'): digest for path, digest in manifest['sources'].items()}
    with zipfile.ZipFile(args.revision / 'source_snapshot.zip') as snapshot:
        for path, digest in manifest['sources'].items():
            if hashlib.sha256(snapshot.read(path.replace('\\', '/'))).hexdigest() != digest:
                raise ValueError('Source snapshot hash mismatch')
    records = {}
    for path in read_runs(args.revision):
        match = PATTERN.fullmatch(path.stem)
        if not match:
            raise ValueError(f'Unrecognized result file {path}')
        rows, degree, rhs, dist, locality, seed, irregular, process = match.groups()
        key = (int(rows), int(degree), int(rhs), dist, locality, int(seed), int(irregular))
        methods = ['direct', 'direct_changing', 'csr_default', 'csr_alg1_pre', 'csr_alg2',
                   'csr_alg3_pre', 'grouped_cached', 'grouped_changing']
        if dist == 'uniform':
            methods.append('bsr32')
        records.setdefault(key, {})[int(process)] = read_timings(path, methods)
    expected = {tuple(case) for case, process in manifest['jobs']}
    if set(records) != expected or any(set(p) != {0, 1, 2} for p in records.values()):
        raise ValueError(f'Incomplete revision campaign, found {sum(map(len, records.values()))}/504 process records')
    core = {k: p for k, p in records.items() if k[0] == 4096 and k[6] == 0}
    if len(core) != 128:
        raise ValueError('Core grid is not 128 configurations')
    names = [('csr_default', 'CSR default'), ('csr_best', 'CSR lower envelope'),
             ('grouped_cached', 'Grouped cached'), ('grouped_changing', 'Grouped changing')]
    summary = {name: summarize(core, name) for name, label in names}
    table('core.tex', ['Comparator', 'GPU ratio', 'Host ratio', 'Minimum', 'Wins'],
          [[label, f"{summary[name]['geomean']:.3f}", f"{summarize(core, name, 'host')['geomean']:.3f}",
            f"{summary[name]['minimum']:.3f}", f"{summary[name]['wins']}/128"] for name, label in names], 'lrrrr')
    table('process.tex', ['Comparator', 'Geometric mean', 'Process-label range'],
          [[label, f"{summary[name]['geomean']:.3f}",
            f"{summary[name]['process_range'][0]:.3f} to {summary[name]['process_range'][1]:.3f}"]
           for name, label in names], 'lrr')
    rows = []
    for rhs in (8, 16, 32, 64):
        for degree in (2, 4, 8, 16):
            subset = {k: p for k, p in core.items() if k[1] == degree and k[2] == rhs}
            direct_ms = geomean(m['direct']['gpu'] for p in subset.values() for m in p.values())
            rows.append([rhs, degree, f'{direct_ms:.3f}', f"{summarize(subset, 'csr_best')['geomean']:.3f}",
                         f"{summarize(subset, 'grouped_cached')['geomean']:.3f}"])
    table('regimes.tex', ['RHS', 'Degree', 'Direct ms', 'CSR ratio', 'Grouped ratio'], rows, 'rrrrr')
    rows = []
    for size in (256, 1024, 4096):
        for irregular in (0, 1):
            subset = {k: p for k, p in records.items() if k[0] == size and k[1] == 16 and k[3] in ('high', 'bimodal')
                      and k[4] == 'random' and k[6] == irregular}
            rows.append([size, '0, 1, 16, 16' if irregular else '16 in every row',
                         f"{summarize(subset, 'csr_best')['geomean']:.3f}",
                         f"{summarize(subset, 'grouped_cached')['geomean']:.3f}",
                         f"{summarize(subset, 'csr_best')['wins']}/8"])
    table('scaling.tex', ['Block rows', 'Degree pattern', 'CSR ratio', 'Grouped ratio', 'CSR wins'], rows, 'rlrrr')
    uniform = {k: p for k, p in core.items() if k[3] == 'uniform'}
    summary['bsr32'] = summarize(uniform, 'bsr32')
    rows = []
    for rhs in (8, 16, 32, 64):
        subset = {k: p for k, p in uniform.items() if k[2] == rhs}
        result = summarize(subset, 'bsr32')
        rows.append([rhs, f"{result['geomean']:.3f}", f"{result['minimum']:.3f}", f"{result['wins']}/8"])
    table('bsr.tex', ['RHS', 'BSR ratio', 'Minimum', 'Wins'], rows, 'rrrr')

    application = {}
    app_manifest = json.loads((args.application / 'manifest.json').read_text())
    app_manifest['sources'] = {path.replace('\\', '/'): digest for path, digest in app_manifest['sources'].items()}
    with zipfile.ZipFile(args.application / 'source_snapshot.zip') as snapshot:
        for path, digest in app_manifest['sources'].items():
            if hashlib.sha256(snapshot.read(path.replace('\\', '/'))).hexdigest() != digest:
                raise ValueError('Application source snapshot hash mismatch')
    for path, digest in app_manifest['inputs'].items():
        if hashlib.sha256((ROOT / 'data/application' / path).read_bytes()).hexdigest() != digest:
            raise ValueError('Application input checksum mismatch')
    for path in read_runs(args.application):
        match = re.fullmatch(r'(bcsstk\d+)_rhs(\d+)_p(\d+)', path.stem)
        if not match:
            raise ValueError('Invalid application filename')
        name, rhs, process = match.groups()
        application.setdefault((name, int(rhs)), {})[int(process)] = read_timings(path,
            ['direct', 'compact_alg1_pre', 'compact_alg2', 'compact_alg3_pre', 'grouped_cached'])
    if len(application) != 12 or any(set(p) != {0, 1, 2} for p in application.values()):
        raise ValueError('Incomplete application results')
    rows = []
    for number in (13, 14, 15):
        name = f'bcsstk{number}'
        meta = json.loads((ROOT / f'data/application/{name}.json').read_text())
        rows.append([name, f"{100 * meta['fill_fraction']:.1f}\\%", *[
            f"{summarize({(name, rhs): application[name, rhs]}, 'compact_best')['geomean']:.3f}" for rhs in (8, 16, 32, 64)]])
    table('application.tex', ['Matrix', 'Block fill', 'RHS 8', 'RHS 16', 'RHS 32', 'RHS 64'], rows, 'lrrrrr')
    summary['application'] = summarize(application, 'compact_best')
    summary['application_grouped'] = summarize(application, 'grouped_cached')
    summary['core_configurations'] = len(core)
    summary['total_configurations'] = len(records)
    summary['raw_trials'] = sum((9 if k[3] == 'uniform' else 8) * 20 * 3 for k in records) + 12 * 5 * 20 * 3
    summary['matrix_seed'] = 1
    summary['provenance'] = {
        'synthetic_directory': args.revision.as_posix(),
        'application_directory': args.application.as_posix(),
        'synthetic_executable_sha256': manifest['executable_sha256'],
        'application_executable_sha256': app_manifest['executable_sha256'],
        'kernel_sha256': manifest['sources']['src/kernels/row_owned.cu'],
    }
    if manifest['sources']['src/kernels/row_owned.cu'] != app_manifest['sources']['src/kernels/row_owned.cu']:
        raise ValueError('Synthetic and application campaigns used different kernels')
    (GENERATED / 'summary.json').write_text(json.dumps(summary, indent=2), newline='\n')
    commands = [('AuditCsr', summary['csr_best']['geomean']), ('AuditGrouped', summary['grouped_cached']['geomean']),
                ('AuditBsr', summary['bsr32']['geomean']), ('AuditApplication', summary['application']['geomean']),
                ('AuditDefaultCsr', summary['csr_default']['geomean']),
                ('AuditApplicationGrouped', summary['application_grouped']['geomean']),
                ('AuditApplicationGroupedLow', summary['application_grouped']['process_range'][0]),
                ('AuditApplicationGroupedHigh', summary['application_grouped']['process_range'][1]),
                ('AuditCompactSpeedup', 1 / summary['application']['geomean'])]
    text = ''.join('\\newcommand{\\' + name + '}{' + f'{value:.3f}' + '}\n' for name, value in commands)
    for command, key in [('AuditCsrWins', 'csr_best'), ('AuditGroupedWins', 'grouped_cached'),
                         ('AuditBsrWins', 'bsr32'), ('AuditApplicationWins', 'application')]:
        text += '\\newcommand{\\' + command + '}{' + str(summary[key]['wins']) + '}\n'
    text += '\\newcommand{\\AuditBsrLosses}{' + str(32 - summary['bsr32']['wins']) + '}\n'
    text += '\\newcommand{\\AuditApplicationLosses}{' + str(12 - summary['application']['wins']) + '}\n'
    (GENERATED / 'macros.tex').write_text(text, newline='\n')
    from plot_revision import render_figures
    chart_data = {
        'provenance': summary['provenance'],
        'core': [dict(configuration=list(key), **{
            name: geomean(ratio(methods, name) for methods in processes.values())
            for name in ('csr_default', 'csr_best', 'grouped_cached')})
            for key, processes in sorted(core.items())],
        'uniform': [dict(configuration=list(key), rhs=key[2],
                         ratio=geomean(ratio(methods, 'bsr32') for methods in processes.values()))
                    for key, processes in sorted(uniform.items())],
        'application': [dict(matrix=key[0], rhs=key[1],
                             ratio=geomean(ratio(methods, 'compact_best') for methods in processes.values()))
                        for key, processes in sorted(application.items())],
    }
    (GENERATED / 'figure-data.json').write_text(json.dumps(chart_data, indent=2), newline='\n')
    render_figures(GENERATED, chart_data)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
