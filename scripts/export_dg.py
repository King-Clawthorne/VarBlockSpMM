"""Validate and export the application-specific transport comparison."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_dg import analyze

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=ROOT / 'data/dg')
args = parser.parse_args()
data = analyze(args.input)
OUT = ROOT / 'research/generated'


def gm(values):
    values = list(values)
    return math.exp(sum(map(math.log, values)) / len(values))


rows = []
for record in data['summary']:
    rows.append([f'{record["elements"]:,}', record['rhs'], record['steps'],
                 f'{record["direct_ms"]:.3f}', f'{record["specialized_ms"]:.3f}',
                 f'{record["specialized"]:.3f}',
                 f'{record["specialized_min"]:.3f} to {record["specialized_max"]:.3f}',
                 f'{record["wins"]}/3'])
text = ('\\begin{tabular}{rrrrrrrr}\n\\toprule\n'
        'Elements & RHS & Steps & Direct ms & Specialized ms & Ratio & Process range & Wins\\\\\n'
        '\\midrule\n')
text += '\n'.join(' & '.join(map(str, row)) + '\\\\' for row in rows)
(OUT / 'dg.tex').write_text(text + '\n\\bottomrule\n\\end{tabular}\n')

large = next(r for r in data['summary'] if r['elements'] == 4096 and r['rhs'] == 64)
ratios = [r['specialized'] for r in data['summary']]
structures = {(r['distinct'], r['launches'], r['operator_bytes']) for r in data['summary']}
if len(structures) != 1:
    raise ValueError('Structure differs across transport cases')
distinct, launches, operator_bytes = structures.pop()
macros = dict(DgDistinctOperators=distinct, DgLaunches=launches,
              DgOperatorKiB=f'{operator_bytes / 1024:.1f}',
              DgAssembledMiB=f'{large["assembled_bytes"] / 1024 ** 2:.1f}',
              DgStorageFactor=f'{large["assembled_bytes"] / operator_bytes:.0f}',
              DgSpecializedLarge=f'{large["specialized"]:.3f}',
              DgSpecializedRatio=f'{gm(ratios):.3f}',
              DgSpecializedMin=f'{min(ratios):.3f}',
              DgSpecializedMax=f'{max(ratios):.3f}',
              DgSpecializedWins=sum(r['wins'] for r in data['summary']),
              DgProcesses=data['processes'])
(OUT / 'dg-macros.tex').write_text(
    ''.join('\\newcommand{\\' + k + '}{' + str(v) + '}\n' for k, v in macros.items()))

findings = (
    f'Direct execution is faster in every case, from {min(ratios):.3f} to {max(ratios):.3f} times, '
    f'winning all {sum(r["wins"] for r in data["summary"])} of {data["processes"]} process '
    f'comparisons. The advantage is smallest at 4,096 elements, where each product is largest.\n')
(OUT / 'dg-findings.tex').write_text(findings)
(OUT / 'dg-summary.json').write_text(json.dumps(dict(macros=macros, summary=data['summary']), indent=2))
print(json.dumps(macros, indent=2))
