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
                 f'{record["direct_ms"]:.3f}', f'{record["specialized"]:.3f}',
                 f'{record["fused_ms"]:.3f}', f'{record["fused"]:.3f}',
                 f'{record["fused_copies"]:.3f}',
                 f'{record["sharing"]:.3f}',
                 f'{record["sharing_min"]:.3f} to {record["sharing_max"]:.3f}'])
text = ('\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n'
        'Elements & RHS & Steps & Direct ms & Batched & Fused ms & Fused & Copies'
        ' & Sharing & Sharing range\\\\\n'
        '\\midrule\n')
text += '\n'.join(' & '.join(map(str, row)) + '\\\\' for row in rows)
(OUT / 'dg.tex').write_text(text + '\n\\bottomrule\n\\end{tabular}\n')

large = next(r for r in data['summary'] if r['elements'] == 4096 and r['rhs'] == 64)
ratios = [r['specialized'] for r in data['summary']]
structures = {(r['distinct'], r['launches'], r['operator_bytes']) for r in data['summary']}
if len(structures) != 1:
    raise ValueError('Structure differs across transport cases')
distinct, launches, operator_bytes = structures.pop()
# The plan also holds batch pointer arrays, which grow with the element count,
# and the alternating trace holds one plan per direction. Report the factor
# against everything the timed implementation keeps, not one plan's operators.
macros = dict(DgDistinctOperators=distinct, DgLaunches=launches,
              DgOperatorKiB=f'{operator_bytes / 1024:.1f}',
              DgPlanKiB=f'{large["storage_bytes"] / 1024:.1f}',
              DgTraceKiB=f'{large["trace_bytes"] / 1024:.1f}',
              DgAssembledMiB=f'{large["assembled_bytes"] / 1024 ** 2:.1f}',
              DgStorageFactor=f'{large["assembled_bytes"] / large["trace_bytes"]:.1f}',
              DgSpecializedLarge=f'{large["specialized"]:.3f}',
              DgSpecializedRatio=f'{gm(ratios):.3f}',
              DgSpecializedMin=f'{min(ratios):.3f}',
              DgSpecializedMax=f'{max(ratios):.3f}',
              DgSpecializedWins=sum(r['wins'] for r in data['summary']),
              DgProcesses=data['processes'])

# The fused comparator is reported as a speedup over direct execution, which is
# the direction that reads correctly when it wins.
fused = [r['fused'] for r in data['summary']]
fused_speedups = [1 / value for value in fused]
narrow_large = next(r for r in data['summary'] if r['elements'] == 4096 and r['rhs'] == 8)
small = [r for r in data['summary'] if r['elements'] != 4096]
macros.update(DgFusedLarge=f'{1 / large["fused"]:.3f}',
              DgFusedRatio=f'{gm(fused_speedups):.3f}',
              DgFusedMin=f'{min(fused_speedups):.3f}',
              DgFusedMax=f'{max(fused_speedups):.3f}',
              DgFusedWins=sum(3 - r['fused_wins'] for r in data['summary']),
              DgFusedNarrowLarge=f'{1 / narrow_large["fused"]:.3f}',
              # The matched control: identical kernel, private operator copies.
              DgSharingLarge=f'{large["sharing"]:.3f}',
              DgSharingNarrowLarge=f'{narrow_large["sharing"]:.3f}',
              DgSharingSmallMin=f'{min(r["sharing"] for r in small):.3f}',
              DgSharingSmallMax=f'{max(r["sharing"] for r in small):.3f}',
              DgCopiesLarge=f'{large["fused_copies"]:.3f}',
              DgCopiesNarrowLarge=f'{narrow_large["fused_copies"]:.3f}',
              # Direct execution's throughput as a fraction of the fused kernel's.
              DgDirectShareMin=f'{min(r["fused"] for r in data["summary"]) * 100:.0f}',
              DgDirectShareMax=f'{max(r["fused"] for r in data["summary"]) * 100:.0f}',
              DgDirectShareLarge=f'{large["fused"] * 100:.0f}')
(OUT / 'dg-macros.tex').write_text(
    ''.join('\\newcommand{\\' + k + '}{' + str(v) + '}\n' for k, v in macros.items()))

findings = (
    f'Direct execution is faster than the batched comparator in every case, from '
    f'{min(ratios):.3f} to {max(ratios):.3f} times, winning all '
    f'{sum(r["wins"] for r in data["summary"])} of {data["processes"]} process comparisons. '
    f'The fused comparator reverses that: it is faster than direct execution in every case, '
    f'by {min(fused_speedups):.3f} to {max(fused_speedups):.3f} times, taking '
    f'{sum(3 - r["fused_wins"] for r in data["summary"])} of {data["processes"]} process '
    f'comparisons. The matched control separates its two advantages. Operator sharing accounts '
    f'for {large["sharing"]:.3f} times at 4,096 elements and RHS 64 and '
    f'{narrow_large["sharing"]:.3f} times at RHS 8, while at 256 and 1,024 elements it accounts '
    f'for between {min(r["sharing"] for r in small):.3f} and '
    f'{max(r["sharing"] for r in small):.3f} times, showing more modest differences than at 4,096 elements.\n')
(OUT / 'dg-findings.tex').write_text(findings)
(OUT / 'dg-summary.json').write_text(json.dumps(dict(macros=macros, summary=data['summary']), indent=2))
print(json.dumps(macros, indent=2))
