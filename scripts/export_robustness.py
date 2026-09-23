"""Validate the robustness archive and render its paper table."""
import argparse
import json
from pathlib import Path
from benchmark_runs import ROOT
from run_robustness import analyze
from analyze_revision import geomean


def main():
  parser=argparse.ArgumentParser(__doc__)
  parser.add_argument('--input',type=Path,default=ROOT/'data/robustness')
  args=parser.parse_args()
  result=analyze(args.input)
  original=json.loads((ROOT/'research/generated/summary.json').read_text())
  if result['library_sha256'] != original['provenance']['library_sha256']:
    raise ValueError('Robustness and original library sources differ')
  lines=[r'\begin{tabular}{rrrrrr}',r'\toprule',
         r'Batch & RHS & Best library & BSR8 & Wins & Seed range \\',r'\midrule']
  for r in result['summary']:
    lines.append(f"{r['batch']} & {r['rhs']} & {r['best']:.3f} & {r['bsr8']:.3f} & "
                 f"{r['wins']}/{r['count']} & {r['seed_min']:.3f} to {r['seed_max']:.3f}" + r' \\')
  lines += [r'\bottomrule',r'\end{tabular}']
  (ROOT/'research/generated/robustness.tex').write_text('\n'.join(lines)+'\n')
  macros=[]
  for batch,label in ((1,'Latency'),(8,'Queued')):
    rows=[r for r in result['records'] if r['batch']==batch]
    value=geomean(r['best'] for r in rows)
    macros.append(r'\newcommand{\Robust' + label + 'Overall}{' + f'{value:.3f}' + '}')
    macros.append(r'\newcommand{\Robust' + label + 'Wins}{' + str(sum(r['best']>1 for r in rows)) + '}')
  (ROOT/'research/generated/robustness-macros.tex').write_text('\n'.join(macros)+'\n')


if __name__=='__main__':main()
