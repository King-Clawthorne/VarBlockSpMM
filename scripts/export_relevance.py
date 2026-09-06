import sys
import argparse
from pathlib import Path
import json, math, statistics
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_relevance import analyze
parser=argparse.ArgumentParser(description='Validate and export the focused comparison.')
parser.add_argument('--input',type=Path,default=ROOT/'data/relevance')
args=parser.parse_args()
data=analyze(args.input)
manifest=json.loads((args.input/'manifest.json').read_text())
original=json.loads((ROOT/'research/generated/summary.json').read_text())
if manifest['sources']['src/kernels/row_owned.cu'] != original['provenance']['kernel_sha256']:
    raise ValueError('Focused comparison and original direct kernel sources differ')
OUT=ROOT/'research/generated'
def gm(x):
    x=list(x);return math.exp(sum(map(math.log,x))/len(x))
def time(r,m):return r['medians'][m]['gpu_ms']
def ratio(r,m):return time(r,m)/time(r,'direct')
def magma(r):return min(ratio(r,m) for m in ('magma_slots','magma_reduce'))
def best(r):return min(ratio(r,m) for m in r['medians'] if m!='direct')
core=[r for r in data if r['kind']=='core']
transport=[r for r in data if r['kind']=='transport']
controls=[r for r in data if r['kind']=='control']
def table(name,columns,header,rows):
    text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+'\\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+'\\\\' for row in rows)
    (OUT/name).write_text(text+'\n\\bottomrule\n\\end{tabular}\n')
rows=[]
for rhs in (8,16,32,64):
    part=[r for r in core if r['args'][2]==rhs]
    rows.append([rhs,f'{gm(ratio(r,"bsr8") for r in part):.3f}',f'{gm(magma(r) for r in part):.3f}',
                 f'{gm(best(r) for r in part):.3f}',f'{sum(best(r)>1 for r in part)}/32'])
table('magma-use-cases.tex','rrrrr','RHS & BSR8 & MAGMA best & Best library & Direct wins',rows)
groups=defaultdict(list)
for r in transport:
    e=int(r['args'][0].split('_e')[1].split('_')[0]);groups[e,r['args'][1]].append(r)
rows=[]
for (e,rhs),part in sorted(groups.items()):
    rows.append([f'{e:,}',rhs,f'{gm(time(r,"direct") for r in part):.3f}',f'{gm(magma(r) for r in part):.3f}',
                 f'{gm(best(r) for r in part):.3f}',f'{sum(best(r)>1 for r in part)}/3'])
transport_rows=list(rows)
table('transport.tex','rrrrrr','Elements & RHS & Direct ms & MAGMA best & Best library & Wins',rows)
groups_control=defaultdict(list)
for r in controls:groups_control[tuple(r['args'][:4])].append(r)
rows=[]
for (nr,d,rhs,s),part in sorted(groups_control.items()):
    rows.append([nr,d,s,rhs,f'{gm(time(r,"direct") for r in part):.3f}',f'{gm(ratio(r,"bsr8") for r in part):.3f}',
                 f'{gm(magma(r) for r in part):.3f}',f'{gm(best(r) for r in part):.3f}'])
table('controlled.tex','rrrrrrrr','Rows & Degree & Size & RHS & Direct ms & BSR8 & MAGMA best & Best library',rows)
large=groups[4096,64]
macros=dict(RelevanceCoreRatio=gm(best(r) for r in core),RelevanceCoreWins=sum(best(r)>1 for r in core),
            RelevanceMagmaRatio=gm(magma(r) for r in core),TransportLargeRatio=gm(best(r) for r in large),
            TransportLargeMagmaRatio=gm(magma(r) for r in large))
(OUT/'relevance-macros.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+(str(v) if isinstance(v,int) else f'{v:.3f}')+'}\n' for k,v in macros.items()))
findings=(f'On the 4,096-element, RHS-64 transport case, the fastest-library-to-direct ratio is '
          f'{macros["TransportLargeRatio"]:.3f}, with {sum(best(r)>1 for r in large)} of three direct wins. '
          f'The ratio against the faster MAGMA composition is {macros["TransportLargeMagmaRatio"]:.3f}. '
          'The table retains smaller traces and narrow ensembles, including cases that favor another implementation. '
          'This is evidence for the stated dependent transport product, not a universal application speedup.\n')
contrasts={}
for rhs in (8,64):
    part=[r for r in controls if r['args'][2]==rhs]
    values={}
    for name,index,low,high in [('degree',1,4,128),('rows',0,128,512),('size',3,8,32)]:
        lo=gm(ratio(r,'bsr8') for r in part if r['args'][index]==low)
        hi=gm(ratio(r,'bsr8') for r in part if r['args'][index]==high)
        values[name]=dict(low=lo,high=hi,relative=hi/lo)
    contrasts[rhs]=values
(OUT/'relevance-findings.tex').write_text(findings)
control_text = ('Averaging equally over the other crossed factors, increasing degree from 4 to 128 changes '
    f'the BSR8-to-direct ratio from {contrasts[8]["degree"]["low"]:.3f} to {contrasts[8]["degree"]["high"]:.3f} at RHS 8, '
    f'and from {contrasts[64]["degree"]["low"]:.3f} to {contrasts[64]["degree"]["high"]:.3f} at RHS 64. '
    'Changing block size from 8 to 32 increases that ratio by factors of '
    f'{contrasts[8]["size"]["relative"]:.3f} and {contrasts[64]["size"]["relative"]:.3f}, respectively. '
    'The row-count contrast is much smaller at RHS 8 and unfavorable at RHS 64. '
    'Thus both degree and shape affect the performance boundary, and more block rows alone do not guarantee an advantage.\n')
(OUT/'controlled-findings.tex').write_text(control_text)
updates=[r for r in data if r['kind']=='updates']
ut={name:gm(r['medians'][name]['host_ms'] for r in updates) for name in ('value_update','structure_replan')}
(OUT/'update-findings.tex').write_text(
    '\\paragraph{Update paths.} On the 256-block-row RHS-32 API test, value replacement plus execution takes '
    f'{ut["value_update"]:.3f} ms in synchronized host time, versus {ut["structure_replan"]:.3f} ms for '
    'borrowing changed structure, validating its metadata, rebuilding the row order, and executing. '
    'These are geometric means of three process medians, with twenty samples each. Values alternate between '
    'two pre-existing device payloads on fixed degree-four structure. Structural cases alternate between '
    'degree-four and degree-eight matrices already on the GPU. Thus this is an API-path cost comparison, '
    'not an equal-work kernel speedup or a measurement of assembling a new structure.\n')
summary=dict(macros=macros,contrasts=contrasts,updates=ut,transport_rows=transport_rows)
(OUT/'relevance-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(macros=macros,contrasts=contrasts,updates=ut),indent=2))

# Replace the main width plot with the follow-up that includes MAGMA.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),sharey=True)
for ax,fn,title in zip(axes,(lambda r:ratio(r,'bsr8'),best),('Padding-free BSR8','Fastest tested library, including MAGMA')):
    means=[]
    for i,rhs in enumerate((8,16,32,64)):
        vals=sorted(fn(r) for r in core if r['args'][2]==rhs)
        ax.scatter([i+(j-15.5)/90 for j in range(32)],vals,s=9,alpha=.48,color='#487db5',edgecolors='none')
        means.append(gm(vals))
    ax.plot(range(4),means,marker='D',color='#123d65',lw=1.4,ms=4)
    ax.axhline(1,color='.3',ls='--',lw=.8);ax.set_yscale('log',base=2)
    ax.set_ylim(.75,1.75)
    ax.set_yticks([.75,1,1.25,1.5,1.75])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(4),['8','16','32','64']);ax.set_xlabel('Dense panel width (RHS)')
    ax.set_title(title,fontsize=9);ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Comparator time / direct time\nAbove 1: direct faster')
fig.tight_layout();fig.savefig(OUT/'magma-use-cases.pdf');plt.close(fig)
