"""Independent matrix seeds and queued-throughput controls on the full core grid."""
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
from run_revision import SyntheticCase
from analyze_revision import geomean, library_digest

SEEDS = (2, 3, 5)
ORDER_SEED = 20260909
REPS = 20
BATCHES = (1, 8)

def jobs():
  result = [(SyntheticCase(4096, degree, rhs, dist, locality, seed), batch)
            for seed, dist, locality, degree, rhs, batch in itertools.product(
                SEEDS, ('uniform', 'low', 'high', 'bimodal'), ('local', 'random'),
                (2, 4, 8, 16), (8, 16, 32, 64), BATCHES)]
  rng = random.Random(ORDER_SEED)
  rng.shuffle(result)
  return [(c, b, rng.randrange(2**31)) for c, b in result]

def arguments(c, batch, order):
  return [str(c.rows), str(c.degree), str(c.rhs), c.distribution, c.locality,
          str(c.seed), str(order), '0', str(REPS), str(batch)]

def stem(c, b):
  return c.stem(0) + '_b' + str(b)

def analyze(folder):
  manifest = json.loads((folder / 'manifest.json').read_text())
  validate_build_receipt(manifest)
  expected = {stem(c,b): (c,b,arguments(c,b,o)) for c,b,o in jobs()}
  if manifest['jobs'] != [[list(c), b, o] for c,b,o in jobs()]:
    raise ValueError('Robustness design mismatch')
  if manifest['reps'] != REPS or manifest['seeds'] != list(SEEDS):
    raise ValueError('Robustness repetition or seed mismatch')
  with zipfile.ZipFile(folder / 'source_snapshot.zip') as archive:
    if set(archive.namelist()) != set(manifest['sources']):
      raise ValueError('Snapshot coverage mismatch')
    for name, digest in manifest['sources'].items():
      if hashlib.sha256(archive.read(name)).hexdigest() != digest:
        raise ValueError('Snapshot checksum mismatch')
  rows=[]
  records=list(read_runs(folder))
  if {r.stem for r in records} != set(expected):
    raise ValueError('Robustness coverage mismatch')
  for r in records:
    c,b,command = expected[r.stem]
    validate_run(r,c.methods(),REPS)
    if r.metadata['command'][1:] != command or r.metadata['executable_sha256'] != manifest['executable_sha256']:
      raise ValueError('Command or executable mismatch')
    medians={m: statistics.median(float(t['gpu_ms']) for t in r.rows() if t['method']==m) for m in c.methods()}
    best=min(v for m,v in medians.items() if m not in ('direct','direct_changing','grouped_changing'))
    rows.append(dict(seed=c.seed, rhs=c.rhs, degree=c.degree, distribution=c.distribution,
                     locality=c.locality, batch=b, best=best/medians['direct'],
                     bsr8=medians['bsr8']/medians['direct'], medians=medians))
  summary=[]
  for batch,rhs in itertools.product(BATCHES,(8,16,32,64)):
    selected=[r for r in rows if r['batch']==batch and r['rhs']==rhs]
    means=[geomean(r['best'] for r in selected if r['seed']==s) for s in SEEDS]
    summary.append(dict(batch=batch,rhs=rhs,best=geomean(r['best'] for r in selected),
                        bsr8=geomean(r['bsr8'] for r in selected),wins=sum(r['best']>1 for r in selected),
                        count=len(selected),seed_min=min(means),seed_max=max(means)))
  output=dict(processes=len(rows),timed_calls=sum(len(r.rows())*expected[r.stem][1] for r in records),
              library_sha256=library_digest(manifest),summary=summary,records=rows)
  (folder/'summary.json').write_text(json.dumps(output,indent=2))
  print(json.dumps(summary,indent=2),flush=True)
  return output

def main():
  parser=argparse.ArgumentParser(__doc__)
  parser.add_argument('--output',type=Path,default=ROOT/'data/robustness')
  parser.add_argument('--analyze-only',action='store_true')
  args=parser.parse_args()
  if args.analyze_only:
    analyze(args.output);return
  executable,receipt=verified_build('vbsr_audit')
  args.output.mkdir(parents=True,exist_ok=True)
  sources=source_paths('run_robustness.py')
  manifest=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                jobs=jobs(),reps=REPS,seeds=SEEDS,batches=BATCHES,
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                sources=source_hashes(sources),build_receipt=receipt)
  save_manifest(args.output,manifest,sources)
  existing={r.stem:r for r in read_runs(args.output)}
  start=time.monotonic()
  for i,(c,b,o) in enumerate(jobs()):
    name=stem(c,b);command=[str(executable),*arguments(c,b,o)]
    if name in existing:
      validate_run(existing[name],c.methods(),REPS)
      if existing[name].metadata['command'] != command:
        raise ValueError('Resume command mismatch')
      continue
    run_process(args.output,name,command,c.methods(),REPS,'environment')
    if i%10==0 or i+1==len(jobs()):
      print(f'{i+1}/{len(jobs())} processes, {time.monotonic()-start:.0f}s elapsed',flush=True)
  if not (args.output/'runs.zip').exists():archive_runs(args.output)
  analyze(args.output)

if __name__=='__main__':main()
