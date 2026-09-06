"""Bounded follow-up campaign. Never runs the historical ablation sweep."""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import zipfile
from prepare_magma import COMMIT
from prepare_transport import INPUT_DIRECTORY, transport_steps
from archive_inputs import input_zip, split_inputs

ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def design(version=2):
    jobs=[]
    seeds, processes, batches = ((2,), (0,), (1,)) if version == 1 else ((2,3,5), (0,1,2), (1,8))
    for degree,rhs,shape,locality,seed,process,batch in itertools.product(
            (2,4,8,16),(8,16,32,64),(-3,-2,-1,0),('local','random'),seeds,processes,batches):
        args=[4096,degree,rhs,shape,seed,6 if version==1 else 20,0,'none',locality]
        jobs.append(dict(kind='core',args=args) if version==1 else
                    dict(kind='core',process=process,batch=batch,args=args+[batch]))
    # Full crossed control. Three process repetitions of matrix seed 11.
    for rows,degree,rhs,shape,process in itertools.product((128,512),(4,16,128),(8,64),(8,32),(0,1,2)):
        jobs.append(dict(kind='control',process=process,args=[rows,degree,rhs,shape,11,6,0,'none']))
    for elements,rhs,process in itertools.product((256,1024,4096),(8,64),(0,1,2)):
        directory = 'build/transport-inputs' if version==1 else INPUT_DIRECTORY.as_posix()
        steps = 32 if version==1 else transport_steps(elements)
        jobs.append(dict(kind='transport',process=process,args=[f'{directory}/transport_e{elements}_n{rhs}',rhs,steps,6,0]))
    for process in range(3): jobs.append(dict(kind='updates',process=process,args=[]))
    rng=random.Random(20260910)
    for job in jobs:
        order=rng.randrange(1,2**31)
        if job['kind'] in ('core','control'): job['args'][6]=order
        elif job['kind']=='transport': job['args'][4]=order
    rng.shuffle(jobs)
    return jobs

def expected_inputs(jobs):
    names = {str(INPUT_DIRECTORY/'convergence.json').replace('\\','/')}
    for job in jobs:
        if job['kind']=='transport':
            names.update(job['args'][0]+suffix for suffix in ('.bin','.input','.reference','.exact','.weights','.json'))
    return names


def validate_inputs(dest, manifest):
    with input_zip(dest, manifest) as archive:
        validate_input_archive(manifest, archive)


def validate_input_archive(manifest, archive):
    expected = expected_inputs(manifest['jobs'])
    if set(manifest.get('inputs', {})) != expected:
        raise ValueError('Input manifest coverage mismatch')
    if not archive.is_file() or sha(archive) != manifest.get('input_archive_sha256'):
        raise ValueError('Input archive checksum mismatch')
    with zipfile.ZipFile(archive) as inputs:
        if len(inputs.namelist()) != len(set(inputs.namelist())) or set(inputs.namelist()) != expected:
            raise ValueError('Input archive coverage mismatch')
        for name, digest in manifest['inputs'].items():
            # Stream large payloads instead of allocating another matrix copy.
            with inputs.open(name) as payload:
                actual = hashlib.file_digest(payload, 'sha256').hexdigest()
            if actual != digest: raise ValueError('Input payload checksum mismatch')
        for job in manifest['jobs']:
            if job['kind']!='transport': continue
            base, rhs, steps = job['args'][:3]
            meta=json.loads(inputs.read(base+'.json'))
            elements=int(base.split('_e')[1].split('_')[0])
            if (meta.get('schema')!=2 or meta.get('integrator')!='characteristic_l2_projection'
                or meta.get('elements')!=elements or meta.get('rhs')!=rhs or meta.get('steps')!=steps
                or meta.get('final_time')!=1/128 or meta.get('dt')!=.25/elements
                or meta.get('scalar_rows')!=elements*30 or meta.get('precision')!='fp32'):
                raise ValueError('Transport metadata differs from command or physical protocol')
            import math
            error=meta.get('analytic_relative_l2', float('nan'))
            identity=meta.get('identity_error', {})
            if (not math.isfinite(error) or not 0 <= error <= 2e-3
                or not identity.get('max_absolute',0)>.005
                or not identity.get('relative_l2',0)>.0005):
                raise ValueError('Transport numerical validation failed')
            count=elements*30
            for suffix, size in (('.input',count*rhs*4),('.reference',count*rhs*4),
                                 ('.exact',count*rhs*4),('.weights',count*4)):
                if inputs.getinfo(base+suffix).file_size != size:
                    raise ValueError('Transport payload extent mismatch')
        convergence=json.loads(inputs.read((INPUT_DIRECTORY/'convergence.json').as_posix()))
        if [r['elements'] for r in convergence]!=[16,32,64]:
            raise ValueError('Convergence design mismatch')
        errors=[r['analytic_relative_l2'] for r in convergence]
        if not all(math.isfinite(e) and e>0 for e in errors) or not all(a>4*b for a,b in zip(errors,errors[1:])):
            raise ValueError('Convergence check failed')
        if any(r['final_time']!=1/16 or r['precision']!='fp64' or r['rhs']!=8 for r in convergence):
            raise ValueError('Convergence physical protocol mismatch')


def analyze(dest, *, allow_legacy=False):
    m=json.loads((dest/'manifest.json').read_text())
    version=m.get('schema',1)
    if version!=2 and not (version==1 and allow_legacy):
        raise ValueError('Legacy campaign lacks archived input validation. Use explicitly for historical review only')
    if not m.get('complete') or m['jobs']!=design(version): raise ValueError('Incomplete or changed experiment design')
    if len(m['records'])!=len(m['jobs']): raise ValueError('Missing process records')
    if any(m['sources'].get(p)!=h for p,h in m['build_receipt']['sources'].items()): raise ValueError('Build inputs disagree with archive')
    if any(m['build_receipt']['executables'].get(p)!=h for p,h in m['executables'].items()): raise ValueError('Build binaries disagree with manifest')
    if sha(dest/'sources.zip')!=m['source_archive_sha256']: raise ValueError('Source archive checksum mismatch')
    if sha(dest/'runs.zip')!=m['run_archive_sha256']: raise ValueError('Run archive checksum mismatch')
    with zipfile.ZipFile(dest/'sources.zip') as sources:
        if len(sources.namelist())!=len(set(sources.namelist())) or set(sources.namelist())!=set(m['sources']):
            raise ValueError('Source archive coverage mismatch')
        for name,digest in m['sources'].items():
            if hashlib.sha256(sources.read(name)).hexdigest()!=digest: raise ValueError('Source hash mismatch')
    if version==2: validate_inputs(dest,m)
    summaries=[]
    with zipfile.ZipFile(dest/'runs.zip') as runs:
        expected_files={f'{i}.{suffix}' for i in range(len(m['jobs'])) for suffix in ('csv','log')}
        if len(runs.namelist())!=len(set(runs.namelist())) or set(runs.namelist())!=expected_files:
            raise ValueError('Run archive coverage mismatch')
        for index,job in enumerate(m['jobs']):
            record=m['records'][index]
            expected_exe={'core':'vbsr_relevance','control':'vbsr_relevance','transport':'vbsr_transport','updates':'vbsr_updates'}[job['kind']]+'.exe'
            if record['executable_sha256']!=m['executables'][expected_exe] or record['returncode']!=0:
                raise ValueError('Failed process or executable mismatch')
            if Path(record['command'][0]).name!=expected_exe or record['command'][1:]!=list(map(str,job['args'])):
                raise ValueError('Process command differs from design')
            raw=runs.read(f'{index}.csv')
            if hashlib.sha256(raw).hexdigest()!=record['stdout_sha256']: raise ValueError('Raw checksum mismatch')
            err=runs.read(f'{index}.log')
            if hashlib.sha256(err).hexdigest()!=record['stderr_sha256']: raise ValueError('Log checksum mismatch')
            if version==2 and job['kind']=='transport':
                log=err.decode()
                if 'identity_negative_control,rejected' not in log:
                    raise ValueError('Missing transport negative control')
                physical=[float(line.split(',')[1]) for line in log.splitlines() if line.startswith('analytic_relative_l2,')]
                import math
                if len(physical)!=24 or any(not math.isfinite(e) or not 0<=e<=2e-3 for e in physical):
                    raise ValueError('Missing or failed GPU analytic checks')
            data={}; positions={}
            for row in csv.reader(io.StringIO(raw.decode())):
                if len(row)!=5 or row[0]=='method': continue
                try: position,trial=int(row[1]),int(row[2]); gpu,host=float(row[3]),float(row[4])
                except ValueError: continue
                import math
                if not all(math.isfinite(x) and x>0 for x in (gpu,host)): raise ValueError('Invalid timing')
                if row[0] in positions and positions[row[0]]!=position: raise ValueError('Inconsistent method position')
                positions[row[0]]=position
                data.setdefault(row[0],[]).append((trial,gpu,host))
            expected={'direct','magma_slots','magma_reduce','bsr8','csr1','csr2','csr3','grouped'}
            if job['kind'] in ('core','control') and job['args'][3] in (-3,32): expected.add('bsr32')
            if job['kind']=='updates': expected={'value_update','structure_replan'}
            if set(data)!=expected: raise ValueError('Wrong method coverage')
            if sorted(positions.values())!=list(range(len(expected))): raise ValueError('Wrong method positions')
            reps=20 if job['kind']=='updates' else job['args'][5] if job['kind'] in ('core','control') else job['args'][3]
            if any(sorted(v[0] for v in trials)!=list(range(reps)) for trials in data.values()): raise ValueError('Wrong trial coverage')
            medians={name:dict(gpu_ms=statistics.median(x[1] for x in trials),host_ms=statistics.median(x[2] for x in trials)) for name,trials in data.items()}
            summaries.append(dict(**job,medians=medians))
    (dest/'summary.json').write_text(json.dumps(summaries,indent=2))
    print(f'Validated {len(summaries)} fresh processes')
    return summaries

def run(dest,seconds):
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists(): raise RuntimeError('Use a fresh output directory')
    magma=ROOT/'build/deps/magma'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=magma,text=True).strip()!=COMMIT: raise RuntimeError('Unexpected MAGMA revision')
    jobs=design(); source_paths={}
    for folder in ('src','include','bench','tests','scripts','cmake'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.cpp','.cu','.h','.hpp','.cuh','.py','.ps1','.cmake','.in'):
                source_paths[p.relative_to(ROOT).as_posix()]=p
    for name in ('CMakeLists.txt','pyproject.toml','uv.lock','LICENSE'): source_paths[name]=ROOT/name
    for folder in ('include','control','magmablas','interface_cuda'):
        for p in (magma/folder).rglob('*'):
            if p.is_file(): source_paths['magma/'+p.relative_to(magma).as_posix()]=p
    for p in (ROOT/INPUT_DIRECTORY).glob('*'):
        if p.suffix=='.json': source_paths['inputs/'+p.name]=p
    sources={name:sha(p) for name,p in source_paths.items()}
    with zipfile.ZipFile(dest/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,p in source_paths.items(): z.write(p,name)
    bins=ROOT/'build/relevance/Release'
    executables={n:sha(bins/n) for n in ('vbsr_relevance.exe','vbsr_transport.exe','vbsr_updates.exe')}
    receipt=json.loads((ROOT/'build/relevance/build-receipt.json').read_text())
    if any(sources.get(p)!=h for p,h in receipt['sources'].items()): raise RuntimeError('Stale comparison build receipt')
    if any(receipt['executables'].get(p)!=h for p,h in executables.items()): raise RuntimeError('Stale comparison executable')
    inputs={name:sha(ROOT/name) for name in sorted(expected_inputs(jobs))}
    with zipfile.ZipFile(dest/'inputs.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in inputs: archive.write(ROOT/name,name)
    manifest=dict(schema=2,jobs=jobs,sources=sources,executables=executables,inputs=inputs,magma_commit=COMMIT,build_receipt=receipt,
                  input_archive_sha256=sha(dest/'inputs.zip'),
                  source_archive_sha256=sha(dest/'sources.zip'),records=[],complete=False)
    validate_inputs(dest,manifest)
    started=time.monotonic()
    with zipfile.ZipFile(dest/'runs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,job in enumerate(jobs):
            remaining=None if seconds is None else seconds-(time.monotonic()-started)
            if remaining is not None and remaining<=0: break
            name={'core':'vbsr_relevance','control':'vbsr_relevance','transport':'vbsr_transport','updates':'vbsr_updates'}[job['kind']]+'.exe'
            if sha(bins/name)!=executables[name]: raise RuntimeError('Executable changed')
            command=[str(bins/name),*map(str,job['args'])]
            tick=time.monotonic()
            try: p=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=remaining)
            except subprocess.TimeoutExpired:
                print('Wall-clock cap reached',flush=True); break
            z.writestr(f'{i}.csv',p.stdout);z.writestr(f'{i}.log',p.stderr)
            manifest['records'].append(dict(command=command,returncode=p.returncode,elapsed=time.monotonic()-tick,
               executable_sha256=executables[name],stdout_sha256=hashlib.sha256(p.stdout).hexdigest(),stderr_sha256=hashlib.sha256(p.stderr).hexdigest()))
            print(f'{i+1}/{len(jobs)} {job["kind"]} elapsed={time.monotonic()-started:.1f}s',flush=True)
            if p.returncode:
                print(p.stderr.decode(errors='replace'),flush=True);break
    manifest['complete']=len(manifest['records'])==len(jobs) and all(x['returncode']==0 for x in manifest['records'])
    manifest['run_archive_sha256']=sha(dest/'runs.zip')
    manifest['elapsed_seconds']=time.monotonic()-started
    if any(sha(p)!=sources[name] for name,p in source_paths.items()): raise RuntimeError('Sources changed during campaign')
    if any(sha(ROOT/p)!=h for p,h in inputs.items()): raise RuntimeError('Inputs changed during campaign')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if not manifest['complete']: raise RuntimeError('Campaign stopped before completion. Partial evidence retained.')
    split_inputs(dest)
    analyze(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/relevance-v2')
    p.add_argument('--max-seconds',type=int,default=None);p.add_argument('--analyze',action='store_true')
    a=p.parse_args()
    if a.analyze: analyze(a.output)
    else: run(a.output,a.max_seconds)
