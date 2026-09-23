"""Build and test the optional comparison, binding compiled inputs to binaries."""
import json
import os
from pathlib import Path
import subprocess
from build_verified import compiled_hashes
from run_relevance import sha
from prepare_magma import COMMIT, DEST

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'build/relevance'
def inputs():
  result=compiled_hashes()
  for folder in ('include','control','magmablas','interface_cuda'):
    for p in (DEST/folder).rglob('*'):
      if p.is_file(): result['magma/'+p.relative_to(DEST).as_posix()]=sha(p)
  return result

if __name__=='__main__':
  if subprocess.check_output(['git','rev-parse','HEAD'],cwd=DEST,text=True).strip()!=COMMIT:
    raise RuntimeError('Wrong MAGMA revision')
  before=inputs()
  toolkit=os.environ.get('CUDAToolkit_ROOT') or os.environ.get('CUDA_PATH')
  configure=['cmake','-S',str(ROOT),'-B',str(BUILD),'-DCMAKE_CUDA_ARCHITECTURES=native',
             '-DCMAKE_BUILD_TYPE=Release',
             '-DVARBLOCKSPMM_MAGMA_SOURCE='+DEST.as_posix()]
  if os.name=='nt':
    configure+=['-G','Visual Studio 17 2022','-A','x64']
    if toolkit: configure+=['-Tcuda='+Path(toolkit).as_posix()]
  if toolkit: configure+=['-DCUDAToolkit_ROOT='+Path(toolkit).as_posix()]
  commands=[configure,['cmake','--build',str(BUILD),'--config','Release','--clean-first','--parallel'],
            ['ctest','--test-dir',str(BUILD),'-C','Release','--output-on-failure']]
  BUILD.mkdir(parents=True,exist_ok=True)
  with (BUILD/'verified.log').open('w') as log:
    for command in commands: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  if inputs()!=before: raise RuntimeError('Compiled inputs changed during build')
  cache={}
  for line in (BUILD/'CMakeCache.txt').read_text().splitlines():
    if line and not line.startswith(('#','//')) and '=' in line:
      key,value=line.split('=',1)
      cache[key.split(':',1)[0]]=value
  if cache.get('CMAKE_BUILD_TYPE')!='Release':
    raise RuntimeError('Comparison build must configure Release')
  binaries=BUILD/'Release' if cache.get('CMAKE_CONFIGURATION_TYPES') else BUILD
  suffix='.exe' if os.name=='nt' else ''
  names=['vbsr_relevance','vbsr_transport','vbsr_updates','vbsr_tests','vbsr_magma_tests']
  receipt=dict(sources=before,commands=commands,magma_commit=COMMIT,
               configuration='Release',generator=cache['CMAKE_GENERATOR'],
               build_type=cache['CMAKE_BUILD_TYPE'],
               configuration_types=cache.get('CMAKE_CONFIGURATION_TYPES',''),
               cmake_cache_sha256=sha(BUILD/'CMakeCache.txt'),
               executables={name+suffix:sha(binaries/(name+suffix)) for name in names})
  (BUILD/'build-receipt.json').write_text(json.dumps(receipt,indent=2))
  print('Release comparison build and all correctness tests passed.')
