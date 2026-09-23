"""Build benchmark binaries and bind their bytes to the exact compiled inputs."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build/verified'


def compiled_hashes():
  paths = [ROOT / 'CMakeLists.txt', Path(__file__).resolve()]
  paths += [p for p in (ROOT / 'cmake').glob('*') if p.is_file() and (p.name.endswith('.cmake') or p.name.endswith('.cmake.in'))]
  paths += [p for folder in ('src', 'include', 'bench', 'tests')
            for p in (ROOT / folder).rglob('*')
            if p.is_file() and p.suffix in ('.cpp', '.cu', '.hpp', '.h', '.cuh', '.c')]
  return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(paths)}


def verified_build(target):
  inputs = compiled_hashes()
  suffix = '.exe' if os.name == 'nt' else ''
  bin_dir = BUILD / 'Release' if os.name == 'nt' else BUILD
  settings = {k: os.environ.get(k) for k in ('CUDAToolkit_ROOT', 'CUDA_PATH')}
  receipt_path = BUILD / 'build-receipt.json'
  binaries = ('vbsr_tests', 'vbsr_audit', 'vbsr_application_audit', 
              'vbsr_ablation', 'vbsr_dg')
  if receipt_path.exists():
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('settings') == settings and receipt['sources'] == inputs and set(receipt['executables']) == {n + suffix for n in binaries} and all(
            (bin_dir / name).is_file() and
            hashlib.sha256((bin_dir / name).read_bytes()).hexdigest() == digest
            for name, digest in receipt['executables'].items()):
      return bin_dir / (target + suffix), receipt
  BUILD.mkdir(parents=True, exist_ok=True)
  configure = ['cmake', '--fresh', '-S', str(ROOT), '-B', str(BUILD),
               '-DCMAKE_CUDA_ARCHITECTURES=native', '-DCMAKE_BUILD_TYPE=Release']
  toolkit = os.environ.get('CUDAToolkit_ROOT') or os.environ.get('CUDA_PATH')
  if toolkit:
    toolkit = Path(toolkit).resolve()
    configure += ['-DCUDAToolkit_ROOT=' + toolkit.as_posix()]
  if os.name == 'nt':
    configure += ['-G', 'Visual Studio 17 2022', '-A', 'x64']
    if toolkit:
      configure += ['-T', 'cuda=' + toolkit.as_posix()]
  elif toolkit:
    configure += ['-DCMAKE_CUDA_COMPILER=' + (toolkit / 'bin/nvcc').as_posix()]
  commands = [configure,
      ['cmake', '--build', str(BUILD), '--config', 'Release', '--clean-first', '--parallel'],
      ['ctest', '--test-dir', str(BUILD), '-C', 'Release', '--output-on-failure']]
  with (BUILD / 'build.log').open('w', encoding='utf-8') as log:
    for command in commands:
      result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
      if result.returncode:
        raise RuntimeError('Verified build failed. See ' + str(BUILD / 'build.log'))
  if compiled_hashes() != inputs:
    raise RuntimeError('Build inputs changed during compilation')
  receipt = dict(schema=1, sources=inputs, commands=commands, settings=settings,
                 cmake_cache_sha256=hashlib.sha256((BUILD / 'CMakeCache.txt').read_bytes()).hexdigest(),
                 executables={name + suffix: hashlib.sha256(
                     (bin_dir / (name + suffix)).read_bytes()).hexdigest() for name in binaries})
  receipt_path.write_text(json.dumps(receipt, indent=2))
  return bin_dir / (target + suffix), receipt


def validate_build_receipt(manifest):
  receipt = manifest.get('build_receipt')
  if receipt is None:
    raise ValueError('Missing verified build receipt')
  if not receipt['sources'] or any(manifest['sources'].get(p) != h for p, h in receipt['sources'].items()):
    raise ValueError('Snapshot does not match compiled inputs')
  if manifest['executable_sha256'] not in receipt['executables'].values():
    raise ValueError('Executable does not match verified build')


if __name__ == '__main__':
  print(verified_build('vbsr_tests')[0])


def find_sanitizer():
  executable = 'compute-sanitizer.exe' if os.name == 'nt' else 'compute-sanitizer'
  toolkit = os.environ.get('CUDAToolkit_ROOT') or os.environ.get('CUDA_PATH')
  if toolkit:
    for directory in ('compute-sanitizer', 'bin'):
      candidate = Path(toolkit) / directory / executable
      if candidate.is_file():
        return candidate
  found = shutil.which(executable)
  if found:
    return Path(found)
  raise FileNotFoundError('Set CUDAToolkit_ROOT or add compute-sanitizer to PATH')
