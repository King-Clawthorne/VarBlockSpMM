"""Build benchmark binaries and bind their bytes to the exact compiled inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build/verified'


def compiled_hashes():
    paths = [ROOT / 'CMakeLists.txt', Path(__file__).resolve()]
    paths += [p for folder in ('src', 'include', 'bench', 'tests')
              for p in (ROOT / folder).rglob('*')
              if p.is_file() and p.suffix in ('.cpp', '.cu', '.hpp', '.h', '.cuh', '.c')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def verified_build(target):
    inputs = compiled_hashes()
    receipt_path = BUILD / 'build-receipt.json'
    binaries = ('vbsr_tests', 'vbsr_audit', 'vbsr_application_audit', 'vbsr_kernel_audit', 'vbsr_ablation')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['sources'] == inputs and all(
                (BUILD / 'Release' / name).is_file() and
                hashlib.sha256((BUILD / 'Release' / name).read_bytes()).hexdigest() == digest
                for name, digest in receipt['executables'].items()):
            return BUILD / 'Release' / (target + '.exe'), receipt
    BUILD.mkdir(parents=True, exist_ok=True)
    commands = [
        ['cmake', '-S', str(ROOT), '-B', str(BUILD), '-G', 'Visual Studio 17 2022', '-A', 'x64',
         '-DCMAKE_CUDA_ARCHITECTURES=native', '-T',
         'cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4'],
        ['cmake', '--build', str(BUILD), '--config', 'Release', '--clean-first', '--parallel'],
        ['ctest', '--test-dir', str(BUILD), '-C', 'Release', '--output-on-failure']]
    with (BUILD / 'build.log').open('w', encoding='utf-8') as log:
        for command in commands:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Verified build failed. See ' + str(BUILD / 'build.log'))
    if compiled_hashes() != inputs:
        raise RuntimeError('Build inputs changed during compilation')
    receipt = dict(schema=1, sources=inputs, commands=commands,
                   cmake_cache_sha256=hashlib.sha256((BUILD / 'CMakeCache.txt').read_bytes()).hexdigest(),
                   executables={name + '.exe': hashlib.sha256(
                       (BUILD / 'Release' / (name + '.exe')).read_bytes()).hexdigest() for name in binaries})
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return BUILD / 'Release' / (target + '.exe'), receipt


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
