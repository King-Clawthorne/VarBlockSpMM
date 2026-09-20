"""Fetch pinned upstream MAGMA and generate its precision headers."""
from pathlib import Path
import subprocess
import sys

COMMIT = 'c3de65f855dd24cdbe44649a146286558f528087'
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'build/deps/magma'
if __name__ == '__main__':
    if not DEST.exists():
        subprocess.run(['git','clone','--no-checkout','--depth','1','https://github.com/icl-utk-edu/magma.git',str(DEST)],check=True)
        subprocess.run(['git','fetch','--depth','1','origin',COMMIT],cwd=DEST,check=True)
        subprocess.run(['git','checkout','--detach',COMMIT],cwd=DEST,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DEST,text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f'MAGMA checkout is {head}, expected {COMMIT}. Use a separate checkout at the pinned commit.')
    headers=[p.relative_to(DEST).as_posix() for p in sorted((DEST/'include').glob('*.h'))]
    subprocess.run([sys.executable,'tools/codegen.py',*headers],cwd=DEST,check=True)
    print(DEST)
