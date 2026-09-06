"""Periodic 1D upwind modal DG transport, with variable polynomial orders.

The block product advances an ensemble by forward Euler. This is a standalone
PDE demonstrator, not a captured production solver. No performance tuning is
used to select its polynomial order schedule or timestep.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.sparse import csr_matrix
from prepare_application import write_vector


def prepare(elements, rhs, steps, destination):
    if elements < 4 or elements % 4 or rhs not in (8,16,32,64) or not 1 <= steps <= 1024:
        raise ValueError('Use an element count divisible by four, a supported panel width, and 1 to 1024 steps')
    sizes = np.array([8, 16, 32, 64] * ((elements + 3) // 4), dtype='<i4')[:elements]
    offsets = np.r_[0, np.cumsum(sizes)].astype('<i8')
    h = 1.0 / elements
    dt = 0.05 * h / 64**2
    rp, bc, vo, values = [0], [], [0], []
    scalar_r, scalar_c, scalar_v = [], [], []
    for row, m in enumerate(sizes):
        i = np.arange(m)[:, None]
        for col in sorted({row, (row - 1) % elements}):
            n = sizes[col]
            j = np.arange(n)[None, :]
            if col == row:
                # Weak form: M^-1 (S^T - right trace outer product).
                stiffness = 2.0 * ((i > j) & ((i - j) % 2 == 1))
                block = np.eye(m) + dt / h * (2 * i + 1) * (stiffness - 1)
            else:
                block = np.broadcast_to(dt / h * (2 * i + 1) * (-1.0)**i, (m, n))
            block = block.astype('<f4')
            bc.append(col)
            values.append(block.ravel(order='F'))
            vo.append(vo[-1] + block.size)
            rr, cc = np.indices(block.shape)
            scalar_r.append((rr + offsets[row]).ravel())
            scalar_c.append((cc + offsets[col]).ravel())
            scalar_v.append(block.ravel())
        rp.append(len(bc))
    matrix = csr_matrix((np.concatenate(scalar_v), (np.concatenate(scalar_r), np.concatenate(scalar_c))), shape=(offsets[-1], offsets[-1]))
    matrix.sort_indices()
    # Project smooth periodic sine waves onto each local Legendre basis.
    xq, wq = leggauss(96)
    basis = legvander(xq, 63)
    modes = np.arange(64)
    frequencies = np.arange(1, rhs + 1)
    def project(time):
        result = np.zeros((offsets[-1], rhs))
        for row, m in enumerate(sizes):
            x = (row + 0.5 * (xq + 1)) * h
            f = np.sin(2 * np.pi * (x[:, None] - time) * frequencies)
            result[offsets[row]:offsets[row+1]] = ((2*modes[:m]+1)/2)[:,None] * (basis[:,:m].T @ (wq[:,None]*f))
        return result
    initial = project(0).astype('<f4')
    reference = initial.astype(np.float64)
    double_matrix = matrix.astype(np.float64)
    for _ in range(steps):
        reference = double_matrix @ reference
    exact = project(steps * dt)
    weights = np.concatenate([h/(2*np.arange(m)+1) for m in sizes])
    relative_l2 = float(np.sqrt(np.sum(weights[:,None]*(reference-exact)**2) / np.sum(weights[:,None]*exact**2)))
    if relative_l2 > 2e-3:
        raise ValueError(f'Transport discretization failed analytic check: {relative_l2}')
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / f'transport_e{elements}_n{rhs}'
    with base.with_suffix('.bin').open('wb') as out:
        for array in (sizes,sizes,offsets,offsets,np.array(rp,dtype='<i4'),np.array(bc,dtype='<i4'),
                      np.array(vo,dtype='<i8'),np.concatenate(values).astype('<f4'),
                      matrix.indptr.astype('<i4'),matrix.indices.astype('<i4'),matrix.data.astype('<f4')):
            write_vector(out,array,array.dtype)
    initial.ravel(order='F').tofile(str(base)+'.input')
    reference.astype('<f4').ravel(order='F').tofile(str(base)+'.reference')
    meta = dict(elements=elements,rhs=rhs,steps=steps,dt=dt,scalar_rows=int(offsets[-1]),
                analytic_relative_l2=relative_l2,polynomial_orders=[7,15,31,63],degree=2)
    base.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    return meta

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--elements',type=int,default=1024)
    p.add_argument('--rhs',type=int,default=64)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--output',type=Path,default=Path('build/transport-inputs'))
    a=p.parse_args()
    print(json.dumps(prepare(a.elements,a.rhs,a.steps,a.output)))
