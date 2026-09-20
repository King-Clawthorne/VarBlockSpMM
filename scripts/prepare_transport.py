"""Periodic 1D upwind modal DG transport, with variable polynomial orders.

Each block product translates by a quarter cell along exact characteristics,
then projects onto the receiving Legendre basis. This semi-Lagrangian DG
demonstrator is separate from the archived forward-Euler experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.sparse import csr_matrix
from prepare_application import write_vector

FINAL_TIME = 1 / 128
INPUT_DIRECTORY = Path('build/transport-v2-inputs')
ABSOLUTE_TOLERANCE = 5e-4
RELATIVE_TOLERANCE = 5e-5

def transport_steps(elements):
    if elements < 32 or elements % 32:
        raise ValueError('Campaign element counts must be positive multiples of 32')
    return elements // 32

def translation_block(m, n, upstream):
    """Exact polynomial overlap integrals, to quadrature roundoff."""
    nodes, weights = leggauss(96)
    left, right = (-1., -.5) if upstream else (-.5, 1.)
    target = (right + left) / 2 + (right - left) / 2 * nodes
    source = target + 1.5 if upstream else target - .5
    test = legvander(target, m - 1)
    trial = legvander(source, n - 1)
    return ((2 * np.arange(m) + 1) / 2)[:, None] * (
        test.T @ (weights[:, None] * (right - left) / 2 * trial))

def error_metrics(actual, reference):
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return dict(max_absolute=float(np.max(np.abs(difference))),
                relative_l2=float(np.linalg.norm(difference) / np.linalg.norm(reference)))

def require_identity_rejection(initial, reference):
    error = error_metrics(initial, reference)
    if error['max_absolute'] <= 10 * ABSOLUTE_TOLERANCE or error['relative_l2'] <= 10 * RELATIVE_TOLERANCE:
        raise ValueError('Unchanged-state negative control lacks a tenfold rejection margin')
    return error

def prepare(elements, rhs, steps, destination, *, fp64_study=False):
    if elements < 4 or elements % 4 or rhs not in (8,16,32,64) or not 1 <= steps <= 1024:
        raise ValueError('Use an element count divisible by four, a supported panel width, and 1 to 1024 steps')
    sizes = np.array([8, 16, 32, 64] * ((elements + 3) // 4), dtype='<i4')[:elements]
    offsets = np.r_[0, np.cumsum(sizes)].astype('<i8')
    h = 1.0 / elements
    dt = .25 * h
    block_cache = {}
    rp, bc, vo, values = [0], [], [0], []
    scalar_r, scalar_c, scalar_v = [], [], []
    for row, m in enumerate(sizes):
        for col in sorted({row, (row - 1) % elements}):
            n = sizes[col]
            key = (int(m), int(n), col != row)
            if key not in block_cache:
                block_cache[key] = translation_block(*key).astype('<f8' if fp64_study else '<f4')
            block = block_cache[key]
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
    initial = project(0).astype('<f8' if fp64_study else '<f4')
    reference = initial.astype(np.float64)
    double_matrix = matrix.astype(np.float64)
    for _ in range(steps):
        reference = double_matrix @ reference
    exact = project(steps * dt)
    weights = np.concatenate([h/(2*np.arange(m)+1) for m in sizes])
    relative_l2 = float(np.sqrt(np.sum(weights[:,None]*(reference-exact)**2) / np.sum(weights[:,None]*exact**2)))
    if relative_l2 > 2e-3:
        raise ValueError(f'Transport discretization failed analytic check: {relative_l2}')
    identity = require_identity_rejection(initial, reference)
    meta = dict(schema=2, integrator='characteristic_l2_projection', elements=elements,
                rhs=rhs,steps=steps,dt=dt,final_time=steps*dt,scalar_rows=int(offsets[-1]),
                analytic_relative_l2=relative_l2,polynomial_orders=[7,15,31,63],degree=2,
                identity_error=identity, precision='fp64' if fp64_study else 'fp32')
    if fp64_study:
        return meta
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / f'transport_e{elements}_n{rhs}'
    with base.with_suffix('.bin').open('wb') as out:
        for array in (sizes,sizes,offsets,offsets,np.array(rp,dtype='<i4'),np.array(bc,dtype='<i4'),
                      np.array(vo,dtype='<i8'),np.concatenate(values).astype('<f4'),
                      matrix.indptr.astype('<i4'),matrix.indices.astype('<i4'),matrix.data.astype('<f4')):
            write_vector(out,array,array.dtype)
    initial.ravel(order='F').tofile(str(base)+'.input')
    reference.astype('<f4').ravel(order='F').tofile(str(base)+'.reference')
    exact.astype('<f4').ravel(order='F').tofile(str(base)+'.exact')
    weights.astype('<f4').tofile(str(base)+'.weights')
    base.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    return meta

def convergence_study(destination):
    records = [prepare(e, 8, e // 4, destination, fp64_study=True) for e in (16, 32, 64)]
    errors = [r['analytic_relative_l2'] for r in records]
    if not all(a > 4 * b for a, b in zip(errors, errors[1:], strict=False)):
        raise ValueError(f'Spatial refinement failed to reduce error: {errors}')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'convergence.json').write_text(json.dumps(records, indent=2))
    return records

def prepare_campaign(destination=INPUT_DIRECTORY):
    for elements in (256, 1024, 4096):
        for rhs in (8, 64):
            print(json.dumps(prepare(elements, rhs, transport_steps(elements), destination)), flush=True)
    print(json.dumps(convergence_study(destination)), flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--elements',type=int,default=1024)
    p.add_argument('--rhs',type=int,default=64)
    p.add_argument('--steps',type=int,default=None)
    p.add_argument('--all',action='store_true',help='Prepare all six campaign inputs and the convergence study')
    p.add_argument('--output',type=Path,default=INPUT_DIRECTORY)
    a=p.parse_args()
    if a.all: prepare_campaign(a.output)
    else: print(json.dumps(prepare(a.elements,a.rhs,a.steps or transport_steps(a.elements),a.output)))
