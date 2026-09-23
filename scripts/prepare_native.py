"""Generate the dense near-field component of a spatial covariance operator.

Leaf partitions are made by the geometry before any sparse matrix is assembled.
This is a reproducible application model, not a trace from a production H2 solver.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix
from prepare_application import write_vector

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/native'


def build(name, count, geometry, seed):
  start = time.perf_counter()
  rng = np.random.default_rng(seed)
  points = rng.uniform(0, 1, (count, 3))
  if geometry == 'clustered':
    selected = np.arange(count) < 3 * count // 4
    points[selected] = np.clip(rng.normal(0.3, 0.07, (int(selected.sum()), 3)), 0, 1)
  leaves = []

  def split(indices):
    cloud = points[indices]
    extent = np.ptp(cloud, axis=0)
    if len(indices) <= 8 or (len(indices) <= 64 and np.linalg.norm(extent) <= 0.18):
      leaves.append(indices)
      return
    # Eight-point granularity is an explicit input restriction of this model.
    # No points or scalar entries are padded or dropped.
    axis = int(np.argmax(extent))
    order = indices[np.argsort(cloud[:, axis], kind='stable')]
    middle = max(8, min(len(order) - 8, (len(order) // 16) * 8))
    split(order[:middle])
    split(order[middle:])

  split(np.arange(count))
  permutation = np.concatenate(leaves)
  xyz = points[permutation]
  sizes = np.array([len(x) for x in leaves], dtype='<i4')
  offsets = np.r_[0, np.cumsum(sizes)].astype('<i8')
  lower = np.array([points[x].min(axis=0) for x in leaves])
  upper = np.array([points[x].max(axis=0) for x in leaves])
  diameters = np.linalg.norm(upper - lower, axis=1)
  gap = np.maximum(0, np.maximum(lower[:, None] - upper[None, :], lower[None, :] - upper[:, None]))
  distance = np.linalg.norm(gap, axis=2)
  # Boukaram et al. Eq. (1), eta=0.5. Keep inadmissible leaf pairs.
  near = (diameters[:, None] + diameters[None, :]) * 0.5 > 0.5 * distance
  np.fill_diagonal(near, True)
  assert np.array_equal(near, near.T)
  amplitude = 1 / np.sqrt(count)
  geometry_ms = (time.perf_counter() - start) * 1000

  def covariance(a, b):
    difference = a[:, None, :] - b[None, :, :]
    return (amplitude * np.exp(-np.sum(difference * difference, axis=2) / 0.02)).astype('<f4')

  rp, bc, vo, blocks = [0], [], [0], []
  for row in range(len(sizes)):
    for column in np.flatnonzero(near[row]):
      values = covariance(xyz[offsets[row]:offsets[row + 1]],
                          xyz[offsets[column]:offsets[column + 1]]).ravel(order='F')
      bc.append(int(column))
      blocks.append(values)
      vo.append(vo[-1] + len(values))
    rp.append(len(bc))
  packed = np.concatenate(blocks)
  assembly_ms = (time.perf_counter() - start) * 1000
  scalar_start = time.perf_counter()
  # Independently enumerate scalar rows and recompute the kernel. Do not expand
  # packed block values to build the reference representation.
  owner = np.repeat(np.arange(len(sizes)), sizes)
  cp, ci, cv = [0], [], []
  for row in range(count):
    columns = np.flatnonzero(near[owner[row], owner]).astype('<i4')
    values = covariance(xyz[row:row + 1], xyz[columns]).ravel()
    ci.append(columns)
    cv.append(values)
    cp.append(cp[-1] + len(columns))
  columns, values = np.concatenate(ci), np.concatenate(cv)
  reference = csr_matrix((values, columns, cp), shape=(count, count))
  scalar_ms = (time.perf_counter() - scalar_start) * 1000
  for row in range(len(sizes)):
    for block in range(rp[row], rp[row + 1]):
      column = bc[block]
      expected = reference[offsets[row]:offsets[row + 1], offsets[column]:offsets[column + 1]].toarray()
      np.testing.assert_array_equal(expected.ravel(order='F'), packed[vo[block]:vo[block + 1]])
  path = DEST / (name + '.bin')
  with path.open('wb') as file:
    for vector, dtype in ((sizes, '<i4'), (sizes, '<i4'), (offsets, '<i8'), (offsets, '<i8'),
                          (rp, '<i4'), (bc, '<i4'), (vo, '<i8'), (packed, '<f4'),
                          (cp, '<i4'), (columns, '<i4'), (values, '<f4')):
      write_vector(file, vector, dtype)
  geometry_path = DEST / (name + '-geometry.npz')
  if geometry_path.exists():
    with np.load(geometry_path) as saved:
      for key, value in dict(points=points, permutation=permutation, leaf_offsets=offsets).items():
        np.testing.assert_array_equal(saved[key], value)
  else:
    np.savez_compressed(geometry_path, points=points, permutation=permutation, leaf_offsets=offsets)
  meta = dict(name=name, count=count, geometry=geometry, seed=seed, eta=0.5,
              leaf_diameter=0.18, leaf_quantum=8, max_leaf=64, length_scale=0.1,
              amplitude=amplitude, block_rows=len(sizes), block_count=len(bc),
              shape_counts={int(s): int((sizes == s).sum()) for s in np.unique(sizes)},
              mean_degree=float(np.diff(rp).mean()), max_degree=int(np.diff(rp).max()),
              packed_values=len(packed), scalar_nonzeros=int(np.count_nonzero(values)),
              fill_fraction=float(np.count_nonzero(values) / len(packed)),
              generation_and_packing_ms=assembly_ms,
              geometry_ms=geometry_ms, scalar_assembly_ms=scalar_ms,
              binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              numpy_version=np.__version__,
              source='https://arxiv.org/html/2506.16759v1#S2.SS1',
              scope='Generated covariance near-field component, excluding far-field and solver work')
  metadata_path = DEST / (name + '.json')
  if metadata_path.exists():
    saved = json.loads(metadata_path.read_text())
    if saved['binary_sha256'] != meta['binary_sha256']:
      raise ValueError('Regenerated native input differs from its recorded checksum')
        # Preserve the archived CPU assembly observation when only reconstructing
        # ignored binary inputs for paper validation. Fresh directories remeasure it.
  else:
    metadata_path.write_text(json.dumps(meta, indent=2))
  print(name, meta['shape_counts'], 'degree', meta['mean_degree'], flush=True)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--output', type=Path, default=DEST)
  DEST = parser.parse_args().output
  DEST.mkdir(parents=True, exist_ok=True)
  for count in (2048, 8192):
    for geometry in ('uniform', 'clustered'):
      for seed in (1, 2, 3):
        build(f'covariance_{count}_{geometry}_s{seed}', count, geometry, seed)
