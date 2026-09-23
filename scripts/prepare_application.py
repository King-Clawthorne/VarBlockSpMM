"""Download published matrices and export reproducible VBSR and compact CSR inputs."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import requests
import scipy

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/application'


def partition(n, cycle):
  sizes = []
  remaining = (n + 7) // 8 * 8
  while remaining:
    size = min(cycle[len(sizes) % len(cycle)], remaining)
    sizes.append(size)
    remaining -= size
  return np.array(sizes, dtype='<i4')


def write_vector(file, values, dtype):
  values = np.asarray(values, dtype=dtype)
  file.write(struct.pack('<Q', len(values)))
  file.write(values.tobytes())


def main():
  DEST.mkdir(exist_ok=True, parents=True)
  for number in (13, 14, 15):
    name = f'bcsstk{number}'
    group = 'bcsstruc1' if number == 13 else 'bcsstruc2'
    url = f'https://sparse.tamu.edu/MM/HB/{name}.tar.gz'
    archive = DEST / (name + '.tar.gz')
    if not archive.exists():
      response = requests.get(url, timeout=60)
      response.raise_for_status()
      archive.write_bytes(response.content)
    with tarfile.open(archive, 'r:gz') as bundle:
      matrix = mmread(io.BytesIO(bundle.extractfile(f'{name}/{name}.mtx').read())).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    original_shape = matrix.shape
    scale = float(np.abs(matrix.data).max())
    matrix.data = (matrix.data / scale).astype(np.float32)
    matrix.eliminate_zeros()
    row_sizes = partition(matrix.shape[0], (8, 16, 32, 64))
    col_sizes = partition(matrix.shape[1], (64, 32, 16, 8))
    ro = np.r_[0, np.cumsum(row_sizes)].astype('<i8')
    co = np.r_[0, np.cumsum(col_sizes)].astype('<i8')
    source = matrix.tocoo()
    padded = csr_matrix((source.data, (source.row, source.col)), shape=(int(ro[-1]), int(co[-1])))
    block_rows = np.searchsorted(ro, source.row, side='right') - 1
    block_cols = np.searchsorted(co, source.col, side='right') - 1
    blocks = {}
    for r, c, value, br, bc in zip(source.row, source.col, source.data, block_rows, block_cols):
      key = (int(br), int(bc))
      if key not in blocks:
        blocks[key] = np.zeros((row_sizes[br], col_sizes[bc]), dtype=np.float32)
      blocks[key][r - ro[br], c - co[bc]] = value
    row_ptr, columns, offsets, packed = [0], [], [0], []
    for br in range(len(row_sizes)):
      for _, bc in sorted(key for key in blocks if key[0] == br):
        columns.append(bc)
        values = blocks[br, bc].ravel(order='F')
        packed.extend(values)
        offsets.append(len(packed))
      row_ptr.append(len(columns))
    path = DEST / (name + '.bin')
    with path.open('wb') as file:
      for values, dtype in ((row_sizes, '<i4'), (col_sizes, '<i4'), (ro, '<i8'), (co, '<i8'),
                            (row_ptr, '<i4'), (columns, '<i4'), (offsets, '<i8'), (packed, '<f4'),
                            (padded.indptr, '<i4'), (padded.indices, '<i4'), (padded.data, '<f4')):
        write_vector(file, values, dtype)
    meta = dict(name=name, url=url, source_page=f'https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/{group}/{name}.html',
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                original_shape=original_shape, padded_shape=padded.shape,
                normalization_divisor=scale, scalar_nonzeros=int(padded.nnz),
                packed_values=len(packed), block_count=len(columns),
                block_rows=len(row_sizes), block_cols=len(col_sizes),
                fill_fraction=padded.nnz / len(packed),
                row_partition_cycle=[8, 16, 32, 64], column_partition_cycle=[64, 32, 16, 8])
    (DEST / (name + '.json')).write_text(json.dumps(meta, indent=2))
    print(name, original_shape, 'fill', round(meta['fill_fraction'], 4))


if __name__ == '__main__':
  main()
