"""Census of subnormal FP32 values in the prepared campaign inputs.

The kernels are compiled with --use_fast_math, so subnormal operands are
flushed to zero. The manuscript states which measured inputs contain such
values, and these checks keep that statement tied to the payloads rather than
to prose. The affected values are a vanishing share of the total magnitude, so
they do not move any measured ratio, but the claim has to stay accurate.
"""
import struct
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TINY = np.finfo(np.float32).tiny
# Transport preparation writes outside data/, so name the directory explicitly
# rather than globbing a location that happens to hold other matrices.
TRANSPORT_DIRECTORY = Path('build/transport-v2-inputs')

# The preparation format is a sequence of length-prefixed little-endian
# vectors. The eighth is the packed block payload the kernels read.
LAYOUT = [('row_size', '<i4'), ('col_size', '<i4'), ('row_off', '<i8'), ('col_off', '<i8'),
          ('row_ptr', '<i4'), ('block_col', '<i4'), ('value_off', '<i8'), ('packed', '<f4'),
          ('csr_ptr', '<i4'), ('csr_col', '<i4'), ('csr_values', '<f4')]

# Counted from the checked-in generators. Every other prepared input is free of
# subnormal packed values.
EXPECTED = {
    'covariance_2048_clustered_s1.bin': 162,
    'covariance_2048_clustered_s2.bin': 212,
    'covariance_2048_clustered_s3.bin': 294,
}


def packed_values(path):
  raw = path.read_bytes()
  cursor = 0
  for name, dtype in LAYOUT:
    (count,) = struct.unpack_from('<Q', raw, cursor)
    cursor += 8
    values = np.frombuffer(raw, dtype=dtype, count=count, offset=cursor)
    cursor += count * np.dtype(dtype).itemsize
    if name == 'packed':
      return values
  raise ValueError('packed payload not found in ' + path.name)


def subnormal_count(values):
  magnitude = np.abs(values.astype(np.float32))
  return int(((magnitude > 0) & (magnitude < TINY)).sum())


class InputSubnormalTests(unittest.TestCase):
  def native_inputs(self):
    paths = sorted((ROOT / 'data/native').glob('covariance_*.bin'))
    if not paths:
      self.skipTest('native binaries are not present; run prepare_native.py')
    return paths

  def test_reported_counts_match_the_payloads(self):
    for path in self.native_inputs():
      with self.subTest(path.name):
        self.assertEqual(subnormal_count(packed_values(path)),
                         EXPECTED.get(path.name, 0))

  def test_flushed_magnitude_is_negligible(self):
    """Their numerical contribution is negligible.

        This bounds what flushing changes in the values, not what it changes in
        the time. Nothing here compares flush-to-zero against gradual-underflow
        execution, so it says nothing about the measured ratios.
        """
    for path in self.native_inputs():
      if path.name not in EXPECTED:
        continue
      values = packed_values(path).astype(np.float64)
      magnitude = np.abs(values)
      flushed = magnitude[(magnitude > 0) & (magnitude < TINY)].sum()
      with self.subTest(path.name):
        self.assertLess(flushed / magnitude.sum(), 1e-30)

  def test_published_inputs_have_none(self):
    paths = sorted((ROOT / 'data/application').glob('*.bin'))
    if not paths:
      self.skipTest('application binaries are not present; run prepare_application.py')
    self.assertEqual({p.stem for p in paths}, {'bcsstk13', 'bcsstk14', 'bcsstk15'})
    for path in paths:
      with self.subTest(path.name):
        self.assertEqual(subnormal_count(packed_values(path)), 0)

  def transport_bases(self):
    """The six prepared transport configurations, or skip."""
    directory = ROOT / TRANSPORT_DIRECTORY
    expected = {f'transport_e{elements}_n{rhs}'
                for elements in (256, 1024, 4096) for rhs in (8, 64)}
    present = {p.stem for p in directory.glob('transport_*.bin')} if directory.is_dir() else set()
    if not present:
      self.skipTest('transport inputs are not present; run prepare_transport.py --all')
    # A partial preparation must fail rather than silently checking less.
    self.assertEqual(present, expected, 'transport input coverage is incomplete')
    return sorted(directory / name for name in expected)

  def test_transport_packed_values_have_none(self):
    for base in self.transport_bases():
      with self.subTest(base.name):
        self.assertEqual(subnormal_count(packed_values(base.with_suffix('.bin'))), 0)

  def test_transport_panels_have_none(self):
    """The panels are operands too, so they carry the same requirement."""
    for base in self.transport_bases():
      for suffix in ('.input', '.reference', '.exact', '.weights'):
        path = base.with_suffix(suffix)
        with self.subTest(path.name):
          self.assertTrue(path.is_file(), path.name)
          values = np.fromfile(path, dtype='<f4')
          self.assertGreater(values.size, 0)
          self.assertEqual(subnormal_count(values), 0)


if __name__ == '__main__':
  unittest.main()
