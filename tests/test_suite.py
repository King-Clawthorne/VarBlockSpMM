
# ===== Merged from test_dg.py =====
"""The specialized DG comparison archive supports the numbers the paper reports."""
import csv
import json
import io
import math
import re
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_dg import analyze

SOURCE = ROOT / 'data/dg'


@unittest.skipUnless((SOURCE / 'manifest.json').exists(), 'campaign archive not present')
class DgEvidenceTests(unittest.TestCase):
  def test_process_aggregation_matches_raw_trial_products(self):
    """Independent reduction: trial medians, then products and roots.

        Deliberately do not reuse analyzer records or its geometric-mean
        helper. The retained asymmetric processes distinguish a median of
        ratios from equal process weighting in log space.
        """
    grouped = {}
    with zipfile.ZipFile(SOURCE / 'runs.zip') as archive:
      for name in archive.namelist():
        match = re.fullmatch(r'dg_e(\d+)_n(\d+)_p\d+\.csv', name)
        if not match:
          continue
        trials = {}
        for row in csv.DictReader(io.StringIO(archive.read(name).decode())):
          trials.setdefault(row['method'], []).append(float(row['gpu_ms']))
        medians = {}
        for method, values in trials.items():
          values.sort()
          middle = len(values) // 2
          medians[method] = (values[(len(values)-1)//2] + values[middle]) / 2
        grouped.setdefault(tuple(map(int, match.groups())), []).append(medians)
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      self.copy(destination)
      data = analyze(destination)
    self.assertEqual(len(grouped), 6)
    for row in data['summary']:
      processes = grouped[row['elements'], row['rhs']]
      self.assertEqual(len(processes), 3)
      expected = {
          'direct_ms': [p['direct'] for p in processes],
          'specialized_ms': [p['dg_specialized'] for p in processes],
          'fused_ms': [p['dg_fused'] for p in processes],
          'specialized': [p['dg_specialized']/p['direct'] for p in processes],
          'fused': [p['dg_fused']/p['direct'] for p in processes],
          'fused_copies': [p['dg_fused_copies']/p['direct'] for p in processes],
          'sharing': [p['dg_fused_copies']/p['dg_fused'] for p in processes],
          'best_generic': [min(p['grouped'],p['bsr8'])/p['direct'] for p in processes],
      }
      for field, values in expected.items():
        with self.subTest(elements=row['elements'], rhs=row['rhs'], field=field):
          self.assertAlmostEqual(row[field], math.prod(values)**(1/len(values)), places=11)

  def copy(self, destination):
    for name in ('manifest.json', 'source_snapshot.zip', 'runs.zip'):
      shutil.copyfile(SOURCE / name, destination / name)

  def test_accepts_the_recorded_campaign(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      self.copy(destination)
      data = analyze(destination)
    self.assertEqual(data['processes'], 18)
    # Every process must agree that direct execution won its trace.
    self.assertTrue(all(row['wins'] == 3 for row in data['summary']))

  def test_recorded_trace_allocation_is_two_plans(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      self.copy(destination)
      data = analyze(destination)
    for row in data['summary']:
      self.assertEqual(row['trace_bytes'], 2 * row['storage_bytes'])

  def test_inputs_match_the_archived_transport_campaign(self):
    """The reuse claim is checked against the canonical archive, not asserted."""
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    canonical = ROOT / 'data/relevance-v2/manifest.json'
    if not canonical.is_file():
      self.skipTest('canonical transport archive not present')
    published = json.loads(canonical.read_text())['inputs']
    self.assertTrue(manifest['inputs'])
    for name, digest in manifest['inputs'].items():
      self.assertIn(name, published, name)
      self.assertEqual(published[name], digest, name)





# ===== Merged from test_evidence.py =====
"""Tampered or incomplete evidence must be rejected.

These checks guard the archived campaigns against silent corruption. They need
no GPU and make no timing assertions. Each archive test stages a private copy,
alters one thing, and expects the matching validator to refuse it.
"""
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from archive_inputs import digest, input_zip, split_inputs
from benchmark_runs import RunRecord, validate_run, CSV_FIELDS
from build_verified import validate_build_receipt
from prepare_transport import prepare, convergence_study, INPUT_DIRECTORY
import run_dg
import run_relevance
from run_relevance import validate_inputs, expected_inputs, sha

DG = ROOT / 'data/dg'
RELEVANCE = ROOT / 'data/relevance-v2'


def stage(source, destination):
  """Copy an archive directory. The large input parts are never rewritten by a
  test, so they are hard-linked instead of copied."""
  for path in source.iterdir():
    if path.name.startswith('inputs.zip.'):
      os.link(path, destination / path.name)
    else:
      shutil.copyfile(path, destination / path.name)


def rewrite_zip(path, change):
  with zipfile.ZipFile(path) as archive:
    contents = {name: archive.read(name) for name in archive.namelist()}
  change(contents)
  with zipfile.ZipFile(path, 'w') as archive:
    for name, payload in contents.items():
      archive.writestr(name, payload)


class ArchiveTestCase(unittest.TestCase):
  source: Path
  analyze = None

  def assert_rejected(self, error, manifest=None, archive=None, exception=ValueError):
    """Stage the archive, apply `manifest(m)` and/or `archive(directory)`, expect refusal."""
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      stage(self.source, destination)
      if manifest:
        data = json.loads((destination / 'manifest.json').read_text())
        manifest(data)
        (destination / 'manifest.json').write_text(json.dumps(data))
      if archive:
        archive(destination)
      with self.assertRaisesRegex(exception, error):
        type(self).analyze(destination)


class RunRecordTests(unittest.TestCase):
  def record(self, transform=lambda x: x, returncode=0):
    rows = [dict(method=m, position=str(p), iteration=str(i), gpu_ms='1', host_ms='2')
            for p, m in enumerate(('direct', 'baseline')) for i in range(2)]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(transform(rows))
    return RunRecord('test', text.getvalue(), dict(returncode=returncode))

  def test_valid(self):
    validate_run(self.record(), ['direct', 'baseline'], 2)

  def test_failed_process(self):
    with self.assertRaises(ValueError): validate_run(self.record(returncode=1), ['direct', 'baseline'], 2)

  def test_corrupt_trials(self):
    mutations = [lambda r: r[:-1], lambda r: [r[0], r[0], *r[2:]],
                 lambda r: [dict(x, gpu_ms='nan') for x in r],
                 lambda r: [dict(x, host_ms='0') for x in r],
                 lambda r: [dict(x, position='0') for x in r]]
    for mutation in mutations:
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        validate_run(self.record(mutation), ['direct', 'baseline'], 2)

  def test_stale_binary_and_source(self):
    valid = dict(sources={'src/a.cpp': 'source'}, executable_sha256='binary',
                 build_receipt=dict(sources={'src/a.cpp': 'source'}, executables={'test.exe': 'binary'}))
    validate_build_receipt(valid)
    for key, replacement in [('sources', {'src/a.cpp': 'changed'}), ('executable_sha256', 'stale')]:
      broken = copy.deepcopy(valid)
      broken[key] = replacement
      with self.assertRaises(ValueError): validate_build_receipt(broken)
    with self.assertRaises(ValueError): validate_build_receipt({})


class InputPartTests(unittest.TestCase):
  """A fresh checkout must reconstruct exactly the recorded input ZIP bytes."""
  def fixture(self, root):
    (root / 'inputs.zip').write_bytes(b'exact archive bytes for checksum regression')
    manifest = dict(complete=True, input_archive_sha256=digest(root / 'inputs.zip'))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    split_inputs(root)
    return json.loads((root / 'manifest.json').read_text())

  def test_checkout_without_original_zip(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory); manifest = self.fixture(root)
      (root / 'inputs.zip').unlink()
      with input_zip(root, manifest) as reconstructed:
        self.assertEqual(digest(reconstructed), manifest['input_archive_sha256'])

  def test_corrupt_or_missing_parts_rejected(self):
    def corrupt(root, manifest): (root / 'inputs.zip.000').write_bytes(b'corrupted')
    def drop(root, manifest): manifest['input_archive_parts'] = {}
    for change, error in ((corrupt, 'part checksum'), (drop, 'part coverage')):
      with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
        root = Path(directory); manifest = self.fixture(root)
        change(root, manifest)
        with self.assertRaisesRegex(ValueError, error):
          with input_zip(root, manifest): pass


class TransportInputTests(unittest.TestCase):
  """Input-archive validation on a small freshly prepared transport case."""
  @classmethod
  def setUpClass(cls):
    cls.directory = tempfile.TemporaryDirectory()
    cls.root = Path(cls.directory.name)
    payloads = cls.root / INPUT_DIRECTORY
    prepare(256, 8, 8, payloads)
    convergence_study(payloads)
    jobs = [dict(kind='transport', args=[(INPUT_DIRECTORY / 'transport_e256_n8').as_posix(), 8, 8, 6, 1])]
    names = expected_inputs(jobs)
    with zipfile.ZipFile(cls.root / 'inputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
      for name in names: archive.write(cls.root / name, name)
    cls.manifest = dict(jobs=jobs, inputs={name: sha(cls.root / name) for name in names},
                        input_archive_sha256=sha(cls.root / 'inputs.zip'))

  @classmethod
  def tearDownClass(cls): cls.directory.cleanup()

  def test_valid_archive(self): validate_inputs(self.root, self.manifest)

  def test_altered_manifest_is_rejected(self):
    def drop_all(m): m['inputs'] = {}
    def drop_one(m): m['inputs'] = dict(list(m['inputs'].items())[1:])
    def zero_digest(m): m['inputs'][next(n for n in m['inputs'] if n.endswith('.bin'))] = '0' * 64
    def change_steps(m): m['jobs'][0]['args'][2] = 32
    for change, error in ((drop_all, 'Input manifest coverage'), (drop_one, 'Input manifest coverage'),
                          (zero_digest, 'Input payload checksum'), (change_steps, 'metadata differs')):
      broken = copy.deepcopy(self.manifest)
      change(broken)
      with self.subTest(change=change.__name__), self.assertRaisesRegex(ValueError, error):
        validate_inputs(self.root, broken)

  def test_rehashed_corrupt_archive_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with zipfile.ZipFile(self.root / 'inputs.zip') as source, zipfile.ZipFile(root / 'inputs.zip', 'w') as target:
        for name in source.namelist():
          payload = source.read(name)
          if name.endswith('.reference'): payload = b'bad' + payload[3:]
          target.writestr(name, payload)
      broken = dict(self.manifest, input_archive_sha256=sha(root / 'inputs.zip'))
      with self.assertRaisesRegex(ValueError, 'Input payload checksum'):
        validate_inputs(root, broken)


@unittest.skipUnless((RELEVANCE / 'manifest.json').exists(), 'campaign archive not present')
class RelevanceArchiveTests(ArchiveTestCase):
  source = RELEVANCE
  analyze = run_relevance.analyze

  def test_altered_manifest_is_rejected(self):
    def add_argument(m): m['records'][0]['command'].append('unexpected')
    def drop_process(m): m['records'].pop()
    def stale_receipt(m): m['build_receipt']['sources']['src/kernels/row_owned.cu'] = '0' * 64
    def raw_hash(m): m['records'][0]['stdout_sha256'] = '0' * 64
    for change, error in ((add_argument, 'command'), (drop_process, 'Missing process'),
                          (stale_receipt, 'Build inputs'), (raw_hash, 'Raw checksum')):
      with self.subTest(change=change.__name__):
        self.assert_rejected(error, manifest=change)


@unittest.skipUnless((DG / 'manifest.json').exists(), 'campaign archive not present')
class DgArchiveTests(ArchiveTestCase):
  source = DG
  analyze = run_dg.analyze

  def test_altered_manifest_is_rejected(self):
    def reps(m): m['reps'] += 1
    def methods(m): m['methods'] = ['direct']
    def stale_receipt(m): m['build_receipt']['sources']['bench/dg_specialized.cpp'] = '0' * 64
    def no_inputs(m): m.pop('inputs')
    def drop_input(m): m['inputs'].pop(sorted(m['inputs'])[0])
    def input_digest(m): m['inputs'][next(n for n in sorted(m['inputs']) if n.endswith('.bin'))] = '0' * 64
    for change, error in ((reps, 'design mismatch'), (methods, 'Comparator set'),
                          (stale_receipt, 'compiled inputs'), (no_inputs, 'Input coverage mismatch'),
                          (drop_input, 'Input coverage mismatch'), (input_digest, 'checksum mismatch')):
      with self.subTest(change=change.__name__):
        self.assert_rejected(error, manifest=change)

  def test_altered_archives_are_rejected(self):
    def unreadable_snapshot(d): (d / 'source_snapshot.zip').write_bytes(b'not a zip archive')
    def edit_snapshot(d):
      def change(c):
        c[next(n for n in c if n.endswith('bench/dg_specialized.cpp'))] += b'\n// tampered\n'
      rewrite_zip(d / 'source_snapshot.zip', change)
    def drop_snapshot_member(d):
      rewrite_zip(d / 'source_snapshot.zip', lambda c: c.pop(next(n for n in c if n.endswith('scripts/run_dg.py'))))
    def append_trial(d):
      def change(c):
        c[next(n for n in c if n.endswith('.csv'))] += b'direct,0,99,1.0,1.0\n'
      rewrite_zip(d / 'runs.zip', change)
    for change, error, exception in ((unreadable_snapshot, '', Exception),
                                     (edit_snapshot, 'Snapshot checksum mismatch', ValueError),
                                     (drop_snapshot_member, 'Snapshot coverage mismatch', ValueError),
                                     (append_trial, 'checksum mismatch', ValueError)):
      with self.subTest(change=change.__name__):
        self.assert_rejected(error, archive=change, exception=exception)

  def test_implausible_structure_records_are_rejected(self):
    cases = [
        ('identity_negative_control,rejected\n', 'Missing specialized structure'),
        ('specialized_structure,8,10,4096\n', 'Missing specialized structure'),
        # Operators as large as the assembled matrix would mean no reuse.
        ('specialized_structure,8,10,4096,4096,8192,16384\n', 'Implausible specialized structure'),
        # Plan storage below the operator bytes.
        ('specialized_structure,8,10,34560,35389440,1024,2048\n', 'Implausible specialized structure'),
        # The timed trace builds one plan per direction, so a record claiming a
        # single plan's bytes would understate what the comparator retains.
        ('specialized_structure,8,10,34560,35389440,231168,231168\n', 'Implausible specialized structure'),
    ]
    for line, error in cases:
      with self.subTest(line=line), self.assertRaisesRegex(ValueError, error):
        run_dg.structure({'environment': line})





# ===== Merged from test_input_subnormals.py =====
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





# ===== Merged from test_transport.py =====
"""Transport physics and campaign-design checks."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_transport import prepare, require_identity_rejection, INPUT_DIRECTORY, translation_block
from run_relevance import design


class TransportTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.directory = tempfile.TemporaryDirectory()
    cls.root = Path(cls.directory.name)
    payloads = cls.root / INPUT_DIRECTORY
    prepare(256, 8, 8, payloads)

  @classmethod
  def tearDownClass(cls): cls.directory.cleanup()

  def test_unchanged_state_is_rejected(self):
    base = self.root / INPUT_DIRECTORY / 'transport_e256_n8'
    initial=np.fromfile(str(base)+'.input', dtype='<f4')
    reference=np.fromfile(str(base)+'.reference', dtype='<f4')
    require_identity_rejection(initial, reference)
    with self.assertRaisesRegex(ValueError, 'negative control'):
      require_identity_rejection(initial, initial)

  def test_constant_field_is_preserved_by_projection(self):
    for m,n in ((8,64),(16,8),(32,16),(64,32)):
      result=translation_block(m,m,False)[:,0]+translation_block(m,n,True)[:,0]
      expected=np.zeros(m); expected[0]=1
      np.testing.assert_allclose(result,expected,atol=2e-12,rtol=0)

  def test_core_crosses_seed_process_and_queue_mode(self):
    jobs=[j for j in design() if j['kind']=='core']
    self.assertEqual(len(jobs),2304)
    cells={}
    for j in jobs:
      key=tuple(j['args'][:4])+(j['args'][8],)
      cells.setdefault(key,set()).add((j['args'][4],j['process'],j['batch']))
      self.assertEqual(j['args'][5],20)
    self.assertEqual(len(cells),128)
    self.assertTrue(all(len(v)==18 for v in cells.values()))





# ===== Merged from test_workflow_coverage.py =====
"""The documented full rerun must produce every campaign the paper build reads.

`build_paper.ps1 -ResultsRoot <root>` consumes a fixed set of subdirectories
under that root. `run_final_validation.py` is the documented way to fill it.
A campaign added to one and not the other leaves the README workflow broken,
which is only discovered after hours of GPU time, so check it here.
"""
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'scripts/build_paper.ps1'
RUNNER = ROOT / 'scripts/run_final_validation.py'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_final_validation


def consumed():
  """Subdirectories the paper build reads from the results root."""
  text = BUILD.read_text(encoding='utf-8')
  return set(re.findall(r"Join-Path \$ResultsRoot '([^']+)'", text))


def produced():
  """Subdirectories the documented runner writes under the results root."""
  text = RUNNER.read_text(encoding='utf-8')
  return set(re.findall(r"str\(root / '([^']+)'\)", text))


class WorkflowCoverageTests(unittest.TestCase):
  def test_build_consumes_at_least_one_campaign(self):
    # Guards the regexes above against silently matching nothing.
    self.assertTrue(consumed())
    self.assertTrue(produced())

  def test_runner_produces_every_consumed_campaign(self):
    missing = consumed() - produced()
    self.assertEqual(missing, set(),
                     'run_final_validation.py does not produce: ' + ', '.join(sorted(missing)))

  def test_every_export_has_an_input_override(self):
    """Each export invoked with a results root must accept that root."""
    text = BUILD.read_text(encoding='utf-8')
    for script, folder in re.findall(
            r'python "\$PSScriptRoot/(\w+\.py)"[^\n]*Join-Path \$ResultsRoot \'([^\']+)\'', text):
      self.assertTrue((ROOT / 'scripts' / script).is_file(), script)
      self.assertIn(folder, produced(), f'{script} reads {folder}, which no runner produces')

  def test_dg_campaign_is_covered(self):
    # The specialized transport comparison is the most recent addition.
    self.assertIn('dg', consumed())
    self.assertIn('dg', produced())

  def test_runner_prepares_and_collects_relevance(self):
    with tempfile.TemporaryDirectory() as directory,          patch.object(sys, 'argv', ['run_final_validation.py', '--output-root', directory]),          patch.object(run_final_validation, 'verified_build', return_value=(Path('test.exe'), {})),          patch.object(run_final_validation, 'find_sanitizer', return_value=Path('sanitizer.exe')),          patch.object(run_final_validation.subprocess, 'run') as run:
      run.return_value.returncode = 0
      run_final_validation.main()
      commands = [call.args[0] for call in run.call_args_list]
      scripts = {Path(c[1]).name: c for c in commands if c[0] == sys.executable}
      self.assertIn('prepare_magma.py', scripts)
      self.assertIn('--all', scripts['prepare_transport.py'])
      self.assertIn('build_relevance.py', scripts)
      self.assertEqual(scripts['run_relevance.py'][-2:], ['--output', str(Path(directory) / 'relevance')])





if __name__ == '__main__':
  unittest.main(verbosity=2)
