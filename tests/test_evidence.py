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


if __name__ == '__main__': unittest.main()
