"""Fresh checkouts must reconstruct exactly the recorded input ZIP bytes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from archive_inputs import digest, input_zip, split_inputs


class InputPartTests(unittest.TestCase):
  def fixture(self, root):
    (root/'inputs.zip').write_bytes(b'exact archive bytes for checksum regression')
    manifest=dict(complete=True,input_archive_sha256=digest(root/'inputs.zip'))
    (root/'manifest.json').write_text(json.dumps(manifest))
    split_inputs(root)
    return json.loads((root/'manifest.json').read_text())

  def test_checkout_without_original_zip(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory); manifest=self.fixture(root)
      (root/'inputs.zip').unlink()
      with input_zip(root,manifest) as reconstructed:
        self.assertEqual(digest(reconstructed),manifest['input_archive_sha256'])

  def test_corrupt_part_rejected_even_with_local_zip(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory); manifest=self.fixture(root)
      (root/'inputs.zip.000').write_bytes(b'corrupted')
      with self.assertRaisesRegex(ValueError,'part checksum'):
        with input_zip(root,manifest): pass

  def test_missing_parts_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory); manifest=self.fixture(root)
      manifest['input_archive_parts']={}
      with self.assertRaisesRegex(ValueError,'part coverage'):
        with input_zip(root,manifest): pass


if __name__=='__main__': unittest.main()
