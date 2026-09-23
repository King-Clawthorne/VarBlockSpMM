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


if __name__=='__main__': unittest.main()
