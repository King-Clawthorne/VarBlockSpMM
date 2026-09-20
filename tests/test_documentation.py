"""Regressions for stale public results and acceptance-rule drift."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_documentation', ROOT / 'scripts/check_documentation.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DocumentationTests(unittest.TestCase):
    def test_current_claims(self):
        self.assertEqual(checker.check(), [])

    def test_stale_claims_are_rejected(self):
        changes = (
            ('docs/relevance.md', '1.367x to 5.318x', '1.373x to 5.303x'),
            ('README.md', '1.014x to 2.246x', '1.037x to 2.229x'),
            ('include/varblockspmm/vbsr.hpp', '5e-4 and relative', '5e-4 or relative'),
        )
        for path, before, after in changes:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for name in ('README.md', 'docs/relevance.md', 'include/varblockspmm/vbsr.hpp',
                             'research/generated/dg-macros.tex', 'research/generated/relevance-macros.tex'):
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, target)
                target = root / path
                text = target.read_text()
                self.assertIn(before, text)
                target.write_text(text.replace(before, after))
                self.assertTrue(checker.check(root))


if __name__ == '__main__':
    unittest.main()
