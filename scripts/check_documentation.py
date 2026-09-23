"""Check repeated public claims against the validated paper exports."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
  macros = {}
  for name in ('dg-macros.tex', 'relevance-macros.tex'):
    macros.update(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}',
                            (root / 'research/generated' / name).read_text()))
  # Headline claims appear in the README and are expanded in docs/results.md.
  readme = '\n'.join((root / name).read_text(encoding='utf-8') for name in ('README.md', 'docs/results.md'))
  notes = (root / 'docs/relevance.md').read_text()
  checks = {
      'README core ratio': (readme, f'**{macros["RelevanceCoreRatio"]}x geometric-mean'),
      'README core wins': (readme, f'{macros["RelevanceCoreWins"]}/128 direct wins'),
      'README queued ratio': (readme, f'queued execution gives {macros["RelevanceQueuedRatio"]}x'),
      'README transport ratio': (readme, f'**{macros["TransportLargeRatio"]}x faster'),
      'README MAGMA transport ratio': (readme, f'**{macros["TransportLargeMagmaRatio"]}x faster'),
      'README fused range': (readme, f'{macros["DgFusedMin"]}x to {macros["DgFusedMax"]}x'),
      'method notes fused range': (notes, f'{macros["DgFusedMin"]}x to {macros["DgFusedMax"]}x'),
      'method notes batched range': (notes, f'{macros["DgSpecializedMin"]}x to {macros["DgSpecializedMax"]}x'),
      'method notes storage': (notes, f'1/{macros["DgStorageFactor"]} of'),
  }
  errors = [name + ' differs from validated exports' for name, (text, expected)
            in checks.items() if expected not in text]
  # A second range can silently become stale even when the first is correct.
  expected_ranges = {(macros['DgFusedMin'], macros['DgFusedMax'])}
  ranges = re.findall(r'(\d+\.\d+)x to (\d+\.\d+)x', readme)
  if set(ranges) != expected_ranges:
    errors.append('README contains an unexpected speedup range')
  for path in ('docs/numerics.md', 'include/varblockspmm/vbsr.hpp'):
    text = (root / path).read_text(encoding='utf-8').replace('`', '')
    if not re.search(r'absolute error at most 5e-4 and relative L2 error at most 5e-5', text):
      errors.append(path + ' must require both error thresholds')
  return errors


if __name__ == '__main__':
  errors = check()
  if errors:
    raise SystemExit('\n'.join(errors))
  print('Public numerical claims match validated exports')
