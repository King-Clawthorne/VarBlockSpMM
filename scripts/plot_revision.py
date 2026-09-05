"""Render publication figures from the analyzer's validated GPU ratios.

Chart contract: vector PDF figures for the paper, 6.7 inches wide.
The core distribution contains 128 equally weighted configurations per
comparator. Sorted ratio curves expose spread and losses without inferring
uncertainty. The format controls show 32 uniform cases and 12 application
cases, grouped by panel width. All ratios geometrically average three paired
process observations. Logarithmic ratio axes make parity explicit.
Blue, violet, and gray series also use distinct line styles or markers.
"""

from pathlib import Path
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


def ratio_axis(axis, limits, ticks):
    axis.set_yscale('log', base=2)
    axis.set_ylim(*limits)
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    axis.minorticks_off()
    axis.axhline(1, color='#333333', linestyle='--', linewidth=1)
    axis.grid(axis='y', color='#dddddd', linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)


def save(figure, output, name):
    figure.savefig(output / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(figure)


def render_figures(output: Path, data: dict):
    """Keep chart inputs auditable at full precision beside vector exports."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'pdf.fonttype': 42})
    figure, axis = plt.subplots(figsize=(6.7, 2.9), layout='constrained')
    for name, label, color, style in [
            ('csr_default', 'Default CSR', '#737373', ':'),
            ('csr_best', 'CSR lower envelope', '#225ea8', '-'),
            ('grouped_cached', 'Cached grouped cuBLAS', '#7651a8', '-.')]:
        values = sorted(row[name] for row in data['core'])
        axis.plot(range(1, len(values) + 1), values, label=label,
                  color=color, linestyle=style, linewidth=1.6)
    core_values = [row[name] for row in data['core']
                   for name in ('csr_default', 'csr_best', 'grouped_cached')]
    core_top = max(16, 2 ** math.ceil(math.log2(max(core_values) * 1.05)))
    core_bottom = min(0.75, min(core_values) / 1.05)
    ratio_axis(axis, (core_bottom, core_top),
               [2 ** power for power in range(math.floor(math.log2(core_bottom)) + 1,
                                               int(math.log2(core_top)) + 1)])
    axis.set(xlim=(1, 128), xticks=[1, 32, 64, 96, 128],
             xlabel='Configuration rank within each comparator (128 configurations)',
             ylabel='Comparator / optimized direct\nGPU time ratio (log scale)',
             title='Optimized kernel: core-grid library comparisons')
    axis.legend(loc='upper left', frameon=False)
    axis.text(0.99, 0.03, 'Above 1: optimized direct faster', transform=axis.transAxes,
              ha='right', fontsize=8)
    save(figure, output, 'core-distribution')

    figure, axes = plt.subplots(1, 2, figsize=(6.7, 3.1), sharey=True, layout='constrained')
    widths = [8, 16, 32, 64]
    control_values = [row['ratio'] for group in ('uniform', 'application') for row in data[group]]
    control_bottom = min(0.1, min(control_values) / 1.05)
    control_top = max(2, 2 ** math.ceil(math.log2(max(control_values) * 1.05)))
    for axis in axes:
        ratio_axis(axis, (control_bottom, control_top),
                   [2 ** power for power in range(math.floor(math.log2(control_bottom)) + 1,
                                                   int(math.log2(control_top)) + 1)])
        axis.set(xticks=range(4), xticklabels=widths, xlabel='Dense panel width (RHS)', xlim=(-0.35, 3.35))
    axes[0].set(ylabel='Comparator / optimized direct\nGPU time ratio (log scale)',
                title='(a) Uniform blocks: fixed-block BSR')
    for index, width in enumerate(widths):
        values = sorted(row['ratio'] for row in data['uniform'] if row['rhs'] == width)
        axes[0].scatter([index + (i - 3.5) * 0.04 for i in range(8)], values,
                        s=23, facecolors='none', edgecolors='#225ea8', linewidth=0.9)
    axes[0].text(0.03, 0.04, '8 configurations per width\nEach point averages 3 processes',
                 transform=axes[0].transAxes, fontsize=8)
    axes[1].set_title('(b) Artificial partitions: compact CSR')
    for name, color, marker in [('bcsstk13', '#225ea8', 'o'),
                                ('bcsstk14', '#7651a8', 's'),
                                ('bcsstk15', '#555555', '^')]:
        rows = sorted((row for row in data['application'] if row['matrix'] == name),
                      key=lambda row: row['rhs'])
        axes[1].plot(range(4), [row['ratio'] for row in rows], marker=marker,
                      color=color, linewidth=1, markersize=4, label=name.upper())
    axes[1].legend(loc='upper left', frameon=False, ncol=1)
    save(figure, output, 'format-controls')
