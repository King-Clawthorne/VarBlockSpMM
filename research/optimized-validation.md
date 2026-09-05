# Optimized-kernel paper validation

The current paper evaluates the optimized direct dispatch, including eight-row
RHS-8 tiles for matrices with at least 512 block rows and the scalar fallback
for smaller grids. The synthetic and application campaigns were rerun with the
current executable. Main results are not extrapolated from the scalar ablation.

## Results

Ratios are comparator time divided by optimized direct time. Configuration
ratios give equal weight to three paired process medians in log space.

| Comparator | Geometric ratio | Direct wins |
| --- | ---: | ---: |
| CSR lower envelope, core grid | 3.518 | 128/128 |
| Cached grouped cuBLAS, core grid | 4.769 | 128/128 |
| Changing grouped cuBLAS, core grid | 4.769 | 128/128 |
| Fixed BSR, uniform subset | 0.834 | 9/32 |
| Compact CSR, application inputs | 0.296 | 0/12 |
| Cached grouped cuBLAS, application inputs | 6.924 | 12/12 |

Direct wins all 128 core configurations against the CSR lower envelope and
cached grouped cuBLAS. Fixed BSR wins all uniform RHS-8 and RHS-16 cases, seven
of eight RHS-32 cases, and none of the eight RHS-64 cases. Compact CSR wins all
12 application cases. Those application partitions have 68, 61, and 132 block
rows, so their RHS-8 executions use the small-grid fallback.

The cached grouped application ratio has a descriptive process-label range
of 6.020 to 7.851. The paper reports this variability. These ranges
are not confidence intervals, and clocks were not locked.

## Evidence and provenance

- `data/optimized-revision/` contains 168 configurations, 504 successful processes,
  and 82,560 raw timing rows.
- `data/application/optimized-results/` contains 36 successful processes and
  3,600 raw timing rows for the three published matrices.
- All 540 processes passed their correctness checks. Synthetic runs use 64 CPU
  output probes per method. Application runs use full-output comparison against
  an independent compact-CSR CPU reference.
- Both campaigns use five warmups and 20 measured calls per method, randomized
  method order, and serial GPU execution. The three processes repeat each fixed
  matrix instance. Matrix seed 1 is fixed in the synthetic campaign.
- Each campaign retains raw CSV and process records in `runs.zip`, per-entry
  checksums, its executable checksum, and its source snapshot and manifest.
- The synthetic and application kernel source hashes match each other and the
  current working-tree kernel: `480305b815bca081efcf1536baaf168b07a5d833fcfc6b13a13b2723c8e28fc9`.
- Independent calculation from the raw synthetic archive reproduced the core
  CSR ratio 3.518341423220429 and grouped ratio 4.768925726606555,
  and both 128/128 win counts. The check is retained in the validation directory.

## Paper and graphs

`scripts/analyze_revision.py` defaults to the optimized campaigns. It validates
run coverage, trial indices, finite positive GPU and host durations, process
success, input checksums, and both source snapshots. It also requires matching
kernel source hashes across the two campaigns. Explicit paths can still select
the historical datasets.

All main tables, numerical macros, `figure-data.json`, `core-distribution.pdf`,
and `format-controls.pdf` were regenerated from the optimized campaign. Figure
data retain provenance. Both PDF graphs and their source data differ from the
previous versions. The plot axes expand to include all observed ratios.

The scalar-versus-tiled comparison remains an ablation, generated separately by
`scripts/analyze_kernel.py`. It is not used to multiply or adjust library ratios.
The current manuscript has twelve pages. The LaTeX build reports no undefined
references or overfull/underfull boxes. Rendered graph pages and the full-paper
contact sheet were checked for clipping, layout, and numerical consistency.

## Build and runtime validation

The Release build passed before the campaigns. The kernel source is unchanged
from the validated tiled implementation. Its full-output CTest, memcheck, and
targeted racecheck records remain in
`data/kernel-rhs8-final-20260906/validation/`. Memcheck reported zero errors and
the tiled-kernel racecheck reported zero errors and warnings. The current
campaign's build log, paper build log, and independent calculation are retained
under `data/optimized-revision/validation/`.

## Reproduction

Run GPU campaigns serially and use fresh output directories:

```powershell
scripts/build.ps1
python scripts/run_revision.py --output data/revision-rerun
python scripts/run_application.py --output data/application/rerun
python scripts/analyze_revision.py --revision data/revision-rerun --application data/application/rerun --output build/rerun-tables
scripts/build_paper.ps1
```

The final command rebuilds the checked-in paper from the optimized datasets.
The separate rerun tables do not overwrite those datasets. Earlier measurements
remain in `data/revision/` and `data/application/results/`, with their validation
history in `research/revision-validation.md`.

The evidence remains limited to one GPU and operating environment. The
application matrices use artificial low-fill partitions. The results establish
neither native hierarchical-application performance nor superiority over
compatible research kernels.
