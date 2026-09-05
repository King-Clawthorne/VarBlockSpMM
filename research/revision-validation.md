# Paper revision validation

Validated on 5 September 2026. The manuscript is ready to share as a scoped single-device implementation study. It does not establish state-of-the-art performance or validate native hierarchical application partitions.

## Changes prompted by review

| Issue | Resolution |
| --- | --- |
| Avoidable grouped baseline work | Added a cached-pointer mode that skips the full output clear for nonempty rows. Measured cached addresses and alternating addresses separately, with matching direct controls. |
| Default-only cuSPARSE comparison | Measured default CSR and explicit algorithms 1, 2, and 3, with preprocessing for 1 and 3. Reported the per-process lower envelope as an optimistic comparator. |
| Missing fixed-block control | Added cuSPARSE BSR algorithm 1 on every uniform-32 configuration. |
| Unsupported uncertainty attribution | Removed the variance attribution and the incorrect interval-width comparison. Reported descriptive repeated-process ranges with explicit limits. |
| Incorrect process description | Corrected the historical runner description to one process per configuration and matrix seed. The new campaign repeats each fixed instance in three processes. |
| Narrow workload coverage | Added matrix sizes, an imbalanced pattern with empty rows, and three published matrices with documented partitions and compact-CSR controls. |
| Results hidden by aggregate statistics | Added width and degree breakdowns, absolute direct times, BSR results, and application results. Reported the losing regimes explicitly. |
| Kernel description mismatch | Documented the two output CTAs used by RHS 8. Wider paths use one CTA per block row. |
| Layout and writing | Shortened the abstract, added an ownership diagram, kept each table intact near its discussion, and rewrote the conclusion around the new evidence. |
| Punctuation constraint | Checked source, generated table text, and extracted PDF text for en dashes, em dashes, and semicolons. None appear in the revised manuscript. |

## Verified results

Ratios are comparator time divided by direct time. Configuration ratios average three process medians in log space. All configurations receive equal weight.

| Comparator and scope | Geometric ratio | Direct wins |
| --- | ---: | ---: |
| Default CSR, 128 core configurations | 4.932 | 128/128 |
| CSR lower envelope, 128 core configurations | 3.036 | 123/128 |
| Cached grouped cuBLAS, 128 core configurations | 4.129 | 128/128 |
| Changing grouped cuBLAS against changing direct, 128 core configurations | 4.153 | 128/128 |
| Fixed BSR, 32 uniform configurations | 0.715 | 8/32 |
| Compact CSR lower envelope, 12 application configurations | 0.302 | 0/12 |
| Cached grouped cuBLAS, 12 application configurations | 4.878 | 12/12 |

The five core losses against the CSR lower envelope all use RHS 8. They are high-variance matrices at degrees 2 and 16 under both localities, and the random-locality bimodal matrix at degree 16.

Fixed BSR wins every uniform configuration at RHS 8, 16, and 32. Direct wins all eight uniform RHS-64 configurations. Compact CSR wins every tested published-matrix configuration. The artificial partitions have only 6.9 to 12.2 percent block fill, so these results do not justify applying the packed format to scalar sparse matrices indiscriminately.

## Evidence and calculations

- `data/revision/manifest.json` defines 168 synthetic configurations, three process observations each, and randomized job order.
- `data/revision/` contains 504 successful process records and 82,560 raw timing rows. Every timing row retains both GPU-event and synchronized host duration.
- `data/application/results/` contains 36 successful process records and 3,600 raw timing rows.
- Total new evidence comprises 540 process runs and 86,160 measured executions, excluding warmups.
- `data/application/bcsstk13.json`, `bcsstk14.json`, and `bcsstk15.json` record download URLs, archive checksums, input checksums, source dimensions, normalization, partitions, and block fill.
- `scripts/analyze_revision.py` checks configuration coverage, process coverage, method counts, trial indices, positive finite durations, process status, and source/input checksums before completing the result generation.
- An independent pandas calculation reproduced the core CSR ratio as 3.035879432659115 and the cached grouped ratio as 4.128752324182442.
- All historical publication data remain unchanged. The earlier manuscript and PDF are under `research/archive/`.

## Runtime validation

The Release build passed. CTest passed the expanded suite of 128 full parameter cases and 16 empty-row cases. The suite covers explicit CSR algorithms, uniform BSR, cached pointer reuse, changed pointer addresses with changed values, and non-default streams.

All 504 synthetic processes passed their CPU probes on the actual timed matrix. All 36 application processes passed full-output comparison against an independent compact-CSR CPU reference, checking both packing and computation.

Compute Sanitizer memcheck passed the expanded test executable with zero reported errors. Build, CTest, and memcheck logs are retained under `research/validation/`.

Racecheck restricted to the project's row-owned kernels passed with zero errors and zero warnings. The retained log is `research/validation/racecheck-direct.log`. An unrestricted run was stopped without a result after prolonged instrumentation. No racecheck claim is made for the library kernels. The successful command was:

```powershell
& "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/compute-sanitizer/compute-sanitizer.exe" --tool racecheck --kernel-name "regex=row_owned" --error-exitcode 9 build/Release/vbsr_tests.exe
```

## PDF validation

`scripts/build_paper.ps1` regenerated the numerical tables and built the PDF in two LaTeX passes. The PDF has nine pages. The build log contains no undefined references, overfull boxes, or underfull boxes. A rendered contact sheet and full-size table pages were inspected for clipping, overlap, table splits, and misplaced results. The current PDF is `research/paper.pdf`.

## Reproduction and provenance

Use fresh output directories for a new campaign. The runners reject existing data with different provenance. Run GPU workloads serially.

```powershell
scripts/build.ps1
python scripts/run_revision.py --output data/revision-rerun
python scripts/prepare_application.py
python scripts/run_application.py --output data/application/rerun
python scripts/analyze_revision.py --revision data/revision-rerun --application data/application/rerun --output build/rerun-tables
scripts/build_paper.ps1
```

The last command rebuilds the checked-in paper from the checked-in datasets. The preceding analysis command writes a separate set of tables for the fresh run.

The synthetic campaign used the source archived in `data/revision/source_snapshot.zip`, with every entry verified against its manifest checksum. Later formatting, API comments, the application build target, and runner improvements do not replace that measured source record. Application executable, source, and binary input checksums are recorded separately in its manifest.

## Subsequent code cleanup

The 504 synthetic process records are consolidated in `data/revision/runs.zip`, and the 36 application records are in `data/application/results/runs.zip`. Each archive retains the original CSV and JSON bytes with SHA256 checksums. The analysis reads these archives directly. Regenerating all eight generated artifacts after consolidation produced identical bytes. Both campaigns retain their measured source in `source_snapshot.zip` beside their manifests.

The benchmark entry points now share timing, validation, device ownership, and input loading helpers. The application benchmark no longer includes another executable's source file. Input parsing rejects malformed integer arguments and invalid compact CSR data. CUDA allocation and descriptor cleanup also covers construction failures.

After this cleanup, the Release build and CTest passed. Another 24 GPU smoke cases passed across all four panel widths, covering uniform, heterogeneous, and irregular synthetic matrices plus the three application matrices. An application memcheck run reported zero errors. Archive roundtrip, corruption rejection, invalid timing rejection, and manifest resume checks passed. Current build, CTest, smoke, and memcheck logs are under `build/code-quality-*`. These checks do not replace the archived campaign or establish new performance results.

## Remaining evidence limits

Only one GPU and one local operating environment were tested. Clocks were not locked, and unrelated operating-system activity was not eliminated. Three processes are a limited repeatability sample. The synthetic revision fixes one matrix seed. The published matrices use artificial partitions and do not validate native dense variable blocks from a hierarchical application. Setup amortization, full memory accounting, compatible research kernels, and additional architectures remain outside the claims.
