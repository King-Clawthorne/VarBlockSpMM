# Review closure

Reviewed on 13 September 2026 on an RTX 5060 Ti, CUDA 13.4, Windows, and MSVC 19.44. The [paper](paper.pdf) and [supplement](supplement.pdf) contain the current results. Dated campaign notes below preserve the protocols and validation performed at those stages.

## Specialized aggregation correction

A subsequent review found that the specialized transport analyzer used medians of process ratios while the manuscript described geometric process averaging. The earlier documentation check compared exports with prose and did not detect this methodological mismatch. The earlier completion assessment missed it.

The analyzer now takes trial medians within each process and geometric means across processes for both ratios and absolute times. The three-process extrema and observed win counts are unchanged. Overall fused speedup changes from 1.293 to 1.274, and sharing at 256 elements with RHS 8 changes from 1.027 to 1.077. The separate 1.139 library headline is unaffected. Tables, macros, README, and method notes use the corrected aggregation. Retained raw measurements are unchanged.

An independent regression reads raw CSVs from the retained ZIP, computes trial medians without the analyzer helper, and checks every process aggregate using products and roots. This distinguishes geometric averaging from the former median reduction. Repeated discussion was compressed and detailed FP32 edge cases moved to the supplement, with unfavorable comparisons retained in the main paper. All 51 Python tests pass. Restoring the old median reduction in a temporary runtime patch makes the new regression reject 48 incorrect aggregates. The paper build validates retained evidence, documentation, references, and punctuation. The 17-page paper and eight-page supplement were rendered and visually inspected. No GPU measurements were rerun.

## Earlier consistency review

The current review covered the public API and implementation paths, numerical acceptance and timing helpers, retained evidence validation, repeated public claims, installation, and rendered documents. It found documentation inconsistencies and did not identify a new numerical correctness failure in the reviewed paths.

- Corrected the error acceptance rule in both README and public header. Passing requires both maximum absolute error and relative L2 error to meet their thresholds.
- Corrected the stale specialized batched-comparator range in the method notes and removed the stale duplicate fused range from the README.
- Replaced claims of no operator-sharing effect with the observed small process-averaged differences. Neither equivalence nor absence of an effect is established.
- Qualified the transport library comparison as five of six configurations, while the specialized batched comparator loses all six after process averaging. Removed wording suggesting the general kernel recovers the throughput it gives up or accepts structural changes without replanning.
- Added `scripts/check_documentation.py` to the canonical paper build. It checks repeated numerical claims against validated exports and checks both copies of the numerical acceptance contract. Regression tests reject the previously stale ranges and the incorrect acceptance conjunction.

A clean CUDA Release build and both CTest tests pass. The installed external consumer also passes its runtime reference comparison. All 50 Python tests pass, including archive corruption and documentation-drift regressions. The paper build validates all retained campaigns, references, and punctuation. All 18 main-paper pages and eight supplement pages were rendered as PNGs and visually inspected. No clipping or unresolved references were found. One underfull bibliography line is a minor spacing warning. No new performance campaigns or sanitizer runs were performed in this review. Current compiled source and executable hashes match the clean-build receipt.

## Explicit study boundaries

An independently captured workload, external solver adoption, a second GPU architecture, and a broader algorithmic novelty claim remain outside the demonstrated evidence. The manuscript states these limits. The transport program is an implemented standalone application component written for this evaluation, not an independently sourced production workload. The fused transport comparator wins its process-averaged cases, BSR8 wins the covariance steady-state comparison, and compact CSR wins the published sparse-matrix controls. These results bound the recommendation and remain visible in the paper.

This review does not turn those limits into demonstrated results or guarantee that future review will find no defects. Extending those claims requires new research, rather than another wording change.

## Replacement campaign validation

The replacement campaign in `data/relevance-v2/` completed all 2,397 processes
in 11,065.2 seconds. The following resolves review findings 1, 2, 3, and 5.

- Transport now uses characteristic DG projection at `dt=h/4` and a common
  physical horizon `T=1/128`. All six inputs reject unchanged state with a
  tenfold margin on both coefficient criteria. All 432 actual GPU analytic
  checks pass, with maximum relative physical L2 error `1.70636e-6`. The
  independent FP64 refinement errors are `1.05856e-6`, `6.25343e-9`, and
  `3.47850e-11` for 16, 32, and 64 elements at fixed physical time.
- Exact transport payloads are archived in checked 64 MiB ZIP parts. A
  parts-only temporary checkout passed the full 2,397-process analyzer.
  Missing hashes, altered payloads, bad command metadata, and corrupt or
  missing parts are covered by rejection regressions. The original unsplit
  ZIP remains a local convenience and is excluded from Git.
- The MAGMA core crosses three seeds, three processes per seed, and two
  timing modes, with twenty samples per method. Independent raw-CSV reduction
  gives ratios `1.1390938758` and `1.1306437714` for individual and queued
  products. Configuration wins are 105/128 and 101/128. The corrected large
  transport trace gives `1.4385603561` over the fastest tested library.
- The complete reproduction runner includes MAGMA preparation, all transport
  inputs, comparison build and CTest, and collection under `relevance/`.
  An orchestration regression checks that routing. The paper build also
  passed through a results-root view of the validated retained campaigns.
  The earlier campaigns were not remeasured for this check.
- Both CUDA correctness suites passed the rebuilt comparison. Transport
  Compute Sanitizer memcheck reported zero errors. All 21 Python regression
  tests pass. The 16-page paper and six-page supplement compile twice with
  no reference or box warnings and pass the punctuation checks.

The current paper and README use the replacement results. The earlier notes
below retain their original protocols and numbers as historical evidence.

## Earlier findings and resolutions

| Finding | Resolution and evidence |
| --- | --- |
| Missing fixed subdivision comparator | Added persistent, padding-free BSR8 conversion and cuSPARSE execution for every supported block shape. Measured it throughout the synthetic, published, and native campaigns. |
| Application motivation lacked a concrete dense-block workload | Added twelve reproducible covariance near-field inputs with geometry-derived partitions, independent scalar assembly, and exact FP32 checks of every packed block. Included dense SGEMM because these components can have high overall density. The paper limits the claim to this component. |
| Mechanism comparisons changed several factors together | Added a complete RHS-8 mapping, accumulator, and CTA factorial experiment. Wider controls match CTA size and output mapping for accumulation and staging, and row order for buffering. Every measured variant passes a full CPU output comparison. |
| Correctness tests could reuse valid output | Poisoned output before every implementation, added a no-op rejection regression, checked CUDA statuses, and exercised all 64 shapes at every RHS width, empty matrices, non-default streams, and queued address changes. |
| Grouped pointer refresh depended on temporary host arrays | Moved pointer generation to the execution stream using persistent device offsets. Tests queue different input and output addresses without intermediate host synchronization. |
| Source snapshots did not identify the executable | Added clean-build receipts with source and executable hashes, per-process executable hashes, and strict archive, command, design, input, and timing validation. Python regressions reject corrupted trials and stale receipts. |
| Narrow baseline indices could overflow | Added explicit scalar CSR entry and dimension limits and grouped GEMM leading-dimension guards. Host metadata validation rejects duplicate columns and checks offsets without overflowing subtraction. |
| Steady-state timings omitted lifecycle costs | Recorded native-input synchronized startup, explicit device storage, CPU preparation observations, and repeated-product cost models. The paper states the starting representation and excluded allocations. |
| Legacy CLI could accept malformed arguments or time unchecked output | Added strict numeric parsing, timing preconditions, CUDA checks, full CPU validation, and correct even-sample medians. |

## Original kernel improvement and measurement outcome

RHS 8 and 16 now use a 64-thread CTA with full-panel accumulation and shared input staging. Two development sweeps compared staged and warp-reduction candidates on seed 4. The final paired evaluation uses seeds 1, 2, and 3, with three processes per instance. The previous policy remains available as an experimental comparison, and the historical mapping benchmark is fixed to its original policy.

The new narrow kernel is 1.777 times faster at RHS 8 and 1.771 times faster at RHS 16 than the previous policy on the paired evaluation grid. Every one of its 24 matrix instances per width improves after averaging process ratios.

All 972 fresh evaluation processes completed successfully and retained 165,360 timed calls. The archives contain 504 synthetic, 36 published-matrix, 144 native-component, and 288 matched-control processes. All library source hashes agree across the four campaigns. The 96 development processes and their 11,520 timed calls are separate from these evaluation counts.

On the 128-case synthetic core, direct execution is 4.262 times faster than the CSR lower envelope, 5.757 times faster than cached grouped cuBLAS, and 1.175 times faster than BSR8. It wins 124 of 128 BSR8 comparisons. Against the fastest tested library option per process, including BSR32 on uniform blocks, the ratio is 1.137 with 103 direct wins. At RHS 64, direct execution wins all 32 comparisons against every tested library option. The README and paper report all four widths.

BSR32 wins 22 of 32 uniform cases, and compact CSR wins all twelve published-matrix configurations. On the higher-degree native component, BSR8 is faster in steady state, with a comparator-to-direct ratio of 0.390. Direct execution has lower aggregate startup cost from prepared host formats: the best-library-to-direct ratio is 3.953 including one product, and the modeled ten-product ratio is 2.668. Adding CPU assembly changes that conclusion, as the paper's separate assembly column shows. Compact scalar CSR uses 1.948 times the explicit device storage of the direct representation on these native inputs.

The paper now leads with the execution, startup, and storage benefits and identifies the input conditions where they apply. It retains the full comparison tables, including the cases where another implementation is preferable.

## Follow-up review resolutions

The standalone scope is explicit. No external solver integration or complete solver speedup is claimed. The later focused campaign below measures a standalone transport component. The contribution is the packed-block execution policy and its measured performance boundary against library conversions.

The complete 128-configuration core was repeated on independent matrix seeds 2, 3, and 5, with one fresh process per seed, configuration, and timing mode. All 768 processes completed, retaining 639,360 timed products. Individual-product synchronization gives a fastest-library-to-direct geometric mean of 1.139 with 308 wins out of 384. Queuing eight products before synchronization gives 1.132 with 304 wins. RHS 64 reaches about 1.39 in both modes. Every width has an aggregate advantage on each new seed in both modes. These observations complement the original three-process campaign without treating seeds as repeated processes.

The retained `data/robustness/` manifest, source snapshot, and raw archive validate against the measured executable. Independent raw aggregation reproduces the reported totals and ratios. The library source hash matches the original campaigns. Later CLI overflow guards and build configuration corrections are outside the archived measured harness.

Adoption changes include C++20, toolkit discovery, an exported CMake package, an installed consumer with full CPU output comparison, and the author-selected MIT license. `pyproject.toml` and `uv.lock` define the Python environment. A fresh environment passes the evidence tests and reproduces a retained native input's geometry and binary checksum.

## Focused resolutions for findings 1, 2, 3, and 5

The approved eight-minute campaign completed in 299.6 seconds. Its 221 processes retain 37,776 timed products under `data/relevance/`, with exact commands, build receipts, input hashes, source snapshots, raw results, and the upstream MAGMA license. No old ablations were rerun.

- **Application relevance:** Added a standalone periodic upwind modal DG transport component with variable polynomial orders and complete dependent 32-step traces. Five of six configurations beat the fastest tested library in all three processes. The largest RHS-64 trace is 1.438 times faster than the best tested library and 1.823 times faster than the better MAGMA composition. Every method is checked against a double-precision reference, and analytic transport error is below 4.2e-7. This is a short component trace, not a full solver or comparison with DG-specific algebraic implementations.
- **Closest implementation:** Built pinned upstream MAGMA variable-size SGEMM kernels and measured persistent slot and product-plus-reduction compositions. The 128-case core gives a 1.138 fastest-library-to-direct geometric ratio, including MAGMA, with 103 direct wins. The prior work's 64 to 256 leaf range and this project's 8 to 64 block range are explicitly distinguished. No claim reproduces the complete prior hierarchical solver.
- **Performance boundary:** Added 24 crossed combinations of degree, block shape, row count, and panel width, each in three processes. Nested adjacency isolates degree changes, shape preserves adjacency, and row-count changes are identified as working-set changes too. Main contrasts use the common BSR8 comparator. The complete table is supplementary.
- **Device inputs and updates:** Added validated plans borrowing existing device allocations and value-only device updates that preserve existing plans. Stream ordering, allocation lifetime, and structural revalidation are documented and tested. Separate measurements report value-update and structural-replan path costs without calling them equal-work kernel speedups.

Native metadata now preserves its original archived CRLF bytes through `.gitattributes`, restoring exact input checksums without changing scientific content. The native fill description also acknowledges the few FP32 underflows instead of claiming every stored value is nonzero.

## Earlier focused-campaign validation

- Current CUDA 13.4 Release builds and both project and MAGMA CTest suites pass. The installed external consumer builds and passes its CPU reference comparison.
- Compute Sanitizer memcheck reports zero errors for the device API and both MAGMA compositions.
- All eight Python evidence regression tests pass. Current and historical analyzers validate, and independent raw aggregation reproduces the focused campaign totals and headline results.
- Current compiled source and executable hashes match their build receipts. The production kernel is unchanged from the archived main campaigns.
- The 15-page main paper and six-page supplement compile twice without reference or box warnings. Rendered PNG pages were visually inspected, including the final supplementary layout.
- Sources, generated TeX, and extracted PDF text pass the enforced prohibition on en dashes, em dashes, and semicolons.

## Earlier validation history

- Release build and CTest pass with full-output reference checks on CUDA 13.4. A separate CUDA 13.1 library build also passes CTest on the same GPU. The installed consumer builds and passes its runtime CPU comparison.
- Compute Sanitizer memcheck reports zero errors. Racecheck filtered to project row-owned and pointer-refresh kernels reports zero errors and warnings. Logs are in `data/product-evaluation/validation/`.
- Four Python evidence regression tests pass. Both current campaign analyzers and the separate historical appendix analyzer pass.
- Independent archive enumeration confirms the process and trial totals and common kernel hash.
- An eight-product queued audit passes all ten methods and Compute Sanitizer memcheck reports zero errors. Invalid batch sizes and repetition overflow are rejected before allocation.
- The paper compiles in two passes. The final pass has no unresolved references or box warnings. All seventeen rendered pages were inspected for layout after the follow-up results were added. Tables and figures fit without large interior blank spaces.
- The paper source, generated TeX tables, and extracted PDF text contain no en dashes, em dashes, or semicolons. The paper build enforces this restriction.

The superseded validated-revision campaign and its companion archives are preserved locally under `build/local-evidence-archive/data/`, including their snapshot-completion notes. The current collector selects compiled source extensions explicitly, and the `data/product-evaluation/` snapshots passed validation without that repair.

## Reproduction and scope

The [README](../README.md) gives fresh-output reproduction commands. The canonical campaigns are under `data/product-evaluation/`, in `synthetic/`, `published/`, `native/`, and `ablation/`. Native binaries are regenerated from retained geometry and checked against archived input hashes.

The evidence covers one GPU, generated covariance components, and a standalone transport trace. The focused MAGMA comparison is separate from the earlier multi-seed NVIDIA-only comparison. It does not establish performance portability, a complete solver speedup, a new sparse format, or superiority over all research implementations. These are boundaries of the study rather than missing evidence for claims made in the paper.
