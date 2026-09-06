# Review closure

Validated on 6 September 2026 on an RTX 5060 Ti, CUDA 13.4, Windows, and MSVC 19.44. The [paper](paper.pdf) and its [source](paper.tex) describe the final evidence. Earlier validation notes are archived history.

## Findings and resolutions

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

## Kernel improvement and measurement outcome

RHS 8 and 16 now use a 64-thread CTA with full-panel accumulation and shared input staging. Two development sweeps compared staged and warp-reduction candidates on seed 4. The final paired evaluation uses seeds 1, 2, and 3, with three processes per instance. The previous policy remains available as an experimental comparison, and the historical mapping benchmark is fixed to its original policy.

The new narrow kernel is 1.777 times faster at RHS 8 and 1.771 times faster at RHS 16 than the previous policy on the paired evaluation grid. Every one of its 24 matrix instances per width improves after averaging process ratios.

All 972 fresh evaluation processes completed successfully and retained 165,360 timed calls. The archives contain 504 synthetic, 36 published-matrix, 144 native-component, and 288 matched-control processes. All library source hashes agree across the four campaigns. The 96 development processes and their 11,520 timed calls are separate from these evaluation counts.

On the 128-case synthetic core, direct execution is 4.262 times faster than the CSR lower envelope, 5.757 times faster than cached grouped cuBLAS, and 1.175 times faster than BSR8. It wins 124 of 128 BSR8 comparisons. Against the fastest tested library option per process, including BSR32 on uniform blocks, the ratio is 1.137 with 103 direct wins. At RHS 64, direct execution wins all 32 comparisons against every tested library option. The README and paper report all four widths.

BSR32 wins 22 of 32 uniform cases, and compact CSR wins all twelve published-matrix configurations. On the higher-degree native component, BSR8 is faster in steady state, with a comparator-to-direct ratio of 0.390. Direct execution has lower aggregate startup cost from prepared host formats: the best-library-to-direct ratio is 3.953 including one product, and the modeled ten-product ratio is 2.668. Adding CPU assembly changes that conclusion, as the paper's separate assembly column shows. Compact scalar CSR uses 1.948 times the explicit device storage of the direct representation on these native inputs.

The paper now leads with the execution, startup, and storage benefits and identifies the input conditions where they apply. It retains the full comparison tables, including the cases where another implementation is preferable.

## Validation

- Release build and CTest pass with full-output reference checks.
- Compute Sanitizer memcheck reports zero errors. Racecheck filtered to project row-owned and pointer-refresh kernels reports zero errors and warnings. Logs are in `data/product-evaluation/validation/`.
- Four Python evidence regression tests pass. Both current campaign analyzers and the separate historical appendix analyzer pass.
- Independent archive enumeration confirms the process and trial totals and common kernel hash.
- The paper compiles in two passes. The final pass has no unresolved references or box warnings. All sixteen rendered pages were inspected for layout after replacing forced table placement with normal floats and preventing isolated paragraph lines at page breaks.
- The paper source, generated TeX tables, and extracted PDF text contain no en dashes, em dashes, or semicolons. The paper build enforces this restriction.

The superseded validated-revision campaign and its companion archives are preserved locally under `build/local-evidence-archive/data/`, including their snapshot-completion notes. The current collector selects compiled source extensions explicitly, and the `data/product-evaluation/` snapshots passed validation without that repair.

## Reproduction and scope

The [README](../README.md) gives fresh-output reproduction commands. The canonical campaigns are under `data/product-evaluation/`, in `synthetic/`, `published/`, `native/`, and `ablation/`. Native binaries are regenerated from retained geometry and checked against archived input hashes.

The evidence covers one GPU and a generated application component. It does not establish performance portability, a complete solver speedup, a new sparse format, or superiority over all research implementations. These are boundaries of the study rather than missing evidence for claims made in the paper.
