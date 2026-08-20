# Optimization report — 20 August 2026

## Environment

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB
- CUDA compiler/libraries: 13.4.46
- Driver reported by `nvidia-smi`: 610.57.01; CUDA UMD 13.3
- Build: MSVC 19.44, Release, native CUDA architecture, fast FP32 math
- Synthetic release grid: 1,024 block rows/columns, seed 1, 7 measured iterations after 3 warmups

Raw measurements are in `data/regime_map.csv` (128 workloads, the accepted hybrid kernel, the previous scalar kernel, and two library baselines). Treat the short release sweep as a regime map, not publication-grade statistics; rerun the supplied script with the README's 20/5 settings for longer measurements.

## Decisions backed by measurement

1. **Remove per-execute host synchronization.** The prototype copied terminal offsets from device to host and synchronized before every launch. Scalar dimensions now live in `DeviceMatrix`, making `Plan::execute` asynchronous and stream-correct.
2. **Specialize only RHS width and split output tiles across CTAs.** Four compiled RHS variants retain runtime block sizes. At high variance, degree 8, RHS 32, local columns, the release kernel measured 0.981 ms versus the prototype's recorded 1.303 ms on the same 1,024-row scale (about 25% lower). The output-CTA split is race-free because CTAs own disjoint `(row,RHS)` elements.
3. **Use persistent baseline plans.** CSR expansion, value upload, handles, and grouped shape construction occur once. Timed calls include only required execution work; grouped-GEMM device pointer-array marshaling remains in the hot path.
4. **Use real grouped GEMM by shape.** Each nonzero slot groups independent GEMMs with identical `(r_i,c_j)` and makes one `cublasSgemmGroupedBatched` call. This observes cuBLAS's no-overlapping-output rule inside each call while accumulating successive slots with beta=1.
5. **Reject split-row partial reduction for 1.0.** Before ILP optimization, row-owned won 112/128 workloads and grouped GEMM won the RHS-64/high-degree cases. A split-row kernel would add partial-C workspace and a reduction launch without addressing the measured dependency stalls.
6. **Reuse A across four RHS accumulators.** For RHS 16/32/64, one thread now computes four output columns for the same local row. Each A value feeds four independent FMAs, reducing repeated A loads and exposing independent B loads. RHS 8 retains the scalar mapping because the all-ILP candidate regressed 12/32 RHS-8 cases. The final hybrid had no regression greater than 5% for RHS 16/32/64 and mean scalar-to-hybrid speedups of 1.36x, 1.68x, and 1.85x respectively.

The accepted hybrid won all 128 workloads. Across the balanced grid, the arithmetic mean median-time ratios were 1.47 for previous-scalar/hybrid, 4.05 for cuSPARSE/hybrid, and 3.61 for grouped-GEMM/hybrid. These are grid summaries, not workload-weighted production speedups.

Representative 4,096-row measurements:

| Workload | Hybrid | Previous scalar | Scalar CSR | Grouped GEMM |
|---|---:|---:|---:|---:|
| uniform, local, degree 2, RHS 8 | 0.211 ms | 0.253 ms | 0.484 ms | 0.909 ms |
| high variance, random, degree 8, RHS 32 | 2.412 ms | 4.529 ms | 8.244 ms | 4.946 ms |
| bimodal, random, degree 16, RHS 64 | 7.940 ms | 16.130 ms | 26.589 ms | 10.819 ms |

Nsight Systems captured `data/nsys_high.nsys-rep` for the final representative case. Its kernel summary confirms one direct launch per call, one cuSPARSE main kernel plus helpers, and 16 grouped-GEMM kernels per call. CPU sampling/context-switch tracing was unavailable without elevation but CUDA tracing succeeded.

## Nsight Compute counter results

Counter access was enabled and the commands in `docs/profiling.md` were rerun. Raw exports are `data/ncu_low.csv`, `data/ncu_high.csv`, `data/ncu_high_detailed.csv`, and `data/ncu_high_warp.csv`. Profiler-instrumented durations are intentionally not used as benchmark timings; only hardware ratios and stall classifications are interpreted here.

| Counter | Low variance, degree 4, RHS 16, local | Bimodal, degree 16, RHS 64, random |
|---|---:|---:|
| DRAM throughput vs peak | 43.67% | 56.93% |
| SM throughput vs peak | 39.63% | 40.63% |
| Achieved occupancy | 98.84% | 99.54% |
| Registers/thread | 40 | 40 |

The detailed high-irregularity pass measured 82.23% L1/TEX hit rate but only 15.92% L2 hit rate, 187.3 GB/s memory throughput, 0 local/shared-memory spills, and 99.54% achieved occupancy. Despite 11.94 active warps per scheduler, only 0.48 were eligible per cycle; schedulers had no eligible warp in 71.5% of cycles. Warp-state analysis attributed 90.1% of the average issue interval to long-scoreboard waits on L1TEX operations.

The scalar kernel was therefore **memory-latency-bound, not occupancy-, register-, shared-memory-, or peak-bandwidth-bound** in the difficult RHS-64/high-degree case. This supported the four-accumulator ILP experiment.

## Accepted ILP result

The accepted RHS-64 ILP kernel reduced the final 20-repeat representative median from 16.130 ms to 7.940 ms (2.03x) and overtook grouped GEMM at 10.819 ms. Under identical detailed profiler sections, instrumented duration fell from 23.84 ms to 9.72 ms. Registers remained 40/thread, occupancy remained high at 96.45%, and there were still no spills. L1/TEX hit rate improved from 82.23% to 85.70% and L2 hit rate from 15.92% to 23.26%; DRAM utilization fell from 56.85% to 37.04% while SM utilization rose from 38.97% to 49.65%, consistent with reduced redundant A traffic and more useful work per load.

Long-scoreboard stalls remain dominant (89.5% of the ILP issue interval), so cooperative staging remains a possible future experiment. It is not required for this release: the ILP hybrid wins the entire controlled grid, has zero sanitizer findings, and adds no workspace or extra launch.
