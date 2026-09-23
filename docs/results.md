# Results and evidence

The full measurements are in the [paper](../research/paper.pdf) and [supplement](../research/supplement.pdf). This page summarizes them.

## Headline results and when to use it

The comparison including **MAGMA variable-size batched GEMM** gives a **1.139x geometric-mean speedup over the fastest tested library**, with 105/128 direct wins on the full core grid. It tests both slot-based MAGMA and batched products followed by reduction. The replacement campaign crosses seeds 2, 3, and 5 with three fresh processes per seed and both single-product and eight-product queued timings. Each method receives twenty samples per process. Headline ratios average paired log ratios over all seeds and processes in single-product mode. Eight-product queued execution gives 1.131x and 101/128 configuration wins.

| RHS width | Over BSR8 | Over faster MAGMA | Over fastest library | Direct wins |
| --- | --- | --- | --- | --- |
| 8 | 1.083x | 2.385x | 1.060x | 27/32 |
| 16 | 1.090x | 2.098x | 1.045x | 24/32 |
| 32 | 1.128x | 1.593x | 1.088x | 22/32 |
| 64 | 1.446x | 2.019x | 1.398x | 32/32 |

A standalone variable-order discontinuous Galerkin transport program provides a dependent application trace. At 4,096 elements and RHS 64, its 128 steps to physical time `T = 1/128` are **1.439x faster than the fastest tested library** and **1.833x faster than the faster MAGMA composition**. Direct execution wins five of the six tested element-count/width configurations in all three process repetitions. The smallest RHS-8 case favors BSR8 in the process-averaged result, with one of three processes favoring direct. Each quarter-cell characteristic DG update translates and projects the field. The program checks every final coefficient against a CPU reference, checks actual GPU output against the analytic solution, and requires rejection of unchanged input. A separate fixed-time FP64 spatial-refinement study verifies convergence.

That trace also has a limit worth stating up front. A hand-written fused kernel that caches the operator's eight distinct blocks and derives the element structure arithmetically is **faster than direct execution on every one of those traces**, by 1.014x to 2.246x, including 1.359x on the 4,096-element RHS-64 case. Direct execution reaches 45% to 99% of that comparator's throughput, and 74% on the headline case. Direct execution beats the fastest tested general library in five of six configurations and the specialized batched-GEMM plan in all six, after averaging process ratios. A matched control attributes the large-case gap to operator sharing specifically. Where your operator repeats a few blocks across many rows, you know it, and you accept a kernel with structural assumptions to maintain, write it. The direct plan trades some throughput for support across the full block-format contract without problem-specific kernel code. Structural changes require a new plan, while value-only updates can reuse it.

The separate three-seed NVIDIA-library campaign retains its 1.139x individual-product and 1.132x queued-product results. Those measurements exclude MAGMA and use twenty timing samples per process. Their protocols and aggregates remain separate from this focused comparison. The [supplement](../research/supplement.pdf) retains detailed controls and historical evidence, and [follow-up documentation](relevance.md) gives the MAGMA compatibility boundary and transport equations.

Use the direct plan when your application already produces dense variable blocks and you want to retain that layout. Wider panels benefit from sharing input tiles across rows and reusing each sparse value across RHS columns. Widths 8 and 16 use a 64-thread CTA with full-panel accumulation, selected through measured comparisons with alternative kernels.

Direct execution can also be useful for matrices that change frequently or are used for only a few products, since converting to a library format has a setup cost. On the generated native component, startup from prepared host formats is 3.953 times faster than the best tested library alternative, including one completed product. Compact scalar CSR uses 1.948 times the explicit device storage of the direct representation on those inputs. The paper reports those costs separately from steady-state speed and distinguishes already prepared blocks from CPU assembly.

On that native component, BSR8 is faster in steady state, with a library-to-direct ratio of 0.390. Including CPU assembly also favors the libraries at every reported reuse count. The startup benefit therefore applies to the stated prepared-input starting point.

The paper's width table compares direct execution with both BSR8 and the fastest tested library method per case. Compact CSR remains the appropriate control for scattered scalar nonzeros, and dense SGEMM is included for high overall density. These comparisons help identify whether your input fits the direct kernel's strengths.

```cpp
vbsr::Matrix matrix(host);
vbsr::Plan plan(matrix, {32, vbsr::Kernel::Auto});
plan.execute(device_B, device_C, stream);
```

Here `host` contains your validated packed matrix, and the dense buffers use column-major layout. The matrix and buffers must remain alive until queued work completes.

## Comparisons and evidence

The evaluation includes:

- 168 synthetic configurations, each repeated in three fresh processes.
- A MAGMA-inclusive full core campaign with three seeds, three processes per seed, and two queue modes, retaining 2,304 core processes plus 93 transport, factor-control, and update processes.
- A separate NVIDIA-only full-core follow-up on matrix seeds 2, 3, and 5, comparing synchronization after one product with synchronization after eight queued products. Each seed and mode uses one process per configuration, for 768 processes.
- Explicit cuSPARSE CSR algorithms with persistent setup and preprocessing, cached grouped cuBLAS, and changing-address controls with stream-ordered device pointer generation.
- Padding-free BSR8 subdivision on every variable-block input, plus BSR32 on uniform inputs.
- Three published scalar sparse matrices with artificial partitions and compact-CSR controls.
- Persistent dense SGEMM on published and native inputs, including zero-fill expansion in startup costs.
- Twelve generated covariance near-field matrices with geometry-derived partitions, three geometry seeds, all four RHS widths, and three processes per instance. This benchmarks a dense-block application component, not a full hierarchical solver.
- Matched kernel controls at two sizes and degrees, with three matrix seeds and three processes per seed. RHS 8 uses a complete mapping/accumulator/CTA factorial design. Wider panels have matched accumulation, staging, and buffering controls.
- Native-input startup times, explicit device-array storage, and serial repeated-product cost models.

The tables report each panel width and the full comparison grid. Numerical results come directly from validated raw records, with historical tuning results archived separately. Performance measurements cover one GPU.

