# Historical development results

These measurements predate the final verified-build evaluation. They are not current library comparisons.

## Historical development results

The optimized hybrid direct kernel won all 128 workloads in the 1,024-row regime grid, including the former RHS-64/high-degree grouped-GEMM regime. The measured release therefore does not include split-row partial buffers.

The asynchronous RHS-32 follow-up improves geometric-mean median time by 1.11x over the single-buffer staged release across its 32-case slice. It wins all degree-8 and degree-16 RHS-32 cases, reaching 1.17x on the 1,024-row bimodal/random/degree-16 case (0.468 ms versus 0.549 ms). Low-degree mixed shapes retain the original single-buffer launch, while RHS 64 is deliberately unchanged. The updated direct path still beats both persistent library baselines in all 128 cases.

For the representative degree-16/RHS-64 workload, the staged hybrid:

- Reduced median time from 4.887 ms for the unstaged wide-ILP kernel to 2.895 ms (1.69x), and from 8.430 ms for the four-accumulator intermediate (2.91x).
- Improved the complete 128-case release grid by a cumulative 1.46x geometric-mean speedup over the checked-in four-accumulator release.
- Won every grid case against the persistent cuSPARSE and grouped-cuBLAS baselines.
- Compiled with 56 and 64 registers per thread plus 8,320 and 16,640 bytes of shared memory per CTA for RHS 32 and 64 respectively, with no local-memory spills.

Unprofiled benchmark timings provide the speedups. Nsight Compute independently confirms that staging cuts long-scoreboard stalls, but its replay-instrumented durations are not mixed into the benchmark results.

