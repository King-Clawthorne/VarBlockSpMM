# VarBlockSpMM 1.0

CUDA FP32 non-transpose `C = A B` for a column-major variable-block sparse matrix `A` and dense column-major panels `B/C`. Block heights and widths independently vary over `{8,16,...,64}`; blocks are packed without global-size padding.

## Situation

Variable-block sparse matrix multiplication is difficult to map efficiently to a GPU. Irregular block dimensions create uneven work, global padding wastes storage and bandwidth, and generic sparse or dense library paths do not consistently fit every workload regime. The original scalar row-owned kernel was memory-latency-bound, especially for wide right-hand-side panels and high-degree rows.

## Task

The project set out to provide a reusable and verifiable CUDA implementation that:

- Supports independently variable block heights and widths from 8 through 64.
- Keeps blocks tightly packed without global-size padding.
- Avoids atomics and temporary workspace in the direct kernel.
- Handles right-hand-side widths of 8, 16, 32, and 64.
- Compares the direct path with persistent cuSPARSE and grouped-cuBLAS baselines.
- Measures correctness and performance across representative size, degree, and locality regimes.

## Action

The implementation includes:

- A validated host format and owning GPU format with 64-bit scalar and value offsets.
- Deterministic generators for uniform, low-variance, high-variance, and bimodal block sizes with local or random column patterns.
- A double-accumulating CPU reference for correctness checks.
- RHS-specialized row-owned CUDA kernels. RHS 16 and 32 reuse each `A` load across eight independent output accumulators, RHS 64 uses sixteen, and RHS 8 retains the lower-overhead scalar mapping.
- RHS 32 uses a measured row-shape dispatch. Rows up to 16 scalars high use a 128-thread single-buffer CTA; taller rows use 256 threads and asynchronous double buffering so the next `B` slice loads while the current block computes. Mixed-height matrices use the split only from mean degree 8 upward, where two launches amortize.
- RHS 64 retains one 256-thread, single-buffer CTA per row. Two full RHS-64 buffers reduce occupancy enough to lose performance on the release GPU; RHS 16 likewise retains direct global loads.
- A persistent scalar-CSR cuSPARSE plan.
- A persistent slot-split plan using CUDA 13.4 `cublasSgemmGroupedBatched`, grouped by row and column block size.
- A 64-case correctness matrix covering every supported RHS width, all distributions, degrees `{1,4,8,16}`, both locality modes, and non-default streams.
- A reproducible 128-case regime sweep reporting GPU-event and synchronized host median/p95 timing.

Nsight Compute identified memory latency as the scalar kernel's primary constraint. The direct kernel was tuned by panel width to reuse each matrix value across eight or sixteen RHS accumulators, then changed to load reusable `B` slices cooperatively for wide panels. Neither optimization introduces atomics or external workspace.

## Result

The optimized hybrid direct kernel won all 128 workloads in the 1,024-row regime grid, including the former RHS-64/high-degree grouped-GEMM regime. The measured release therefore does not include split-row partial buffers.

The asynchronous RHS-32 follow-up improves geometric-mean median time by 1.11x over the single-buffer staged release across its 32-case slice. It wins all degree-8 and degree-16 RHS-32 cases, reaching 1.17x on the 1,024-row bimodal/random/degree-16 case (0.468 ms versus 0.549 ms). Low-degree mixed shapes retain the original single-buffer launch, while RHS 64 is deliberately unchanged. The updated direct path still beats both persistent library baselines in all 128 cases.

For the representative degree-16/RHS-64 workload, the staged hybrid:

- Reduced median time from 4.887 ms for the unstaged wide-ILP kernel to 2.895 ms (1.69x), and from 8.430 ms for the four-accumulator intermediate (2.91x).
- Improved the complete 128-case release grid by a cumulative 1.46x geometric-mean speedup over the checked-in four-accumulator release.
- Won every grid case against the persistent cuSPARSE and grouped-cuBLAS baselines.
- Compiled with 56 and 64 registers per thread plus 9,344 and 17,664 bytes of shared memory per CTA for RHS 32 and 64 respectively, with no local-memory spills.

Unprofiled benchmark timings provide the speedups. Nsight Compute independently confirms that staging cuts long-scoreboard stalls, but its replay-instrumented durations are not mixed into the benchmark results.

## Build and verify on Windows

The project requires CMake 3.25 or newer, CUDA 13.4, a host compiler with C++26 draft support, and a CUDA architecture supported by the installed toolkit. Host `.cpp` files compile with `/std:c++latest` on MSVC or `-std=c++2c` on other compilers. Because CUDA 13.4 does not provide a CUDA C++26 mode, `.cu` files use its newest supported dialect, CUDA C++20.

Use the build script to configure, compile, and run the tests:

```powershell
scripts\build.ps1
```

Equivalent commands are:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

For deeper CUDA validation:

```powershell
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\compute-sanitizer\compute-sanitizer.exe" --tool memcheck --error-exitcode 9 build\Release\vbsr_tests.exe
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\compute-sanitizer\compute-sanitizer.exe" --tool racecheck --error-exitcode 9 build\Release\vbsr_tests.exe
```

## Benchmark

```powershell
build\Release\vbsr_benchmark.exe --rows 4096 --degree 8 --rhs 32 --distribution high --locality random --warmup 10 --reps 50 --seed 1
scripts\run_grid.ps1 -Rows 4096 -Reps 20 -Warmup 5
```

WSL users can run `ROWS=4096 REPS=20 WARMUP=5 bash scripts/run_grid.sh`. Results are written to `data/regime_map.csv`. Format construction and host-to-device input transfer are excluded; required hot-path pointer marshaling for grouped GEMM is included.

## API lifetime

`Matrix` owns immutable GPU structure and values. `Plan` holds a non-owning view, so the matrix must outlive the plan. `execute` is asynchronous on the supplied stream. The baseline plan types own their expanded or copied formats and may be reused, but one instance must not be executed concurrently from multiple host threads because each owns a library handle and mutable descriptors.

See [the design](docs/design.md), [the optimization report](docs/optimization-report.md), and [the prior-art review](docs/prior-art.md).
