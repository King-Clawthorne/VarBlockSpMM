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
- A persistent scalar-CSR cuSPARSE plan.
- A persistent slot-split plan using CUDA 13.4 `cublasSgemmGroupedBatched`, grouped by row and column block size.
- A 64-case correctness matrix covering every supported RHS width, all distributions, degrees `{1,4,8,16}`, both locality modes, and non-default streams.
- A reproducible 128-case regime sweep reporting GPU-event and synchronized host median/p95 timing.

Nsight Compute identified memory latency as the scalar kernel's primary constraint. The direct kernel was then tuned by panel width to reuse each matrix value across eight or sixteen RHS accumulators, increasing instruction-level parallelism without introducing atomics or workspace.

## Result

The optimized hybrid direct kernel won all 128 workloads in the 1,024-row regime grid, including the former RHS-64/high-degree grouped-GEMM regime. The measured release therefore does not include split-row partial buffers.

For the representative degree-16/RHS-64 workload, the wider-ILP follow-up:

- Reduced median time from 8.430 ms for the previous four-accumulator hybrid to 4.887 ms.
- Improved the complete 128-case release grid by a 1.24x geometric-mean speedup.
- Won every grid case against the persistent cuSPARSE and grouped-cuBLAS baselines.
- Compiled with 52–64 registers per thread and no stack, shared-memory, or local-memory allocation in the wider kernels.

The earlier Nsight Compute counters in the optimization report describe the four-accumulator intermediate. The wider kernel is accepted from unprofiled benchmark timings and the complete correctness matrix; profiler-instrumented durations are not mixed into those speedups.

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
