# VarBlockSpMM 1.0

CUDA FP32 non-transpose `C = A B` for a column-major variable-block sparse matrix `A` and dense column-major panels `B/C`. Block heights and widths independently vary over `{8,16,...,64}`; blocks are packed without global-size padding.

## What ships

- Validated host format and owning GPU format with 64-bit scalar/value offsets.
- Deterministic generators for uniform, low-variance, high-variance, and bimodal sizes plus local/random column patterns.
- Double-accumulating CPU reference.
- RHS-specialized row-owned CUDA kernels. RHS 16/32/64 reuse each A load across four independent output accumulators; RHS 8 uses the lower-overhead scalar mapping. All paths require no atomics or workspace.
- Persistent scalar-CSR/cuSPARSE plan.
- Persistent slot-split plan using CUDA 13.4 `cublasSgemmGroupedBatched`, grouped by `(row size, column size)`.
- 64-case correctness matrix for every allowed RHS width, all distributions, degrees `{1,4,8,16}`, both locality modes, and non-default streams.
- Reproducible 128-case regime sweep with GPU-event median/p95 and synchronized host hot-path median/p95.

The measured release does not include split-row partial buffers. After the four-column ILP optimization, the hybrid direct kernel won all 128 workloads in the 1,024-row regime grid, including the former RHS-64/high-degree grouped-GEMM regime.

Nsight Compute identified the previous scalar kernel as memory-latency-bound. Reusing each A value across four RHS accumulators reduced the final representative degree-16/RHS-64 median from 16.130 ms to 7.940 ms, kept registers at 40/thread with 96.45% achieved occupancy and no spills, raised SM utilization from 38.97% to 49.65%, and improved L2 hit rate from 15.92% to 23.26%.

## Build and verify on Windows

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure

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

`Matrix` owns immutable GPU structure and values. `Plan` holds a non-owning view, so the matrix must outlive the plan. `execute` is asynchronous on the supplied stream. The baseline plan types own their expanded/copied formats and may be reused, but one instance must not be executed concurrently from multiple host threads because each owns a library handle and mutable descriptors.

See [docs/design.md](docs/design.md), [docs/optimization-report.md](docs/optimization-report.md), and [docs/prior-art.md](docs/prior-art.md).
