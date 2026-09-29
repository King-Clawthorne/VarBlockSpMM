# VarBlockSpMM

## Situation

VarBlockSpMM multiplies a variable-block sparse matrix by a dense matrix:

```text
Y = A * X
```

The sparse matrix uses compressed sparse row indexing over blocks. Each dense
block is stored contiguously in column-major order. The input and output dense
matrices are also column-major, with the right-hand-side width selecting how
many columns are multiplied together.

## Task

The library validates matrix structure, owns device copies of matrix data, and
dispatches row-owned CUDA kernels. It also provides a double-precision CPU
reference implementation.

## Action

Include `src/vbsr.hpp` and link the CMake target
`VarBlockSpMM::varblockspmm` (or `varblockspmm`). The complete host-side
example constructs a `HostMatrix`, uploads it, creates a `Plan`, runs the GPU
operation, and checks the output in [`benchmark.cpp`](benchmark.cpp). The
benchmark uses device input and output buffers with `Plan::execute`; that call
is asynchronous, so keep the buffers alive until its CUDA stream has completed.
`cpu_reference` accepts `std::span<const float>`, allowing contiguous inputs to
be passed without copying.

### Matrix and plan lifetime

- `HostMatrix` owns the host-side CSR metadata and packed block values.
- `Matrix` validates the host matrix and owns device copies of all arrays.
- A `Plan` created from `Matrix` reuses the matrix's row ordering, so the
  `Matrix` must outlive the plan.
- A `Plan` created from a `DeviceMatrix` view copies metadata to host memory
  for validation, creates its own device row ordering, and does not copy the
  numerical values. The external device allocations must remain valid while
  the plan uses them.
- `Plan::execute` enqueues work on the supplied CUDA stream and returns before
  the result is ready. Keep input, output, and matrix allocations alive until
  that stream has completed.

### C++ features used

The host sources use C++26. The validation code uses C++26
`std::views::concat` to check row and column block dimensions in one pass.
C++23 `std::views::enumerate` keeps each dimension paired with its index, and
`std::ranges::contains` and `std::ranges::sort` handle generated block-column
selection. C++23 `std::ranges::to` builds row-order vectors directly from
sized `iota` views in both plan construction paths. The CPU reference API uses
`std::span` to accept contiguous inputs without requiring a vector.

CUDA translation units use CUDA C++23. NVCC's host pass for the `.cu` source
uses C++26 mode as well.

The NVIDIA CUDA compiler module shipped with CMake 4.2 does not register the
CUDA C++23 dialect, even though NVCC 13.3 and newer support it. The Windows
`build.ps1` configure step loads `cmake/EnableCuda23.cmake` before `project()`
to add CMake's missing CUDA23 standard-option mapping. This lets the project
set `CUDA_STANDARD 23` and generate one `-std=c++23` option without a manual
compiler-flag override.

### Requirements

- CMake 4.2 or newer
- CUDA Toolkit 13.3 or newer
- A host compiler and standard library supported by NVCC with C++26
  `std::views::concat` and C++23 ranges support. The WSL build is verified with
  GCC 15.3 and CUDA Toolkit 13.4.92.
- WSL2 on Windows. The MSVC/NVCC toolchain is rejected because it cannot compile
  this project in CUDA C++23 mode.
- On Linux, load `cmake/EnableCuda23.cmake` before `project()` (for example,
  with `-DCMAKE_PROJECT_INCLUDE_BEFORE=/path/to/VarBlockSpMM/cmake/EnableCuda23.cmake`)
  to supply CMake 4.2's missing NVIDIA CUDA23 mapping.

### Build on Linux

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel
```

### Build on Windows through WSL

`build.ps1` runs CMake and NVCC inside a WSL distribution and micromamba
environment. It expects micromamba at `~/.local/bin/micromamba`, its root at
`~/micromamba`, and an environment named `vbsr-cuda134`. Create the environment
in WSL with:

```bash
micromamba create -y -n vbsr-cuda134 -c nvidia -c conda-forge cuda=13.4 cmake=4.2
```

Then run this in PowerShell:

```powershell
.\build.ps1
```

Use `-WslDistribution` or `-WslEnvironment` to select different names.
`-Configuration` selects the CMake build type, and `-CudaArchitectures` selects
the CUDA target architecture.

### Benchmark

From WSL in the `vbsr-cuda134` environment, build the `vbsr_benchmark` target
with `cmake --build build --parallel`, then run it with an optional
iteration count (default `10000`):

```bash
./build/vbsr_benchmark 10000
/opt/nvidia/nsight-systems/2025.6.3/bin/nsys profile --stats=true --output build/vbsr-nsys ./build/vbsr_benchmark 100
ncu --set basic --launch-count 1 --export build/vbsr-ncu ./build/vbsr_benchmark 100
```

The benchmark warms up the GPU, times repeated RHS-32 executions with CUDA
events, and checks the output after timing. Its workload is a diagonal matrix
with 4096 dense 8-by-8 blocks.

## Result

The library provides a reusable CUDA implementation of `Y = A * X`, along with
host matrix generation and a CPU reference. See [`src/vbsr.hpp`](src/vbsr.hpp)
for the complete public API and its parameter types.
