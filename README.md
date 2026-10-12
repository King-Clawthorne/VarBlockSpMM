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
benchmark uses device input and output buffers with `Plan::execute`. That call
is asynchronous, so keep the buffers alive until its CUDA stream has completed.
`cpu_reference` accepts `std::span<const float>`, allowing contiguous inputs to
be passed without copying.

### Requirements

- CMake 4.2 or newer
- CUDA Toolkit 13.4 or newer
- A host compiler and standard library supported by NVCC with C++26
  `std::views::concat` and C++23 ranges support. The WSL build is verified with
  GCC 15.3 and CUDA Toolkit 13.4.92.
- Doxygen 1.13.2 and LLVM clang/clang-tools 22.1.8, which provide `doxygen`,
  `clang`, `clang-format`, `clang-tidy`, and `run-clang-tidy`.
- WSL2 on Windows. The MSVC/NVCC toolchain is rejected because it cannot compile
  this project in CUDA C++23 mode.
- On Linux, load `cmake/EnableCuda23.cmake` before `project()` (for example,
  with `-DCMAKE_PROJECT_INCLUDE_BEFORE=/path/to/VarBlockSpMM/cmake/EnableCuda23.cmake`)
  to supply CMake 4.2's missing NVIDIA CUDA23 mapping.

### Build on Linux

From Linux or WSL with the required CUDA Toolkit and host compiler available,
configure and build with the checked-in CMake preset:

```bash
cmake --preset wsl-release
cmake --build --preset wsl-release
```

The preset uses a Release configuration, targets the locally detected CUDA
architecture, and loads `cmake/EnableCuda23.cmake` before `project()`.

### Build on Windows through WSL

`build.ps1` runs CMake and NVCC inside a WSL distribution and micromamba
environment. It expects micromamba at `~/.local/bin/micromamba`, its root at
`~/micromamba`, and an environment named `vbsr-cuda134`. Create the environment
in WSL with:

```bash
micromamba create -y -n vbsr-cuda134 -c nvidia -c conda-forge cuda=13.4 cmake=4.2
```

The `build.ps1` script also requires Doxygen, clang, and clang-tools in that environment:

```bash
micromamba install -y -n vbsr-cuda134 -c conda-forge doxygen=1.13.2 clang-tools=22.1.8 clang=22.1.8
```

Then run this in PowerShell:

```powershell
.\build.ps1
```

The script formats the project's C and C++ sources, generates Doxygen HTML in
`build/docs/html`, and runs clang-tidy on every translation unit in parallel
using all available CPU cores before building. Use `-WslDistribution` or
`-WslEnvironment` to select different names. `-Configuration` selects the CMake
build type, and `-CudaArchitectures` selects the CUDA target architecture.

## Result

The library provides a reusable CUDA implementation of `Y = A * X`, along with
host matrix generation and a CPU reference. See [`src/vbsr.hpp`](src/vbsr.hpp)
for the complete public API and its parameter types.
