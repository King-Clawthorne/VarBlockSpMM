# VarBlockSpMM

## Situation

Sparse matrix multiplication can store nonzero values in dense blocks instead
of treating every scalar entry independently. When block dimensions vary across
the matrix, the representation can describe irregular data while still
exploiting dense operations inside each block. VarBlockSpMM uses host C++26
and CUDA C++23 to multiply such a matrix by a dense matrix:

```text
C = A * B
```

The sparse matrix uses compressed sparse row (CSR) indexing at the block level.
Each dense block is stored contiguously in column-major order. The dense input
and output matrices are also column-major, with the right-hand-side width
selecting how many columns are multiplied at once.

## Task

The library takes responsibility for validating the variable-block structure,
moving matrix data to the GPU, and executing the multiplication with a
row-owned CUDA kernel. It also provides a deterministic matrix generator and a
CPU reference implementation for applications that need those utilities.

## Action

Include `vbsr.hpp` and link the CMake target `VarBlockSpMM::varblockspmm` (or
`varblockspmm`). The public API is documented in [`src/vbsr.hpp`](src/vbsr.hpp).
Typical use is to generate or construct a `HostMatrix`, create a device-owning
`Matrix`, configure a reusable `Plan`, and call `Plan::execute` with device
input and output buffers. Execution is asynchronous with respect to the
provided CUDA stream.

### Requirements

- CMake 3.25 or newer
- CUDA Toolkit 13.3 or newer for CUDA C++23 device support
- A C++26-capable host compiler supported by NVCC, such as GCC 14+ or Clang 18+ on Linux
- The MSVC/NVCC toolchain is rejected because it cannot compile this project in CUDA C++23 mode

### Build

For the requested host C++26 and CUDA C++23 modes, use Linux or WSL2 with a
supported GCC or Clang host compiler and CUDA Toolkit 13.3 or newer:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel
```

On Windows, `build.ps1` delegates the build to the configured WSL distribution
and micromamba environment. It expects micromamba at `~/.local/bin/micromamba`,
its root at `~/micromamba`, and an environment named `vbsr-cuda134`. Create
that environment in WSL with:

```bash
micromamba create -y -n vbsr-cuda134 -c nvidia -c conda-forge cuda=13.4 cmake=4.2
```

Override `-WslDistribution` or `-WslEnvironment` if your WSL names differ:

```powershell
.\build.ps1
```

## Result

The result is a reusable CUDA library that computes `C = A * B` for a
variable-block sparse matrix and a dense matrix. `Plan::execute` enqueues the
work on the selected stream; synchronize that stream before reading the output
on the host or reusing buffers that are still involved in the operation.
