# VarBlockSpMM

## Situation

Sparse matrix multiplication can store nonzero values in dense blocks instead
of treating every scalar entry independently. When block dimensions vary across
the matrix, the representation can describe irregular data while still
exploiting dense operations inside each block. VarBlockSpMM is a CUDA C++20
library for multiplying such a matrix by a dense matrix:

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
- CUDA Toolkit 12.6 or newer
- A C++20 and CUDA C++20 compiler supported by the installed toolkit

### Build

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release --parallel
```

## Result

The result is a reusable CUDA library that computes `C = A * B` for a
variable-block sparse matrix and a dense matrix. `Plan::execute` enqueues the
work on the selected stream; synchronize that stream before reading the output
on the host or reusing buffers that are still involved in the operation.
