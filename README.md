# VarBlockSpMM

VarBlockSpMM is a CUDA library for multiplying a sparse matrix stored as dense, variable-sized blocks by a dense matrix: `C = A * B`.

The public API is in [`src/vbsr.hpp`](src/vbsr.hpp). It provides validated host matrices, device matrix storage, deterministic matrix generation, a CPU reference, and reusable execution plans for the row-owned CUDA kernel.

## Requirements

- CMake 3.25 or newer
- CUDA Toolkit 12.6 or newer
- A C++20 and CUDA C++20 compiler supported by the installed toolkit

## Build

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release --parallel
```

The CMake target is `VarBlockSpMM::varblockspmm` (also available as `varblockspmm`). Include `vbsr.hpp` and link the target from a CMake consumer project.
