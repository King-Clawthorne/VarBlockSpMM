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
`VarBlockSpMM::varblockspmm` (or `varblockspmm`). This example builds a valid
one-block matrix with an 8-by-8 block. The same setup works with matrices loaded
from a file or produced by your application:

```cpp
#include <vector>
#include "vbsr.hpp"

vbsr::HostMatrix host_matrix;
host_matrix.block_rows = 1;
host_matrix.block_cols = 1;
host_matrix.row_ptr = {0, 1};
host_matrix.block_col = {0};
host_matrix.row_size = {8};
host_matrix.col_size = {8};
host_matrix.row_scalar_off = {0, 8};
host_matrix.col_scalar_off = {0, 8};
host_matrix.value_off = {0, 64};
host_matrix.values.assign(64, 1.0f); // One dense 8-by-8 block, column-major.
host_matrix.validate();

constexpr int rhs_width = 32;
std::vector<float> input(host_matrix.scalar_cols() * rhs_width, 1.0f);
std::vector<float> reference =
    vbsr::cpu_reference(host_matrix, input, rhs_width);

vbsr::Matrix matrix(host_matrix);
vbsr::Plan plan(matrix, {.rhs_width = rhs_width});

// Allocate device_input and device_output, copy `input` to device_input,
// then call plan.execute(device_input, device_output). Copy the result back
// only after the CUDA stream used for execution has completed.
```

The example's reference result is computed on the CPU. For GPU execution,
`Plan::execute` takes device pointers for input and output; allocate those
buffers and copy `input` to the device before calling it. It enqueues work
asynchronously, so copy the output back only after the execution stream has
completed. `cpu_reference` also accepts `std::span<const float>`, allowing
other contiguous input buffers to be passed without copying.

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

## Result

The library provides a reusable CUDA implementation of `Y = A * X`, along with
host matrix generation and a CPU reference. See [`src/vbsr.hpp`](src/vbsr.hpp)
for the complete public API and its parameter types.
