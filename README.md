# VarBlockSpMM

**Fast GPU multiplication for sparse matrices made of dense blocks of different sizes.**

Many scientific codes produce sparse matrices built from small dense blocks. Standard GPU libraries make you convert those blocks to another format first. VarBlockSpMM skips the conversion and computes `C = A B` directly on your blocks.

**Result:** on average, 1.14x faster than the best NVIDIA or MAGMA library on the benchmarks we tested, and 1.44x faster on the largest real-solver test. 📄 [Read the paper](research/paper.pdf)

## Use it

```cpp
#include <varblockspmm/vbsr.hpp>

vbsr::Matrix matrix(host);                         // your block matrix
vbsr::Plan plan(matrix, {32, vbsr::Kernel::Auto}); // 32 = number of columns in B
plan.execute(device_B, device_C, stream);          // C = A * B
```

Build it with `scripts/build.ps1`. The build needs CUDA 12.6 or newer and CMake 3.25 or newer.

MIT licensed. See [LICENSE](LICENSE).
