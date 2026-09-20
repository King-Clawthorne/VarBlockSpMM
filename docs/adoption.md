# Using the standalone library

The installed package contains the public header, compiled library, CMake package configuration, and MIT license. Benchmark programs and paper tooling are optional consumers of that library.

## Requirements and tested configuration

Host and device sources use C++20. A full Release build and CUDA correctness suite pass with MSVC 19.44 and CUDA 13.4. A separate CUDA 13.1 library build with benchmarks disabled also passes the correctness suite on the same GPU. C++26 draft mode is not required.

CMake accepts CUDA 12.6 or newer. The grouped baseline calls `cublasSgemmGroupedBatched`, introduced in [cuBLAS 12.5](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates). The [CUDA 12.5 Update 1 release notes](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-toolkit-release-notes/index.html) identify the addition of BSR SpMM. The configured minimum is the next minor toolkit release, CUDA 12.6, to avoid accepting the initial CUDA 12.5 release. This establishes the API requirement, not a tested performance or portability claim for every toolkit. The RTX 5060 Ti needs a toolkit that supports its architecture.

The verified Python builder uses Visual Studio 2022 on Windows and CMake's default generator on other systems. Toolkit discovery uses `CUDAToolkit_ROOT`, then `CUDA_PATH`, with ordinary CMake discovery when neither is set. Compute Sanitizer discovery uses those toolkit roots and then `PATH`. Cross-platform discovery is implemented, but only the Windows configuration has been exercised here.

## Package validation

`examples/consumer` uses only `find_package(VarBlockSpMM CONFIG REQUIRED)` and the exported target. It uploads a matrix and dense input, executes the installed direct plan, waits for completion, and compares every scalar output against a CPU reference. It can be configured outside the source checkout using only the installed prefix.

The installed consumer was configured, built, and executed successfully, including its full CPU comparison. See the README for the install and consumer commands. The benchmark targets can be disabled with `VARBLOCKSPMM_BUILD_BENCHMARKS=OFF`, and the test target with `BUILD_TESTING=OFF`.

## Paper environment

`pyproject.toml` defines a `paper` dependency group that pins the required Python packages and their dependency closure from the Python 3.13 measurement environment. It deliberately does not copy unrelated packages from the global environment. `uv.lock` records the resolved distribution hashes. Run `uv sync --locked` and activate `.venv`, or prefix Python commands with `uv run --locked`. A clean virtual environment has passed the Python evidence tests and reproduced the archived geometry and binary checksum for the 2,048-point clustered seed-1 native input in a separate output directory. pdfLaTeX and pdftotext are separate system requirements.

## Ownership and licensing

Original code is MIT licensed by the project author. CUDA libraries are external NVIDIA dependencies. Downloaded matrix datasets remain subject to their providers' terms.
