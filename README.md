# VarBlockSpMM

CUDA FP32 non-transpose `C = A B` for packed variable-block sparse `A` and column-major dense `B/C`. Block heights and widths independently vary over `{8,16,...,64}`, and RHS widths are `{8,16,32,64}`.

VarBlockSpMM executes directly on your packed blocks. It avoids scalar CSR expansion and fixed-block retiling, uses no output atomics, and allocates no temporary workspace during execution. The [paper](research/paper.pdf) measures execution speed, startup, and storage on an RTX 5060 Ti.

## When to use it

Across the full 128-case core on three additional matrix seeds, direct execution has a **1.139x geometric-mean speedup over the fastest tested library for individual products** and **1.132x for batches of eight queued products**. It wins 308/384 individual-product comparisons and 304/384 queued comparisons on one RTX 5060 Ti.

| RHS width | Individual speedup | Queued speedup | Individual wins | Queued wins |
| --- | --- | --- | --- | --- |
| 8 | 1.056x | 1.050x | 75/96 | 73/96 |
| 16 | 1.046x | 1.034x | 72/96 | 72/96 |
| 32 | 1.091x | 1.086x | 65/96 | 64/96 |
| 64 | 1.395x | 1.392x | 96/96 | 95/96 |

Each width includes four shape distributions, two locality patterns, four degrees, and matrix seeds 2, 3, and 5 at 4,096 block rows. Each seed and timing mode has one fresh process per configuration. The library comparison includes explicit CSR algorithms, cached grouped cuBLAS, BSR8, and BSR32 on uniform inputs. The geometric-mean speedups over BSR8 alone are 1.178x for individual products and 1.173x for queued products.

The original seed-1 campaign remains in the paper as a separate three-process comparison. Its best-library ratio is 1.137x with 103/128 wins. The new seed ranges and timing-mode results support the same width-dependent performance pattern.

Use the direct plan when your application already produces dense variable blocks and you want to retain that layout. Wider panels benefit from sharing input tiles across rows and reusing each sparse value across RHS columns. Widths 8 and 16 use a 64-thread CTA with full-panel accumulation, selected through measured comparisons with alternative kernels.

Direct execution can also be useful for matrices that change frequently or are used for only a few products, since converting to a library format has a setup cost. On the generated native component, startup from prepared host formats is 3.953 times faster than the best tested library alternative, including one completed product. Compact scalar CSR uses 1.948 times the explicit device storage of the direct representation on those inputs. The paper reports those costs separately from steady-state speed and distinguishes already prepared blocks from CPU assembly.

On that native component, BSR8 is faster in steady state, with a library-to-direct ratio of 0.390. Including CPU assembly also favors the libraries at every reported reuse count. The startup benefit therefore applies to the stated prepared-input starting point.

The paper's width table compares direct execution with both BSR8 and the fastest tested library method per case. Compact CSR remains the appropriate control for scattered scalar nonzeros, and dense SGEMM is included for high overall density. These comparisons help identify whether your input fits the direct kernel's strengths.

```cpp
vbsr::Matrix matrix(host);
vbsr::Plan plan(matrix, {32, vbsr::Kernel::Auto});
plan.execute(device_B, device_C, stream);
```

Here `host` contains your validated packed matrix, and the dense buffers use column-major layout. The matrix and buffers must remain alive until queued work completes.

## Comparisons and evidence

The evaluation includes:

- 168 synthetic configurations, each repeated in three fresh processes.
- A separate full-core follow-up on matrix seeds 2, 3, and 5, comparing synchronization after one product with synchronization after eight queued products. Each seed and mode uses one process per configuration, for 768 processes.
- Explicit cuSPARSE CSR algorithms with persistent setup and preprocessing, cached grouped cuBLAS, and changing-address controls with stream-ordered device pointer generation.
- Padding-free BSR8 subdivision on every variable-block input, plus BSR32 on uniform inputs.
- Three published scalar sparse matrices with artificial partitions and compact-CSR controls.
- Persistent dense SGEMM on published and native inputs, including zero-fill expansion in startup costs.
- Twelve generated covariance near-field matrices with geometry-derived partitions, three geometry seeds, all four RHS widths, and three processes per instance. This benchmarks a dense-block application component, not a full hierarchical solver.
- Matched kernel controls at two sizes and degrees, with three matrix seeds and three processes per seed. RHS 8 uses a complete mapping/accumulator/CTA factorial design. Wider panels have matched accumulation, staging, and buffering controls.
- Native-input startup times, explicit device-array storage, and serial repeated-product cost models.

The tables report each panel width and the full comparison grid. Numerical results come directly from validated raw records, with historical tuning results archived separately. Performance measurements cover one GPU.

## Build and correctness

The library uses C++20 for both host and CUDA sources and requires CMake 3.25 or newer. The configured toolkit minimum is CUDA 12.6 to include both grouped GEMM and generic BSR SpMM. The measured Windows configuration is CUDA 13.4 with Visual Studio 2022 Build Tools. Earlier toolkits and other operating systems have not been runtime validated. Set `CUDAToolkit_ROOT` to choose a toolkit explicitly. The verified runner also checks `CUDA_PATH`, while ordinary CMake builds use CMake's toolkit discovery.

```powershell
scripts/build.ps1
```

The full-output tests poison output before each comparison, reject a deliberately empty operation, check CUDA errors, cover all 64 block shapes at every RHS width, and include mixed and completely empty rows. They check BSR8 conversion, explicit CSR algorithms, queued input/output address changes, and non-default streams.

## Reproduce the paper

The paper environment uses Python 3.13 and uv. Install its pinned dependencies with `uv sync --locked`, then activate `.venv` before running the commands below (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux). The `paper` dependency group captures NumPy, SciPy, Requests, Matplotlib, and their required dependencies from the validated environment. PDF building additionally requires pdfLaTeX and pdftotext.

```powershell
python scripts/prepare_application.py
python scripts/prepare_native.py
python scripts/run_final_validation.py --output-root data/rerun
powershell -File scripts/build_paper.ps1 -ResultsRoot data/rerun
```

The final runner performs a verified build, CTest, memcheck, and filtered racecheck, then runs every GPU campaign serially into the specified fresh root. It can take substantial time. Benchmark runners verify a build receipt tying compiled source hashes to the actual executable bytes, and snapshot the source used for each campaign. Existing records are resumed only with matching provenance and commands. Run `scripts/build_paper.ps1` without `-ResultsRoot` to rebuild the checked-in paper from its canonical campaigns instead.

Use individual runners and fresh output directories for independent repetitions:

```powershell
python scripts/run_revision.py --output data/revision-rerun
python scripts/run_application.py --output data/application/rerun
python scripts/run_supplement.py native --output data/native-rerun
python scripts/run_supplement.py ablation --output data/ablation-rerun
python scripts/analyze_revision.py --revision data/revision-rerun --application data/application/rerun --output build/rerun-tables
python scripts/analyze_supplement.py --native data/native-rerun --ablation data/ablation-rerun --output build/rerun-tables
python -m unittest discover -s tests -p 'test_*.py'
```

Native binary arrays are large and excluded from Git. Their generator, geometry, partition metadata, assembly observations, and checksums are retained. `prepare_native.py` reconstructs identical binaries while preserving the archived CPU assembly observations. `--output` with a fresh directory remeasures preparation.

The canonical campaigns are under `data/product-evaluation/`, in `synthetic/`, `published/`, `native/`, and `ablation/`. Sanitizer records are in its `validation/` subdirectory. The analyzers reject checksum, build receipt, design, command, process, method, and timing inconsistencies before emitting tables.

## API contract

`HostMatrix` requires positive block dimensions, supported block sizes, sorted unique block columns within each row, and consistent packed payloads. Rows may be empty. Scalar and value offsets are 64-bit, while block indices are 32-bit. Scalar CSR expansion and grouped GEMM explicitly reject sizes that exceed their narrower index or leading-dimension limits.

`Matrix` owns immutable GPU storage. `Plan` holds a non-owning view: the allocation must outlive the plan and all queued work, and must not be replaced by move assignment while referenced. Dense input and output buffers must be sufficiently sized, non-overlapping, and on the same CUDA device. Calls enqueue work on the supplied stream.

Library plans own mutable handles and metadata. Order calls to one plan on one stream, or externally synchronize across streams and host threads. Grouped GEMM refreshes pointers on the execution stream and safely supports queued address changes.

See the [design](docs/design.md), [prior-art discussion](docs/prior-art.md), and [review closure record](research/review-closure.md).

## Install and use from another project

VarBlockSpMM is a standalone library. The consumer example checks its installed public API without depending on the benchmark code.

```powershell
cmake -S . -B build/package -DCMAKE_CUDA_ARCHITECTURES=native -DVARBLOCKSPMM_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=OFF
cmake --build build/package --config Release
cmake --install build/package --config Release --prefix "$PWD/build/install"
cmake -S examples/consumer -B build/consumer "-DCMAKE_PREFIX_PATH=$PWD/build/install"
cmake --build build/consumer --config Release
ctest --test-dir build/consumer -C Release --output-on-failure
```

Consumers use `find_package(VarBlockSpMM CONFIG REQUIRED)` and link `VarBlockSpMM::varblockspmm`. The export supplies headers, the C++20 requirement, and CUDA library dependencies. Choose a CUDA architecture supported by the toolkit and target GPU. The native architecture used in local measurements does not establish binary portability to other GPUs.

## License

Original project code is available under the [MIT license](LICENSE). NVIDIA libraries and the downloaded Harwell-Boeing matrix inputs retain their respective third-party terms. The project license does not relicense those dependencies or input datasets.

## Additional seeds and queued throughput

```powershell
python scripts/run_robustness.py --output data/robustness-rerun
python scripts/export_robustness.py --input data/robustness-rerun
```

The default paper build validates the retained `data/robustness/` archive. A full `run_final_validation.py` rerun also collects this follow-up under its output root. The original three-process, seed-1 campaign remains separately identifiable. Queued measurements report time per product across batches of eight with one synchronization per batch. They use the same mathematical operation and retained library sources.
