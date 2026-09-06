# VarBlockSpMM

CUDA FP32 non-transpose `C = A B` for packed variable-block sparse `A` and column-major dense `B/C`. Block heights and widths independently vary over `{8,16,...,64}`, and RHS widths are `{8,16,32,64}`.

VarBlockSpMM executes directly on your packed blocks. It avoids scalar CSR expansion and fixed-block retiling, uses no output atomics, and allocates no temporary workspace during execution. The [paper](research/paper.pdf) measures execution speed, startup, and storage on an RTX 5060 Ti.

## When to use it

On the 128-case synthetic core grid, direct execution is **1.175 times faster than BSR8 overall**, winning **124 of 128 cases**. It is **1.137 times faster than the fastest tested library option overall**, winning **103 of 128 cases**. These are geometric means of paired process ratios on one RTX 5060 Ti.

| RHS width | Speedup over BSR8 | Speedup over fastest tested library | Wins over all tested libraries |
| --- | --- | --- | --- |
| 8 | 1.078x | 1.057x | 26/32 |
| 16 | 1.086x | 1.042x | 24/32 |
| 32 | 1.128x | 1.087x | 21/32 |
| 64 | 1.444x | 1.397x | 32/32 |

The library comparison includes explicit CSR algorithms, cached grouped cuBLAS, BSR8, and BSR32 on uniform blocks. Each width includes all four shape distributions, two locality patterns, and four degrees at 4,096 block rows, with three processes per configuration.

Use the direct plan when your application already produces dense variable blocks and you want to retain that layout. Wider panels benefit from sharing input tiles across rows and reusing each sparse value across RHS columns. Widths 8 and 16 use a 64-thread CTA with full-panel accumulation, selected through measured comparisons with alternative kernels.

Direct execution can also be useful for matrices that change frequently or are used for only a few products, since converting to a library format has a setup cost. On the generated native component, startup from prepared host formats is 3.953 times faster than the best tested library alternative, including one completed product. Compact scalar CSR uses 1.948 times the explicit device storage of the direct representation on those inputs. The paper reports those costs separately from steady-state speed and distinguishes already prepared blocks from CPU assembly.

The paper's width table compares direct execution with both BSR8 and the fastest tested library method per case. Compact CSR remains the appropriate control for scattered scalar nonzeros, and dense SGEMM is included for high overall density. These comparisons help identify whether your input fits the direct kernel's strengths.

```cpp
vbsr::Matrix matrix(host);
vbsr::Plan plan(matrix, {32, vbsr::Kernel::Auto});
plan.execute(device_B, device_C, stream);
```

Here `host` contains your validated packed matrix, and the dense buffers use column-major layout. The matrix and buffers must remain alive until queued work completes.

## Comparisons and evidence

The final evaluation includes:

- 168 synthetic configurations, each repeated in three fresh processes.
- Explicit cuSPARSE CSR algorithms with persistent setup and preprocessing, cached grouped cuBLAS, and changing-address controls with stream-ordered device pointer generation.
- Padding-free BSR8 subdivision on every variable-block input, plus BSR32 on uniform inputs.
- Three published scalar sparse matrices with artificial partitions and compact-CSR controls.
- Persistent dense SGEMM on published and native inputs, including zero-fill expansion in startup costs.
- Twelve generated covariance near-field matrices with geometry-derived partitions, three geometry seeds, all four RHS widths, and three processes per instance. This benchmarks a dense-block application component, not a full hierarchical solver.
- Matched kernel controls at two sizes and degrees, with three matrix seeds and three processes per seed. RHS 8 uses a complete mapping/accumulator/CTA factorial design. Wider panels have matched accumulation, staging, and buffering controls.
- Native-input startup times, explicit device-array storage, and serial repeated-product cost models.

The tables report each panel width and the full comparison grid. Numerical results come directly from validated raw records, with historical tuning results archived separately. Performance measurements cover one GPU.

## Build and correctness

The Windows reference workflow requires CUDA 13.4, Visual Studio 2022 Build Tools, and CMake 3.25 or newer. Host translation units use the compiler's C++26 draft mode, and CUDA translation units use C++20.

```powershell
scripts/build.ps1
```

The full-output tests poison output before each comparison, reject a deliberately empty operation, check CUDA errors, cover all 64 block shapes at every RHS width, and include mixed and completely empty rows. They check BSR8 conversion, explicit CSR algorithms, queued input/output address changes, and non-default streams.

## Reproduce the paper

Python preparation dependencies are NumPy, SciPy, Requests, and Matplotlib. PDF building additionally requires pdfLaTeX and pdftotext.

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
