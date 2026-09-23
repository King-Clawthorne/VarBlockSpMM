# Reproducing the paper

The complete GPU campaign runner currently requires Windows. The Python analysis tools can read the retained archives on other operating systems.

The paper environment uses Python 3.13 and uv. Install its pinned dependencies with `uv sync --locked`, then activate `.venv` before running the commands below (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux). The `paper` dependency group captures NumPy, SciPy, Requests, Matplotlib, and their required dependencies from the validated environment. PDF building additionally requires pdfLaTeX and pdftotext.

```powershell
$env:CUDAToolkit_ROOT = $env:CUDA_PATH_V13_4
$env:CUDA_PATH = $env:CUDAToolkit_ROOT
python scripts/prepare_application.py
python scripts/prepare_native.py
python scripts/run_final_validation.py --output-root data/rerun
powershell -File scripts/build_paper.ps1 -ResultsRoot data/rerun
```

The final runner performs a verified build, memcheck, and filtered racecheck, then runs the five earlier GPU campaigns serially into the specified fresh root. It also fetches the pinned MAGMA sources, prepares all six transport inputs and the convergence study, builds and runs the comparison correctness tests through CTest, including two untimed large full-output checks, and collects the complete MAGMA/transport campaign under that root's `relevance/` directory, followed by the specialized transport comparison under `dg/`. This supplies every campaign required by the following paper-build command, and `tests/test_workflow_coverage.py` checks that the runner still produces every directory the paper build reads. It can take substantial time. Benchmark runners verify a build receipt tying compiled source hashes to the actual executable bytes, and snapshot the source used for each campaign. Existing records are resumed only with matching provenance and commands. Run `scripts/build_paper.ps1` without `-ResultsRoot` to rebuild the checked-in paper from its canonical campaigns instead.

Use individual runners and fresh output directories for independent repetitions:

```powershell
python scripts/run_revision.py --output data/revision-rerun
python scripts/run_application.py --output data/application/rerun
python scripts/run_supplement.py native --output data/native-rerun
python scripts/run_supplement.py ablation --output data/ablation-rerun
python scripts/run_dg.py --output data/dg-rerun
python scripts/analyze_revision.py --revision data/revision-rerun --application data/application/rerun --output build/rerun-tables
python scripts/analyze_supplement.py --native data/native-rerun --ablation data/ablation-rerun --output build/rerun-tables
python -m unittest discover -s tests -p 'test_*.py'
```

Native binary arrays are large and excluded from Git. Their generator, geometry, partition metadata, assembly observations, and checksums are retained. `prepare_native.py` reconstructs identical binaries while preserving the archived CPU assembly observations. `--output` with a fresh directory remeasures preparation.

The specialized transport comparison is a separate campaign in `data/dg/`, produced by `scripts/run_dg.py` and exported by `scripts/export_dg.py`. It reuses the transport inputs prepared for the main campaign, verified by content hash, and compares direct execution against two application-specific plans that cache the eight distinct block operators: a batched-GEMM plan that direct execution beats, and a fused kernel whose current speedup range is reported in [results](results.md). See [docs/relevance.md](relevance.md).

The replacement MAGMA/transport archive is `data/relevance-v2/`, including exact transport payloads stored as `inputs.zip.000` through `inputs.zip.003`. The analyzer checks and reconstructs these parts automatically. The earlier `data/relevance/` archive remains historical and does not generate the current headline. The original canonical campaigns are under `data/product-evaluation/`, in `synthetic/`, `published/`, `native/`, and `ablation/`. Sanitizer records are in its `validation/` subdirectory. The analyzers reject checksum, build receipt, design, command, process, method, and timing inconsistencies before emitting tables.

## Additional seeds and queued throughput

```powershell
python scripts/run_robustness.py --output data/robustness-rerun
python scripts/export_robustness.py --input data/robustness-rerun
```

The default paper build validates the retained `data/robustness/` archive. A full `run_final_validation.py` rerun also collects this follow-up under its output root. The original three-process, seed-1 campaign remains separately identifiable. Queued measurements report time per product across batches of eight with one synchronization per batch. They use the same mathematical operation and retained library sources.
