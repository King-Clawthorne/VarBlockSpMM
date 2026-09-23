# VarBlockSpMM

Reproduction package for the paper [*VarBlockSpMM*](research/paper.pdf) and its [supplement](research/supplement.pdf): GPU sparse-times-dense multiplication (`C = A B`) on matrices made of dense blocks of different sizes.

This repository holds the kernel, the benchmark drivers, the measured evidence and the scripts that turn it into every number, table and figure in the paper.

## Requirements

- Windows with an NVIDIA GPU (measurements used an RTX 5060 Ti)
- CUDA 12.6 or newer (measurements used 13.4), CMake 3.25 or newer, Visual Studio 2022
- Python 3.13 with the pinned environment: `uv sync`
- `pdflatex` and `pdftotext` for the paper build

## Rebuild the paper from the checked-in data

No GPU needed. This validates the archived campaigns in [data/](data/), regenerates the macros in [research/generated/](research/generated/), and builds both PDFs:

```powershell
powershell -File scripts/build_paper.ps1
```

## Regenerate all data on your GPU

This takes hours of GPU time. Run nothing else on the GPU while it runs.

```powershell
powershell -File scripts/build.ps1
python scripts/run_final_validation.py --output-root build/rerun
powershell -File scripts/build_paper.ps1 -ResultsRoot build/rerun
```

`run_final_validation.py` builds a verified binary, runs the sanitizers, and then runs every campaign the paper reads: synthetic, published matrices, native components, ablation, robustness, the MAGMA and transport comparisons, and the specialized DG comparison. Each campaign writes a manifest, a source snapshot and the raw trials under the output root. `build_paper.ps1 -ResultsRoot` then builds the paper from your measurements instead of the archived ones.

## Layout

| Path | Contents |
| --- | --- |
| [include/](include/), [src/](src/) | The VarBlockSpMM library and cuSPARSE baselines |
| [bench/](bench/) | Benchmark drivers for each campaign |
| [scripts/](scripts/) | Campaign runners (`run_*`, `prepare_*`) and analysis and exporters (`analyze_*`, `export_*`) |
| [data/](data/) | Archived evidence behind the published numbers; see [data/README.md](data/README.md) |
| [research/](research/) | Paper sources, generated macros and figures, and the built PDFs |
| [tests/](tests/) | Kernel correctness and numerics (CTest), plus evidence validation (`python -m unittest discover tests`) |
