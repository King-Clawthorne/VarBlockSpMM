# Benchmark evidence

The current paper uses these retained inputs and measurements:

| Path | Purpose |
| --- | --- |
| `product-evaluation/` | Final synthetic, published-matrix, native-component, and matched-control campaigns, with build receipts, source snapshots, raw trials, and validation records |
| `robustness/` | Full core grid on seeds 2, 3, and 5 with individual and eight-product queued timing, retaining 768 processes in checksummed archives |
| `relevance/` | Bounded MAGMA, transport, controlled-factor, and API-update follow-up with 221 processes, source and raw archives, and upstream MAGMA copyright |
| `narrow-tuning/`, `narrow-tuning-2/` | Development sweeps on seed 4, kept separate from the final evaluation |
| `native/*.json`, `native/*-geometry.npz` | Native input descriptions and geometry used to reconstruct the large binary matrices |
| `application/` | Published-matrix inputs and earlier tracked results |
| `kernel-rhs8-final-20260906/` | Historical measurements used by the paper appendix |

Other already tracked data records earlier development. New experiment directories and generated native binaries are ignored by default. To retain a new campaign, add a specific exception to `.gitignore` after validating its evidence.

Superseded, previously untracked campaigns were moved to `build/local-evidence-archive/data/` on the development machine. They are preserved locally and are not needed to build the current paper.

Run `powershell -File scripts/build_paper.ps1` from the repository root to validate the retained evidence and rebuild the paper. The script reconstructs missing native binaries using `scripts/prepare_native.py`.
