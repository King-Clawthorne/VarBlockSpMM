# Benchmark evidence

The current paper uses these retained inputs and measurements:

| Path | Purpose |
| --- | --- |
| `product-evaluation/` | Final synthetic, published-matrix, native-component, and matched-control campaigns, with build receipts, source snapshots, raw trials, and validation records |
| `robustness/` | Full core grid on seeds 2, 3, and 5 with individual and eight-product queued timing, retaining 768 processes in checksummed archives |
| `revision/` | Seed-1 core campaign, before the MAGMA comparator was added |
| `relevance-v2/` | MAGMA comparison across three seeds, three processes per seed, and two queue modes, plus characteristic DG transport, exact input payloads, source and raw archives |
| `dg/` | Specialized DG transport comparison on the `relevance-v2/` transport inputs |
| `native/*.json`, `native/*-geometry.npz` | Native input descriptions and geometry used to reconstruct the large binary matrices |
| `application/*.json` | Descriptions of the published matrices. `scripts/prepare_application.py` downloads bcsstk13 to 15 from SuiteSparse and rebuilds the binaries |

New experiment directories and generated native binaries are ignored by default. To retain a new campaign, add a specific exception to `.gitignore` after validating its evidence.

Run `powershell -File scripts/build_paper.ps1` from the repository root to validate the retained evidence and rebuild the paper. The script rebuilds missing native binaries with `scripts/prepare_native.py`, and downloads and converts missing published matrices with `scripts/prepare_application.py`.
