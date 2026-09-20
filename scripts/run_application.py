"""Run published-matrix checks serially after the synthetic GPU sweep."""

import argparse
import hashlib
import itertools
from pathlib import Path
import random
from build_verified import verified_build

from benchmark_runs import (ROOT, archive_runs, read_runs, run_process, save_manifest,
                            source_hashes, source_paths, validate_run)

ORDER_SEED = 20260906
REPETITIONS = 20
METHODS = ["direct", "compact_alg1_pre", "compact_alg2", "compact_alg3_pre", "grouped_cached", "bsr8", "dense"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/application/results")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    executable, receipt = verified_build('vbsr_application_audit')
    rng = random.Random(ORDER_SEED)
    jobs = list(itertools.product((13, 14, 15), (8, 16, 32, 64), range(3)))
    rng.shuffle(jobs)
    sources = source_paths("run_application.py")
    manifest = dict(reps=REPETITIONS, warmup=5, processes=3, jobs=jobs, randomization_seed=ORDER_SEED,
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    inputs={f'bcsstk{n}{suffix}': hashlib.sha256((ROOT / 'data/application' / f'bcsstk{n}{suffix}').read_bytes()).hexdigest()
                            for n in (13, 14, 15) for suffix in ('.bin', '.json')},
                    sources=source_hashes(sources), build_receipt=receipt)
    save_manifest(output, manifest, sources)
    existing = {record.stem: record for record in read_runs(output)}
    for number, rhs, process in jobs:
        stem = f"bcsstk{number}_rhs{rhs}_p{process}"
        order_seed = rng.randrange(2**31)
        command = [str(executable), str(ROOT / f"data/application/bcsstk{number}.bin"),
                   str(rhs), str(order_seed), str(REPETITIONS)]
        if stem in existing:
            validate_run(existing[stem], METHODS, REPETITIONS)
            if existing[stem].metadata["command"] != command:
                raise ValueError(f"Stored command differs for {stem}")
            continue
        run_process(output, stem, command, METHODS, REPETITIONS, "stderr")
        print(stem, "passed", flush=True)
    if not (output / "runs.zip").exists():
        archive_runs(output)


if __name__ == "__main__":
    main()
