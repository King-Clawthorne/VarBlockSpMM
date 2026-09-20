"""Collect a serial GPU audit with randomized case and method order."""

import argparse
import hashlib
import itertools
from pathlib import Path
import random
import time
from typing import NamedTuple
from build_verified import verified_build

from benchmark_runs import (ROOT, archive_runs, read_runs, run_process, save_manifest,
                            source_hashes, source_paths, validate_run)

ORDER_SEED = 20260905
PANEL_WIDTHS = (8, 16, 32, 64)
SYNTHETIC_METHODS = ["direct", "csr_default", "csr_alg1_pre", "csr_alg2", "csr_alg3_pre",
                     "grouped_cached", "grouped_changing", "direct_changing", "bsr8"]

class SyntheticCase(NamedTuple):
    rows: int
    degree: int
    rhs: int
    distribution: str
    locality: str
    seed: int = 1
    irregular: int = 0

    def stem(self, process: int) -> str:
        return (f"{self.rows}_{self.degree}_{self.rhs}_{self.distribution}_{self.locality}"
                f"_s{self.seed}_i{self.irregular}_p{process}")

    def methods(self) -> list[str]:
        return SYNTHETIC_METHODS + (["bsr32"] if self.distribution == "uniform" else [])

def build_cases() -> list[SyntheticCase]:
    cases = [SyntheticCase(4096, degree, rhs, distribution, locality)
             for distribution, locality, degree, rhs in itertools.product(
                 ("uniform", "low", "high", "bimodal"), ("local", "random"), (2, 4, 8, 16), PANEL_WIDTHS)]
    for rows, distribution, rhs, irregular in itertools.product(
            (256, 1024, 4096), ("high", "bimodal"), PANEL_WIDTHS, (0, 1)):
        case = SyntheticCase(rows, 16, rhs, distribution, "random", irregular=irregular)
        if case not in cases:
            cases.append(case)
    return cases

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/revision")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    if args.processes < 1 or args.reps < 2:
        parser.error("at least one process and two repetitions are required")
    return args

def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    executable, receipt = verified_build('vbsr_audit')
    cases = build_cases()
    rng = random.Random(ORDER_SEED)
    jobs = [(case, process) for case in cases for process in range(args.processes)]
    rng.shuffle(jobs)
    sources = source_paths("run_revision.py")
    manifest = dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    cases=len(cases), processes=args.processes, reps=args.reps, warmup=5,
                    randomization_seed=ORDER_SEED, matrix_seed=1, jobs=jobs,
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    sources=source_hashes(sources), build_receipt=receipt)
    save_manifest(args.output, manifest, sources)
    existing = {record.stem: record for record in read_runs(args.output)}
    start = time.monotonic()
    for index, (case, process) in enumerate(jobs):
        stem = case.stem(process)
        order_seed = rng.randrange(2**31)
        command = [str(executable), str(case.rows), str(case.degree), str(case.rhs),
                   case.distribution, case.locality, str(case.seed), str(order_seed),
                   str(case.irregular), str(args.reps)]
        if stem in existing:
            validate_run(existing[stem], case.methods(), args.reps)
            if existing[stem].metadata["command"] != command:
                raise ValueError(f"Stored command differs for {stem}")
            continue
        run_process(args.output, stem, command, case.methods(), args.reps, "environment")
        if index % 10 == 0 or index == len(jobs) - 1:
            print(f"{index + 1}/{len(jobs)} processes, {time.monotonic() - start:.0f}s elapsed", flush=True)
    if not (args.output / "runs.zip").exists():
        archive_runs(args.output)
    print("Complete", flush=True)

if __name__ == "__main__":
    main()
