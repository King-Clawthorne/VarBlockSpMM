"""Collect native near-field and matched kernel controls, serially."""
import argparse
import hashlib
import itertools
import random
from benchmark_runs import ROOT, archive_runs, read_runs, run_process, save_manifest, source_paths, source_hashes, validate_run
from build_verified import verified_build


def jobs_for(kind):
    if kind == 'native':
        return [(f'covariance_{n}_{g}_s{s}', rhs, process)
                for n, g, s, rhs, process in itertools.product(
                    (2048, 8192), ('uniform', 'clustered'), (1, 2, 3), (8, 16, 32, 64), range(3))]
    return list(itertools.product((256, 1024), (2, 16), (8, 16, 32, 64), ('high', 'bimodal'), (1, 2, 3), range(3)))


def specification(kind, job, executable, order_seed, ablation_revision=2):
    if kind == 'native':
        name, rhs, process = job
        return (f'{name}_rhs{rhs}_p{process}',
                [str(executable), str(ROOT / 'data/native' / (name + '.bin')), str(rhs), str(order_seed), '20'],
                ['direct', 'compact_alg1_pre', 'compact_alg2', 'compact_alg3_pre', 'grouped_cached', 'bsr8', 'dense'])
    n, degree, rhs, dist, seed, process = job
    methods = ['dispatch'] + [f'v{i}' for i in range(8 if rhs == 8 else 5)]
    if ablation_revision >= 2 and rhs in (8, 16):
        methods += ['v8', 'v9', 'v200'] if rhs == 8 else ['v5', 'v6', 'v200']
    return (f'{n}_{degree}_{rhs}_{dist}_s{seed}_p{process}',
            [str(executable), str(n), str(degree), str(rhs), dist, str(seed), str(order_seed), '20'],
            methods)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('kind', choices=('native', 'ablation'))
    parser.add_argument('--output', type=__import__('pathlib').Path, required=True)
    args = parser.parse_args()
    executable, receipt = verified_build('vbsr_application_audit' if args.kind == 'native' else 'vbsr_ablation')
    jobs = jobs_for(args.kind)
    rng = random.Random(20260907)
    rng.shuffle(jobs)
    sources = source_paths('run_supplement.py')
    manifest = dict(kind=args.kind, jobs=jobs, reps=20, warmup=5, randomization_seed=20260907, ablation_revision=2,
                    sources=source_hashes(sources), build_receipt=receipt,
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    inputs={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ROOT / 'data/native').glob('*') if p.is_file()} if args.kind == 'native' else {})
    args.output.mkdir(parents=True, exist_ok=True)
    save_manifest(args.output, manifest, sources)
    existing = {r.stem: r for r in read_runs(args.output)}
    for index, job in enumerate(jobs):
        stem, command, methods = specification(args.kind, job, executable, rng.randrange(2**31))
        if stem in existing:
            validate_run(existing[stem], methods, 20)
            if existing[stem].metadata['command'] != command:
                raise ValueError('Resume command mismatch')
            continue
        run_process(args.output, stem, command, methods, 20, 'environment')
        if index % 12 == 0:
            print(f'{index + 1}/{len(jobs)} {stem}', flush=True)
    if not (args.output / 'runs.zip').exists():
        archive_runs(args.output)


if __name__ == '__main__':
    main()
