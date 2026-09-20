"""Storage and validation shared by the benchmark runners and analysis."""

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ["method", "position", "iteration", "gpu_ms", "host_ms"]


@dataclass(frozen=True)
class RunRecord:
    stem: str
    csv_text: str
    metadata: dict

    def rows(self) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(self.csv_text))
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"Unexpected timing columns in {self.stem}")
        return list(reader)


def read_runs(directory: Path):
    """Read either an in-progress directory or a completed immutable archive."""
    seen = set()
    archive_path = directory / "runs.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            checksums = json.loads(archive.read("checksums.json"))
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Duplicate archive entries")
            if set(archive.namelist()) != set(checksums) | {"checksums.json"}:
                raise ValueError("Archive checksum coverage mismatch")
            for name, digest in checksums.items():
                if Path(name).name != name or Path(name).suffix not in ('.csv', '.json'):
                    raise ValueError('Unexpected run archive member: ' + name)
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f"Archive checksum mismatch: {name}")
            csv_stems = {Path(n).stem for n in checksums if n.endswith('.csv')}
            json_stems = {Path(n).stem for n in checksums if n.endswith('.json')}
            if csv_stems != json_stems:
                raise ValueError('Run metadata and CSV coverage differ')
            for name in sorted(checksums):
                if not name.endswith(".csv"):
                    continue
                stem = Path(name).stem
                seen.add(stem)
                yield RunRecord(stem, archive.read(name).decode("utf-8"),
                                json.loads(archive.read(stem + ".json")))
    for path in sorted(directory.glob("*.csv")):
        if path.stem in seen:
            raise ValueError(f"Both loose and archived results exist for {path.stem}")
        yield RunRecord(path.stem, path.read_text(encoding="utf-8"),
                        json.loads(path.with_suffix(".json").read_text(encoding="utf-8")))


def archive_runs(directory: Path) -> None:
    """Verify a complete archive before removing the individual source records."""
    directory = directory.resolve()
    destination = directory / "runs.zip"
    if destination.exists():
        raise ValueError(f"Archive already exists: {destination}")
    files = []
    for path in sorted(directory.glob("*.csv")):
        files.extend((path, path.with_suffix(".json")))
    if not files:
        raise ValueError("No runs to archive")
    contents = {}
    for path in files:
        if path.resolve().parent != directory or not path.is_file():
            raise ValueError(f"Unexpected run path: {path}")
        contents[path.name] = path.read_bytes()
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr("checksums.json", json.dumps(checksums, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for name, data in contents.items():
            if archive.read(name) != data:
                raise ValueError(f"Archive byte verification failed: {name}")
    temporary.replace(destination)
    for path in files:
        # Refuse to remove a file modified while the archive was being written.
        if path.read_bytes() != contents[path.name]:
            raise ValueError(f"Run changed during archiving: {path}")
        path.unlink()


def validate_run(record: RunRecord, methods: list[str], repetitions: int) -> None:
    if record.metadata.get("returncode") != 0:
        raise ValueError(f"Unsuccessful process: {record.stem}")
    rows = record.rows()
    if len(rows) != len(methods) * repetitions:
        raise ValueError(f"Incomplete timings: {record.stem}")
    if {row["method"] for row in rows} != set(methods):
        raise ValueError(f"Unexpected methods: {record.stem}")
    positions = set()
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if sorted(int(row["iteration"]) for row in selected) != list(range(repetitions)):
            raise ValueError(f"Missing or duplicate iterations: {record.stem}, {method}")
        position = {int(row["position"]) for row in selected}
        if len(position) != 1 or positions & position:
            raise ValueError(f"Inconsistent method positions: {record.stem}")
        positions.update(position)
    if positions != set(range(len(methods))):
        raise ValueError(f"Invalid method position range: {record.stem}")
    for row in rows:
        for field in ("gpu_ms", "host_ms"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {field}: {record.stem}")


def run_process(output: Path, stem: str, command: list[str], methods: list[str],
                repetitions: int, metadata_field: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    metadata = dict(command=command, returncode=result.returncode,
                    executable_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                    recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    metadata[metadata_field] = result.stderr
    (output / (stem + ".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{stem}: {result.stderr}")
    validate_run(RunRecord(stem, result.stdout, metadata), methods, repetitions)
    temporary = output / (stem + ".csv.tmp")
    temporary.write_text(result.stdout, encoding="utf-8")
    temporary.replace(output / (stem + ".csv"))


def source_paths(runner: str) -> list[Path]:
    """Include shared headers and helpers as well as executable entry points."""
    patterns = ("src/**/*.cu", "src/**/*.cuh", "src/**/*.cpp", "include/**/*.hpp",
                "include/**/*.cuh", "bench/*.cpp", "bench/*.hpp", "bench/*.cu", "bench/*.cuh",
                "tests/*.cpp", "scripts/*.py", "scripts/*.ps1", "cmake/*.cmake.in", "cmake/*.cmake", "pyproject.toml", "uv.lock")
    paths = {path for pattern in patterns for path in ROOT.glob(pattern)}
    paths.update((ROOT / "CMakeLists.txt", ROOT / "scripts" / runner,
                  ROOT / "scripts/benchmark_runs.py", ROOT / "scripts/prepare_application.py"))
    return sorted(paths)


def source_hashes(paths: list[Path]) -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def save_manifest(output: Path, manifest: dict, sources: list[Path]) -> None:
    # Normalize tuples and named tuples to their on-disk JSON representation.
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        expected = {key: value for key, value in manifest.items() if key != "started_utc"}
        actual = {key: value for key, value in previous.items() if key != "started_utc"}
        if actual != expected:
            raise ValueError("Existing results have different provenance. Use a fresh output directory.")
        return
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(ROOT).as_posix())
