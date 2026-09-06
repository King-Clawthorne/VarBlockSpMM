"""Store an exact input ZIP in Git-hostable parts, without changing its bytes."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

PART_BYTES = 64 * 1024 * 1024


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def split_inputs(directory):
    """Run after collection. Keep the original ZIP locally for fast analysis."""
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = directory/'inputs.zip'
    if not manifest.get('complete') or digest(source) != manifest['input_archive_sha256']:
        raise ValueError('Only a complete campaign with the original input ZIP can be split')
    parts = {}
    with source.open('rb') as archive:
        index = 0
        while chunk := archive.read(PART_BYTES):
            name = f'inputs.zip.{index:03d}'
            path = directory/name
            path.write_bytes(chunk)
            parts[name] = hashlib.sha256(chunk).hexdigest()
            index += 1
    manifest['input_archive_parts'] = parts
    manifest['input_archive_size'] = source.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return parts


@contextmanager
def input_zip(directory, manifest):
    """Accept the original ZIP or checked, ordered parts from a fresh checkout."""
    source = directory/'inputs.zip'
    parts = manifest.get('input_archive_parts')
    if parts is not None:
        size = manifest.get('input_archive_size', 0)
        if not isinstance(size, int) or size <= 0:
            raise ValueError('Input archive size is missing or invalid')
        expected = [f'inputs.zip.{i:03d}' for i in range((size + PART_BYTES - 1)//PART_BYTES)]
        if not parts or sorted(parts) != expected:
            raise ValueError('Input archive part coverage mismatch')
        for name, expected_digest in parts.items():
            path = directory/name
            if not path.is_file() or digest(path) != expected_digest:
                raise ValueError('Input archive part checksum mismatch')
        if sum((directory/name).stat().st_size for name in parts) != size:
            raise ValueError('Input archive part extent mismatch')
    if source.is_file():
        yield source
    elif parts:
        with tempfile.TemporaryDirectory(prefix='vbsr-input-archive-') as temporary:
            combined = Path(temporary)/'inputs.zip'
            with combined.open('wb') as output:
                for name in sorted(parts):
                    with (directory/name).open('rb') as part:
                        shutil.copyfileobj(part, output)
            yield combined
    else:
        raise ValueError('Input archive checksum mismatch: no ZIP or parts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(split_inputs(parser.parse_args().directory), indent=2))
