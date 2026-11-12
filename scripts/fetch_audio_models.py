"""Fetch pinned official audio weights for a source build; the release includes them."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(output):
    output.mkdir(parents=True, exist_ok=True)
    models = json.loads((ROOT/'assets/deep-models.json').read_text(encoding='utf-8'))['models']
    for entry in models:
        target = output/entry['filename']
        if target.is_file() and digest(target) == entry['sha256']:
            print(f"Verified: {entry['name']}", flush=True)
            continue
        temporary = target.with_suffix(target.suffix+'.pending')
        sources = [entry['url'], *entry.get('mirror_urls', [])]
        failures = []
        for index, url in enumerate(sources, start=1):
            print(
                f"Downloading: {entry['name']} ({entry['bytes'] / 1048576:.1f} MiB), "
                f"source {index}/{len(sources)}",
                flush=True,
            )
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    with temporary.open('wb') as stream:
                        while block := response.read(1024*1024):
                            stream.write(block)
                if temporary.stat().st_size != entry['bytes']:
                    raise ValueError(f"size mismatch: expected {entry['bytes']} bytes")
                if digest(temporary) != entry['sha256']:
                    raise ValueError("SHA-256 mismatch")
                temporary.replace(target)
                print(f"Verified: {entry['name']}", flush=True)
                break
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                failures.append(f"{type(exc).__name__}: {exc}")
                print(f"Download source failed: {failures[-1]}", flush=True)
        else:
            raise RuntimeError(
                f"Unable to fetch verified model {entry['name']}: {'; '.join(failures)}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/deep')
    fetch(parser.parse_args().output)
