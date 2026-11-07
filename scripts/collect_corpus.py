"""Download public chart metadata for local research, without bulk audio downloads.

Uses the upstream's research dataset. pyarrow is a development-only dependency.
The output directory must stay outside release assets: charts are not redistributed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, validate

DATASET = "project-riz/osu-beatmaps"
REVISION = "b82117a5de13b7df5290482aed7cffc49cb76e6d"


def request_bytes(url, start=None, end=None):
    headers = {"User-Agent": "ManiaLab-corpus-research/0.1"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                if start is not None:
                    expected = f"bytes {start}-"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                        raise ValueError("The server did not honour the requested byte range")
                return response.read()
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


class RangeFile(io.RawIOBase):
    """Seekable HTTP reader; Arrow can fetch JSON columns and skip audio bytes."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.bytes_fetched, self.requests = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if target < 0:
            raise ValueError("Negative seek")
        self.position = target
        return target

    def read(self, length=-1):
        if length < 0:
            length = self.size - self.position
        length = min(length, self.size - self.position)
        if length <= 0:
            return b""
        data = request_bytes(self.url, self.position, self.position + length - 1)
        if len(data) != length:
            raise IOError("Truncated HTTP range")
        self.position += length
        self.bytes_fetched += length
        self.requests += 1
        return data


def song_identity(metadata):
    # Title grouping also keeps alternate audio encodings and difficulties
    # together. Over-grouping remixes is safer than leaking a song into test.
    clean = lambda value: re.sub(r"[^\w]", "", str(value).casefold())
    return clean(metadata.get("artist", "")) + ":" + clean(metadata.get("title", ""))


def save_sample(sample, key, out, source):
    saved = []
    if isinstance(sample, str):
        sample = json.loads(sample)
    for metadata in sample.get("beatmaps", []):
        if int(metadata.get("mode", -1)) != 3 or float(metadata.get("diff_size", 0)) not in range(6, 11):
            continue
        content = metadata.get("content", "")
        try:
            parsed = parse_osu(content)
            if parsed["keys"] not in range(6, 11):
                continue
            report = validate(parsed["notes"], parsed["keys"])
        except (ValueError, IndexError):
            continue
        bid = int(metadata["beatmap_id"])
        (out / "maps").mkdir(exist_ok=True)
        (out / "maps" / f"{bid}.osu").write_text(content, encoding="utf-8", newline="\n")
        style = " ".join(str(metadata.get(k, "")) for k in ("version", "tags", "title")).casefold()
        # This is a conservative exclusion flag, not an automatic style label.
        suspect_edit = bool(re.search(r"\binverse\b|\binvert\b|\bconverted?\b|\bfull ln\b|\bfln\b|\btrainer\b|\b\d[.,]\d+x\b", style))
        saved.append({k: v for k, v in metadata.items() if k != "content"} | dict(
            audio_hash=sample.get("audio_hash"), audio_length=sample.get("audio_length"),
            song_key=song_identity(metadata), sample_key=key, source=source,
            keys=parsed["keys"], sha256=hashlib.sha256(content.encode()).hexdigest(),
            report=report, suspected_conversion=suspect_edit,
            eligible=report["valid"] and len(parsed["notes"]) >= 100 and not suspect_edit))
    return saved


def collect_parquet(out, limit=10):
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    listing = json.loads(request_bytes(f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}"))
    (out / "source.json").write_text(json.dumps(dict(dataset=DATASET, revision=REVISION,
        retrieved_at=time.time(), partial=listing.get("partial"), listing=listing), indent=2), encoding="utf-8")
    files = [f for f in listing["parquet_files"] if f["config"] == "compressed"][:limit]
    for entry in files:
        index = entry["filename"].split(".")[0]
        manifest = out / f"manifest-{index}.jsonl"
        if manifest.exists():
            print(f"Cached {index}", flush=True)
            continue
        remote = RangeFile(entry["url"], entry["size"])
        reader = pq.ParquetFile(remote)
        rows = reader.read(columns=["json", "__key__"], use_threads=False).to_pylist()
        saved = []
        for row in rows:
            saved.extend(save_sample(row["json"], row["__key__"], out, entry["url"]))
        manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in saved), encoding="utf-8")
        print(json.dumps(dict(shard=index, songs_scanned=len(rows), maps=len(saved),
            keys=dict(Counter(r["keys"] for r in saved)), bytes_fetched=remote.bytes_fetched,
            requests=remote.requests)), flush=True)
    records = [json.loads(line) for p in sorted(out.glob("manifest-*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines() if line]
    print(json.dumps(dict(total=len(records), eligible=sum(r["eligible"] for r in records),
        keys=dict(Counter(r["keys"] for r in records)), songs=len({r["song_key"] for r in records}))), flush=True)


class BufferedRangeFile(RangeFile):
    """Coalesce nearby tar headers; seek past large audio members untouched."""
    def __init__(self, url, size):
        super().__init__(url, size)
        self.buffer_start, self.buffer = 0, b""

    def read(self, length=-1):
        if length < 0:
            length = self.size - self.position
        length = min(length, self.size - self.position)
        if length <= 0:
            return b""
        relative = self.position - self.buffer_start
        if relative >= 0 and relative + length <= len(self.buffer):
            self.position += length
            return self.buffer[relative:relative + length]
        first = self.position
        amount = min(self.size - first, max(16384, length))
        self.buffer = request_bytes(self.url, first, first + amount - 1)
        self.buffer_start = first
        self.bytes_fetched += len(self.buffer)
        self.requests += 1
        self.position += length
        return self.buffer[:length]


def collect_tar(out, shard):
    """Read a pinned original archive's JSON members, skipping audio payloads."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"manifest-tar-{shard:06d}.jsonl"
    if manifest.is_file():
        print(f"Cached tar {shard}", flush=True)
        return
    listing = json.loads(request_bytes(f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/compressed?recursive=false&expand=false&limit=100"))
    entry = next(row for row in listing if row["path"] == f"compressed/data-{shard:06d}.tar")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
    remote = BufferedRangeFile(url, entry["size"])
    records, scanned = [], 0
    with tarfile.open(fileobj=remote, mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if member.size > 20 * 1024 * 1024:
                raise ValueError("Unexpected oversized chart metadata")
            with archive.extractfile(member) as stream:
                sample = json.load(stream)
            records.extend(save_sample(sample, member.name.removesuffix(".json"), out, url))
            scanned += 1
            if scanned % 100 == 0:
                print(json.dumps(dict(tar=shard, scanned=scanned, maps=len(records), bytes_fetched=remote.bytes_fetched)), flush=True)
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    print(json.dumps(dict(tar=shard, songs_scanned=scanned, maps=len(records),
        keys=dict(Counter(r["keys"] for r in records)), bytes_fetched=remote.bytes_fetched)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("test-data/corpus"))
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--tar", type=int, nargs="*", help="Pinned full dataset archive indices; skip their audio payloads")
    args = parser.parse_args()
    if args.tar is not None:
        for index in args.tar:
            collect_tar(args.output, index)
    else:
        collect_parquet(args.output, args.shards)
