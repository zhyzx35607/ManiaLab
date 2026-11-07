"""Audit a local .osu corpus before training; no maps or audio are uploaded."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[1]))
from manialab.domain import parse_osu, validate
from manialab.storage import safe_path


def audit(root):
    seen=set()
    for path in sorted(Path(root).rglob('*.osu')):
        try:
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            parsed=parse_osu(raw.decode('utf-8-sig'))
            if parsed['keys'] not in range(6,11):continue
            audio=safe_path(path.parent,parsed['general'].get('AudioFilename','missing'))
            # Exact shared audio is grouped into a single split to avoid song leakage.
            song_hash=hashlib.sha256(audio.read_bytes()).hexdigest() if audio.is_file() else None
            report=validate(parsed['notes'],parsed['keys'])
            durations=sorted(n.end-n.start for n in parsed['notes'] if n.hold)
            split_value=int((song_hash or digest)[:8],16)%100
            yield dict(path=str(path),sha256=digest,audio_sha256=song_hash,duplicate=digest in seen,
                       split=('test' if split_value<10 else 'validation' if split_value<20 else 'train') if song_hash else 'unassigned',
                       keys=parsed['keys'],metadata=parsed['metadata'],report=report,
                       median_ln_ms=durations[len(durations)//2] if durations else 0,
                       style_review_required=True,eligible=report['valid'] and digest not in seen and song_hash is not None)
            seen.add(digest)
        except Exception as error:
            yield dict(path=str(path),error=str(error),eligible=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='6–10K 本地训练语料审计；不启动训练')
    parser.add_argument('folder');parser.add_argument('--output',default='corpus-audit.jsonl');args=parser.parse_args()
    rows=list(audit(args.folder));Path(args.output).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
    print(json.dumps({'files':len(rows),'eligible':sum(r['eligible'] for r in rows),'keycounts':dict(Counter(r.get('keys') for r in rows))},ensure_ascii=False))
