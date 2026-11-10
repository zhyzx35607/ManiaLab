"""Cache central-song Mapper encoder evidence without using chart events."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.mapper_encoder_inputs import load_encoder, audit_geometry, sha, SAMPLE_RATE, WINDOW_SAMPLES, FRAME_SAMPLES

STEP = 131072
EDGE = 65536


def window_positions(length, start, end):
    """Central retention partitions physical time without overlapping frames."""
    result = []
    first = max(0, int((start*SAMPLE_RATE/1000-EDGE)//STEP))
    for number in range(first, int(end*SAMPLE_RATE/1000//STEP)+1):
        base = number*STEP
        positions = base+np.arange(1024)*FRAME_SAMPLES
        lower = base+EDGE if number else 0
        upper = base+EDGE+STEP
        mask = ((positions >= lower) & (positions < upper) & (positions < length)
                & (positions >= start*SAMPLE_RATE/1000) & (positions <= end*SAMPLE_RATE/1000))
        if mask.any(): result.append((base, positions[mask], mask))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--pairs', type=Path, default=Path('test-data/corpus-expanded/paired-stem-release-dataset-v1/manifest.json'))
    cli.add_argument('--prepared', type=Path, default=Path('test-data/mapper-training/mmrs'))
    cli.add_argument('--source', type=Path, default=Path('test-data/mapperatorinator/source'))
    cli.add_argument('--checkpoint', type=Path, default=Path('test-data/mapperatorinator/model/gamemode=3'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--threads', type=int, default=2)
    args = cli.parse_args()
    if (args.output/'manifest.json').exists(): raise ValueError('Preserve completed extraction')
    import torch
    encoder, loader, identity = load_encoder(args.source, args.checkpoint, args.threads)
    pairs = json.loads(args.pairs.read_text(encoding='utf-8'))
    prepared = json.loads((args.prepared/'manifest.json').read_text(encoding='utf-8'))
    lookup = {row['beatmap_id']:row for row in prepared['records']}
    identity.update(pairs_sha256=sha(args.pairs), prepared_manifest_sha256=sha(args.prepared/'manifest.json'),
                    extractor_sha256=sha(__file__), protocol_sha256=sha('docs/mapper-embedding-release-protocol.md'))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'geometry.json').write_text(json.dumps(audit_geometry(encoder), indent=2), encoding='utf-8')
    records = []; began = time.monotonic()
    for index, pair in enumerate(pairs['records']):
        row = lookup[pair['beatmap_id']]
        path = args.prepared/row['split']/'data'/str(row['beatmap_id'])/'audio.wav'
        if (sha(path) != row['decoded_audio_sha256'] or sha(row['audio']) != row['audio_sha256']
                or row['audio_sha256'] != pair['audio_sha256']):
            raise ValueError('Audio identity changed')
        output = args.output/str(row['beatmap_id']); output.mkdir(exist_ok=True)
        local_identity = identity | dict(audio_sha256=row['audio_sha256'], prepared_sha256=sha(path))
        done = output/'identity.json'
        if done.exists():
            cached = json.loads(done.read_text(encoding='utf-8'))
            if cached['input'] != local_identity or sha(output/'hidden.npz') != cached['sha256']:
                raise ValueError('Refuse incompatible partial cache')
        else:
            audio = loader(str(path), SAMPLE_RATE)
            duration = len(audio)/SAMPLE_RATE*1000
            start, end = max(0., duration/2-48000), min(duration, duration/2+48000)
            chunks = []; times = []; windows = []
            with torch.inference_mode():
                for base, positions, mask in window_positions(len(audio), start, end):
                    frame = np.zeros(WINDOW_SAMPLES, np.float32)
                    available = audio[base:base+WINDOW_SAMPLES]; frame[:len(available)] = available
                    hidden = encoder(frames=torch.from_numpy(frame)[None], return_dict=True).last_hidden_state[0]
                    if tuple(hidden.shape) != (1024,768) or not torch.isfinite(hidden).all():
                        raise ValueError('Invalid encoder output')
                    chunks.append(hidden.numpy()[mask].astype(np.float16))
                    times.append(positions/(SAMPLE_RATE/1000))
                    windows.append(dict(start_ms=base/SAMPLE_RATE*1000, end_ms=min(base+WINDOW_SAMPLES,len(audio))/SAMPLE_RATE*1000))
            positions = np.concatenate(times)
            if np.any(np.diff(positions) != 16): raise ValueError('Encoder retention produced clock gaps')
            np.savez_compressed(output/'hidden.npz', times_ms=positions, hidden=np.concatenate(chunks))
            cached = dict(input=local_identity, sha256=sha(output/'hidden.npz'), bounds=[float(positions[0]),float(positions[-1])],
                          duration_ms=duration, frames=len(positions), windows=windows)
            done.write_text(json.dumps(cached, indent=2), encoding='utf-8')
        records.append(dict(id=row['beatmap_id'], chart=row['chart'], chart_sha256=row['chart_sha256'], keys=row['keys'],
                            split=pair['split'], group=pair['group'], path=str(output), identity=cached))
        print(json.dumps(dict(stage='encode', completed=index+1, total=len(pairs['records']), id=row['beatmap_id'],
                              frames=cached['frames'], elapsed_seconds=round(time.monotonic()-began,1))), flush=True)
    manifest = dict(identity=identity, records=records, pretraining_dataset_sha256=pairs['pretraining_dataset_sha256'],
                    enabled_in_application=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
