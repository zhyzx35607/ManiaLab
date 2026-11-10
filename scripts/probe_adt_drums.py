"""Run fixed official ADT_STR inference, retaining tokens and decoding defects."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.adt_token_audit import audit_tokens, group, NAMES

SOURCE_COMMIT = '77dbef225e7029478ebfb916f20c7a00274f0f12'
MODEL_SHA = 'f94d1773d929166e0a3f63d305975aa6667512f8a71d7706872e1f8b23ca930b'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def roster():
    manifest = ROOT/'test-data/physical-drums/dataset-v2/manifest.json'
    input_path = ROOT/'test-data/drum-adt-str-probe/input-roster.json'
    inputs = read(input_path)
    if sha256(manifest) != inputs['manifest_sha256']:
        raise ValueError('Prepared source roster differs from original manifest')
    records = [dict(id='sakura', audio=ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/drums.wav',
                    audio_sha256='97762166e5fef541b0cb8c3a048b03e1cc75f5c1e20575ddda93982ae06ccc9d',
                    dataset='Nova', previous_split='author_development')]
    rows = inputs['records']
    if any(set(r) != {'id', 'audio', 'audio_sha256', 'dataset', 'split'} or r['dataset'] != 'MDB' for r in rows):
        raise ValueError('Inference roster must contain only source metadata')
    if len(rows) != 23 or len({r['id'] for r in rows}) != 23:
        raise ValueError('Expected all twenty-three existing MDB recordings')
    records += [dict(id=r['id'], audio=(ROOT/r['audio']).resolve(), audio_sha256=r['audio_sha256'],
                     dataset='MDB', previous_split=r['split']) for r in rows]
    for row in records:
        if sha256(row['audio']) != row['audio_sha256']:
            raise ValueError('Original audio changed: '+row['id'])
        row['audio'] = str(row['audio'])
    return records, sha256(manifest), sha256(input_path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--resume', action='store_true')
    args = cli.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a local research output')
    probe = ROOT/'test-data/drum-adt-str-probe'
    source, checkpoint = probe/'source', probe/'checkpoint'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != SOURCE_COMMIT or dirty or sha256(checkpoint/'model.safetensors') != MODEL_SHA:
        raise ValueError('Fixed official source/checkpoint differs')
    sys.path[:0] = [str(source), str(probe/'dependencies')]
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import transformers
    import yaml
    from safetensors.torch import load_file
    from model import ADTModel
    from config import ADTModelConfig
    from adt_transcriber import ADTTranscriber
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cfg = yaml.safe_load((checkpoint/'adt_config.yaml').read_text(encoding='utf-8'))
    if cfg['inference']['max_length'] != 512 or cfg['inference']['batch_size'] != 8:
        raise ValueError('Unexpected official inference settings')
    records, manifest_hash, roster_hash = roster()
    paths = {'runner': Path(__file__), 'audit': ROOT/'scripts/adt_token_audit.py',
             'protocol': ROOT/'docs/drum-adt-str-protocol.md'}
    identity = dict(source_commit=revision, model_sha256=MODEL_SHA,
        upstream_python={str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*.py'))},
        license_sha256=sha256(source/'LICENSE'), configs={p.name: sha256(p) for p in checkpoint.glob('*')},
        code={name: sha256(path) for name, path in paths.items()},
        torch=torch.__version__, torchaudio=torchaudio.__version__, transformers=transformers.__version__,
        numpy=np.__version__, soundfile=sf.__version__, device='cpu', threads=2, inter_threads=1,
        seed=42, batch_size=8, max_length=512, chunk_samples=61440, sample_rate=24000,
        manifest_sha256=manifest_hash, input_roster_sha256=roster_hash, inputs=records, training_performed=False,
        reserved_piano_test_accessed=False, application_enabled=False)
    if output.exists():
        if not args.resume or read(output/'identity.json') != identity:
            raise ValueError('Resume requires identical evidence and implementation')
        progress = read(output/'progress.json')
        for row in progress:
            for name, digest in row['artifacts_sha256'].items():
                if sha256(output/row['id']/name) != digest:
                    raise ValueError('Completed recording changed')
    else:
        output.mkdir(parents=True)
        save(output/'identity.json', identity)
        save(output/'progress.json', [])
        progress = []
        for name, path in paths.items():
            shutil.copyfile(path, output/(name+'-snapshot'+path.suffix))
    if [r['id'] for r in progress] != [r['id'] for r in records[:len(progress)]]:
        raise ValueError('Completed recording sequence changed')
    model = ADTModel(ADTModelConfig(**read(checkpoint/'config.json')))
    tensors = load_file(str(checkpoint/'model.safetensors'), device='cpu')
    model.load_state_dict(tensors, strict=True)
    if any(not torch.equal(value, tensors[key]) for key, value in model.state_dict().items()):
        raise ValueError('Model differs after strict checkpoint load')
    model.eval().requires_grad_(False)
    transcriber = ADTTranscriber(model, cfg, device='cpu')
    parameter_count = sum(p.numel() for p in model.parameters())
    save(output/'model-proof.json', dict(strict_tensor_count=len(tensors), parameters=parameter_count,
                                       all_tensors_identical=True))
    print(json.dumps(dict(loaded=True, parameters=parameter_count, tensors=len(tensors))), flush=True)
    for record in records[len(progress):]:
        name = record['id']
        folder = output/name
        if folder.exists():
            raise ValueError('Partial recording exists; preserve it and audit before restarting')
        folder.mkdir()
        started = time.monotonic()
        raw, rate = sf.read(record['audio'], dtype='float32', always_2d=True)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite source PCM')
        waveform = transcriber._prepare_waveform(torch.from_numpy(raw.T.copy()), rate)
        pcm = waveform.cpu().numpy()
        np.save(folder/'input24k.npy', pcm, allow_pickle=False)
        tokens_all, windows, strict_events = [], [], []
        original_sample = model.sample

        def capture(*a, **kw):
            src = kw['src']
            batch_start = len(windows)
            for j in range(len(src)):
                start = (batch_start+j)*61440
                valid = min(61440, waveform.shape[-1]-start)
                if not torch.equal(src[j, :valid], waveform[0, start:start+valid]):
                    raise ValueError('Wrapper passed a different physical input')
                if torch.count_nonzero(src[j, valid:]):
                    raise ValueError('Final padding is not zero')
            result = original_sample(*a, **kw)
            for row in result:
                index = len(windows)
                start = index*61440
                valid = min(61440, waveform.shape[-1]-start)
                sequence = row.cpu().tolist()
                audit = audit_tokens(sequence, valid_samples=valid, max_length=512)
                official = transcriber.tokenizer.decode(np.array(sequence, dtype=np.int64)).tolist()
                tokens_all.append(sequence)
                windows.append(dict(index=index, start_sample=start, valid_samples=valid,
                                    official_local=official, audit=audit))
                for event in audit['events']:
                    # Integer token/sample clock; no chart lead or frame-center correction.
                    strict_events.append(dict(**event, window_index=index,
                        onset=(start+event['tick']*240)/24000,
                        stream_terminated=audit['terminated']))
            save(folder/'tokens-progress.json', tokens_all)
            print(json.dumps(dict(recording=name, windows=len(windows),
                expected_windows=(waveform.shape[-1]+61439)//61440,
                seconds=time.monotonic()-started,
                issues=sum(len(w['audit']['issues']) for w in windows))), flush=True)
            return result

        model.sample = capture
        try:
            official_notes = transcriber._transcribe_waveform(waveform, batch_size=8, max_length=512)
        finally:
            model.sample = original_sample
        official = [dict(onset=float(row[0]), display_offset=float(row[1]), custom_pitch=int(row[2]),
                         velocity=int(row[3]), kind=group(int(row[2]))) for row in official_notes.tolist()]
        save(folder/'tokens.json', tokens_all)
        save(folder/'windows.json', windows)
        save(folder/'official-events.json', official)
        save(folder/'audited-events.json', strict_events)
        summary = dict(id=name, dataset=record['dataset'], previous_split=record['previous_split'],
            audio_sha256=record['audio_sha256'], source_rate=rate, source_channels=raw.shape[1],
            original_samples=len(raw), duration_seconds=len(raw)/rate, input_samples=waveform.shape[-1],
            windows=len(windows), official_events=len(official), audited_events=len(strict_events),
            issues=dict(Counter(i['kind'] for w in windows for i in w['audit']['issues'])),
            unterminated_windows=sum(not w['audit']['terminated'] for w in windows),
            official_input_outside=sum(not 0 <= n['onset'] < len(raw)/rate for n in official),
            audited_input_outside=sum(not e['in_audio'] for e in strict_events),
            audited_class_counts=dict(Counter(e['class_name'] for e in strict_events)),
            elapsed_seconds=time.monotonic()-started,
            artifacts_sha256={p.name: sha256(p) for p in folder.iterdir()})
        progress.append(summary)
        save(output/'progress.json', progress)
        print(json.dumps(dict(completed_recording=name, **{k: summary[k] for k in
            ('official_events', 'audited_events', 'issues', 'elapsed_seconds')})), flush=True)
    if any(not torch.equal(value, tensors[key]) for key, value in model.state_dict().items()):
        raise ValueError('Frozen model state changed')
    if any(sha256(path) != identity['code'][name] for name, path in paths.items()):
        raise ValueError('Implementation changed during inference')
    save(output/'report.json', dict(identity=identity, records=progress, classes=NAMES,
        strict_tensor_count=len(tensors), parameters=parameter_count, frozen_model_unchanged=True,
        training_performed=False, application_enabled=False, direct_listening_performed=False,
        artifacts_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(complete=True, recordings=len(progress))), flush=True)


if __name__ == '__main__':
    main()
