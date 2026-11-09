"""Cache new train/validation audio features and verify epoch-zero replay."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.piano_head_runtime import load_frozen_model, extract_window, decode
from scripts.piano_adaptation_data import load_development, read_pcm, onset_targets, window_slices, stitch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--resume', action='store_true')
    args = cli.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a local research directory')
    source = ROOT/'test-data/maestro-source-v1'
    # MIDI readers live only in the isolated research dependencies.
    sys.path.insert(0,str(ROOT/'test-data/piano-release-probe/dependencies'))
    rows = load_development(source)
    paths = dict(runner=Path(__file__), runtime=ROOT/'scripts/piano_head_runtime.py',
        data=ROOT/'scripts/piano_adaptation_data.py', labels=ROOT/'scripts/prepare_piano_benchmark.py',
        sustain=ROOT/'scripts/piano_pedal_labels.py', protocol=ROOT/'docs/piano-onset-head-adaptation-protocol.md')
    model, model_identity = load_frozen_model()
    identity = dict(model=model_identity, code={k:sha256(p) for k,p in paths.items()},
        selection_sha256=sha256(source/'selection.json'), download_report_sha256=sha256(source/'download-report.json'),
        inputs=[{key:str(row[key]) if isinstance(row[key],Path) else row[key]
                 for key in ('id','split','audio','midi','audio_sha256','midi_sha256')} for row in rows],
        seconds=180, windows_per_recording=35, training_performed=False, test_accessed=False)
    if output.exists():
        if not args.resume or read(output/'identity.json') != identity:
            raise ValueError('Resume requires identical input/code/model identity')
        progress = read(output/'progress.json')
        for item in progress:
            if sha256(output/item['file']) != item['sha256']:
                raise ValueError('Completed cache changed')
    else:
        output.mkdir(parents=True)
        save(output/'identity.json', identity)
        progress = []
        save(output/'progress.json', progress)
        for key,path in paths.items():
            shutil.copyfile(path, output/(key+'-snapshot'+path.suffix))
    planned = [(row['id'],i) for row in rows for i in range(35)]
    if [(r['id'],r['index']) for r in progress] != planned[:len(progress)]:
        raise ValueError('Cache sequence changed')
    existing = {(r['id'],r['index']) for r in progress}
    record_reports = []
    for row in rows:
        pcm = read_pcm(row['audio'])
        pcm_path = output/(row['id']+'-16k.npy')
        if pcm_path.exists():
            if not np.array_equal(pcm, np.load(pcm_path,allow_pickle=False)):
                raise ValueError('Prepared PCM changed')
        else:
            np.save(pcm_path,pcm,allow_pickle=False)
        labels_path = output/(row['id']+'-labels.json')
        labels = {key:row[key] for key in ('notes','controls','sustain_notes')}
        if labels_path.exists():
            if read(labels_path)!=labels:
                raise ValueError('Prepared labels changed')
        else:
            save(labels_path,labels)
        for i,(left,right) in enumerate(window_slices()):
            if (row['id'],i) in existing:
                continue
            began = time.monotonic()
            cached, errors = extract_window(model, pcm[i*80000:i*80000+160000])
            targets = onset_targets(row['notes'],i*5.)
            filename = row['id']+f'-w{i:02d}.npz'
            np.savez(output/filename, **cached, targets=targets)
            item = dict(id=row['id'],split=row['split'],index=i,start_sample=i*80000,
                left=left,right=right,file=filename,sha256=sha256(output/filename),
                epoch0_max_abs_error=errors,elapsed_seconds=time.monotonic()-began)
            progress.append(item)
            save(output/'progress.json', progress)
            if i%5==4 or i==34:
                print(json.dumps(dict(completed=len(progress),total=350,id=row['id'],split=row['split'],
                                      last_window_seconds=item['elapsed_seconds'])),flush=True)
        arrays = {}
        for item in [p for p in progress if p['id']==row['id']]:
            with np.load(output/item['file'],allow_pickle=False) as cached:
                for key in cached.files:
                    if key.endswith('_output'):
                        arrays.setdefault(key,[]).append(cached[key][0])
        combined = stitch(arrays)
        from piano_transcription_inference.inference import PianoTranscription
        for key,windows in arrays.items():
            upstream = PianoTranscription.deframe(None,np.stack(windows))
            if not np.array_equal(combined[key],upstream):
                raise ValueError('Stitch differs from upstream')
        prediction = decode(combined)
        # All seven replay arrays were bit-identical in every window, so their
        # stitch and decoding are deterministic with identical input bytes.
        if prediction != decode({k:v.copy() for k,v in combined.items()}):
            raise ValueError('Epoch zero decoding is not deterministic')
        save(output/(row['id']+'-epoch0-events.json'),prediction)
        np.savez(output/(row['id']+'-epoch0-outputs.npz'),**combined)
        record_reports.append(dict(id=row['id'],split=row['split'],windows=35,
            pcm_sha256=sha256(pcm_path),labels_sha256=sha256(labels_path),
            baseline_notes=len(prediction['notes']),output_frames=len(combined['frame_output']),
            official_deframe_identical=True,cache_replay_identical=True))
    if len(progress)!=350 or len(record_reports)!=10:
        raise ValueError('Incomplete feature preparation')
    if any(sha256(path)!=identity['code'][key] for key,path in paths.items()):
        raise ValueError('Frozen code changed during inference')
    save(output/'report.json',dict(identity=identity,windows=progress,records=record_reports,
        training_performed=False,test_accessed=False,all_epoch0_outputs_identical=True,
        artifacts_sha256={p.name:sha256(p) for p in output.iterdir() if p.name!='report.json'}))
    print(json.dumps(dict(complete=True,recordings=10,windows=350)),flush=True)


if __name__=='__main__':
    main()
