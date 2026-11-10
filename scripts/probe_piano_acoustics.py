"""Frozen source-specific piano comparison, without timing or label inputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha

CHECKPOINT_SHA256 = 'c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141'
SOURCE_COMMIT = '0226e74cbc805660e34bbd6a8fed2083890ebb88'


def verify_specialist(root):
    # This checkout was extracted from an archive, so git would find ManiaLab's
    # parent repository. Verify each imported Python file against that archive.
    identity = json.loads((root/'verified-source.json').read_text(encoding='utf-8'))
    if (identity['commit'] != SOURCE_COMMIT
            or sha(root/'source-verification.zip') != identity['archive_sha256']
            or sha(root/'model.pth') != CHECKPOINT_SHA256):
        raise ValueError('Frozen specialist identity changed')
    import zipfile
    with zipfile.ZipFile(root/'source-verification.zip') as archive:
        prefix = 'piano_transcription_inference-'+SOURCE_COMMIT+'/'
        files = {name[len(prefix):] for name in archive.namelist()
                 if name.startswith(prefix+'piano_transcription_inference/') and name.endswith('.py')}
        if files != set(identity['python_files']):
            raise ValueError('Source verification omits imported files')
        for name, digest in identity['python_files'].items():
            if sha(root/'source'/name) != digest or (root/'source'/name).read_bytes() != archive.read(prefix+name):
                raise ValueError('Specialist source changed: '+name)
    return identity


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--backend', choices=('basic', 'specialist'), required=True)
    cli.add_argument('--root', type=Path, default=Path('test-data/smd/selected-v1/benchmark'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--resume-activations', type=Path,
                     help='Recover saved specialist activations from an interrupted identical-input run')
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve completed or failed inference')
    manifest = json.loads((args.root/'manifest.json').read_text(encoding='utf-8'))
    if sha('docs/piano-acoustic-protocol.md') != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    if sha('docs/piano-pedal-protocol.md') != manifest['pedal_protocol_sha256']:
        raise ValueError('Frozen pedal protocol changed')
    if args.backend == 'basic':
        from manialab.audio import infer, pitched_notes, rhythmic_evidence, source_evidence, refine_releases, fuse_attacks
        checkpoint = Path('assets/basic-pitch.onnx')
        model_identity = dict(model_sha256=sha(checkpoint), audio_code_sha256=sha('manialab/audio.py'))
    else:
        root = Path('test-data/piano-release-probe')
        checkpoint = root/'model.pth'
        source_identity = verify_specialist(root)
        sys.path[:0] = [str((root/'source').resolve()), str((root/'dependencies').resolve())]
        os.environ.update(TORCH_FORCE_WEIGHTS_ONLY_LOAD='1', MPLBACKEND='Agg')
        import torch
        from piano_transcription_inference import PianoTranscription
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = PianoTranscription(checkpoint_path=str(checkpoint), device='cpu')
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.model.load_state_dict(weights['model'], strict=True)
        del weights
        model_identity = dict(model_sha256=CHECKPOINT_SHA256, source_commit=SOURCE_COMMIT,
                              source_archive_sha256=source_identity['archive_sha256'], strict_weights=True,
                              onset_threshold=model.onset_threshold, offset_threshold=model.offset_threshod,
                              frame_threshold=model.frame_threshold)
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    identity = dict(backend=args.backend, manifest_sha256=sha(args.root/'manifest.json'),
                    runner_sha256=sha(__file__), **model_identity)
    (args.output/'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    if args.resume_activations:
        previous = json.loads((args.resume_activations/'identity.json').read_text(encoding='utf-8'))
        if args.backend != 'specialist' or any(previous[k] != identity[k]
                for k in ('backend','manifest_sha256','model_sha256','source_archive_sha256')):
            raise ValueError('Cannot recover activations from a different inference')
        if sha(args.resume_activations/'runner-snapshot.py') != previous['runner_sha256']:
            raise ValueError('Interrupted runner snapshot changed')
    reports = []
    for case in manifest['cases']:
        if sha(case['audio']) != case['audio_sha256']:
            raise ValueError('Benchmark waveform changed')
        rate, raw = wavfile.read(case['audio'])
        if rate != 22050 or raw.ndim != 1 or raw.dtype != np.int16:
            raise ValueError('Unexpected PCM')
        audio = raw[:round(case['duration_seconds']*rate)].astype(np.float32)/32768
        begin = time.monotonic()
        recovered = None
        if args.backend == 'basic':
            frames, onsets, times = infer(audio, checkpoint)
            attacks, _ = rhythmic_evidence(audio, estimate=False)
            attacks, _ = source_evidence(attacks, audio, audio)
            notes = pitched_notes(frames, onsets, times, len(audio)/rate, attacks=attacks)
            notes, _ = source_evidence(notes, audio, audio)
            notes, releases = refine_releases(notes, audio)
            notes, _, _ = fuse_attacks(notes, attacks)
            rows = [dict(onset=n.start/1000, offset=n.end/1000, pitch=n.pitch,
                         velocity=n.strength, program=0, is_drum=False) for n in notes]
        else:
            prior = args.resume_activations/(case['name']+'-activations.npz') if args.resume_activations else None
            if prior is not None and prior.exists():
                from piano_transcription_inference.utilities import RegressionPostProcessor
                with np.load(prior,allow_pickle=False) as saved:
                    activations={key:saved[key] for key in saved.files}
                processor=RegressionPostProcessor(model.frames_per_second,classes_num=model.classes_num,
                    onset_threshold=model.onset_threshold,offset_threshold=model.offset_threshod,
                    frame_threshold=model.frame_threshold,pedal_offset_threshold=model.pedal_offset_threshold)
                events,pedals=processor.output_dict_to_midi_events(activations)
                prediction=dict(est_note_events=events,est_pedal_events=pedals,output_dict=activations)
                recovered=dict(path=str(prior),sha256=sha(prior),identity_sha256=sha(args.resume_activations/'identity.json'))
            else:
                signal = resample_poly(audio, 320, 441).astype(np.float32)
                prediction = model.transcribe(signal, None)
            rows = [dict(onset=float(n['onset_time']), offset=float(n['offset_time']),
                         pitch=int(n['midi_note']), velocity=float(n['velocity']), program=0, is_drum=False)
                    for n in prediction['est_note_events']]
            np.savez_compressed(args.output/(case['name']+'-activations.npz'), **prediction['output_dict'])
            (args.output/(case['name']+'-pedals.json')).write_text(
                json.dumps([{key:float(value) for key,value in event.items()}
                            for event in prediction['est_pedal_events']], indent=2), encoding='utf-8')
        path = args.output/(case['name']+'-notes.json')
        path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        report = dict(name=case['name'], elapsed_seconds=time.monotonic()-begin, notes=len(rows),
                      notes_sha256=sha(path), audio_sha256=case['audio_sha256'],
                      duration_seconds=len(audio)/rate,
                      recovered_activations=recovered,
                      endpoints_beyond_input=sum(n['offset'] > len(audio)/rate for n in rows))
        reports.append(report)
        (args.output/'progress.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    (args.output/'report.json').write_text(json.dumps(dict(identity=identity, cases=reports,
        enabled_in_application=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
