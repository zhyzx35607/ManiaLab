"""Isolated ADTOF drum-timbre probe; never changes application defaults.

Loads every checkpoint tensor strictly and stores raw five-class activations.
No chart, BPM, quantization, or author labels are supplied during inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path('test-data/adtof/source'))
    cli.add_argument('--audio',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--threads',type=int,default=1)
    args=cli.parse_args()
    if (args.output/'report.json').exists():raise ValueError('Choose a fresh output folder to preserve evidence')
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source.resolve()/'src'))
    import numpy as np
    import torch
    from adtof_pytorch.model import create_frame_rnn_model,calculate_n_bins,load_audio_for_model
    from adtof_pytorch.post_processing import PeakPicker,LABELS_5,FRAME_RNN_THRESHOLDS
    torch.set_num_threads(args.threads);torch.manual_seed(42)
    weights=args.source/'src/adtof_pytorch/data/adtof_frame_rnn_pytorch_weights.pth'
    identity=dict(audio_sha256=sha(args.audio),weights_sha256=sha(weights),
        source_commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
        runner_sha256=sha(__file__),thresholds=FRAME_RNN_THRESHOLDS,fps=100,
        device='cpu',threads=args.threads)
    started=time.monotonic()
    model=create_frame_rnn_model(calculate_n_bins()).eval()
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint.get('model_weights',checkpoint),strict=True)
    print(json.dumps(dict(stage='loaded',identity=identity)),flush=True)
    frames=load_audio_for_model(str(args.audio))
    print(json.dumps(dict(stage='spectrogram',shape=list(frames.shape))),flush=True)
    with torch.inference_mode():activation=model(frames).cpu().numpy()
    if not np.isfinite(activation).all():raise ValueError('Non-finite drum activations')
    np.savez_compressed(args.output/'activations.npz',activations=activation)
    peaks=PeakPicker(thresholds=FRAME_RNN_THRESHOLDS,fps=100).pick(activation,labels=LABELS_5)[0]
    names={35:'kick',38:'snare',47:'tom',42:'hi-hat',49:'cymbal'}
    events=sorted([dict(start_ms=round(t*1000),kind=names[pitch],midi_pitch=pitch)
        for pitch,times in peaks.items() for t in times],key=lambda e:(e['start_ms'],e['midi_pitch']))
    (args.output/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    report=dict(identity=identity,duration_seconds=(frames.shape[1]-1)/100,
        class_counts={names[p]:len(v) for p,v in peaks.items()},elapsed_seconds=time.monotonic()-started,
        enabled_in_application=False,weights_loaded_strictly=True,
        protocol='Five-class audio-only transcription with supplied model thresholds. No author labels, beat grid or tuning.',
        source='https://github.com/xavriley/ADTOF-pytorch',
        origin='https://github.com/MZehren/ADTOF')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
