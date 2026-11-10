"""Run frozen upstream drum separation preserving its STFT clock and float PCM."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--source', type=Path, default=Path('test-data/larsnet/source'))
    cli.add_argument('--checkpoint', type=Path, default=Path('test-data/larsnet'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--threads', type=int, default=2)
    cli.add_argument('--ffmpeg', type=Path, help='Use the existing app FFmpeg without installing it into this research environment')
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier separation experiments')
    import numpy as np
    import torch
    import yaml
    from scipy.io import wavfile
    sys.path.insert(0, str(args.source.resolve()))
    from unet import UNet
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    origin = read(args.checkpoint / 'source.json')
    commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != origin['source_commit'] or commit != '17d631fd18e77ee2f1d23ee7b3fc0fb46ae629e2':
        raise ValueError('LarsNet source changed')
    for row in origin['files']:
        if sha(args.checkpoint / row['path']) != row['sha256']:
            raise ValueError('LarsNet checkpoint changed')
    cfg = yaml.safe_load((args.source / 'config.yaml').read_text())
    rate = cfg['global']['sr']
    if rate != 44100:
        raise ValueError('Unreviewed separation sample rate')
    original = next(row for row in read(args.root / 'manifest.json')['audio'] if row['id'] == 'drums')
    path = args.root / original['path']
    if sha(path) != original['sha256']:
        raise ValueError('Original Nova drum track changed')
    if args.ffmpeg:
        ffmpeg = str(args.ffmpeg.resolve())
    else:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-i', str(path), '-vn', '-ac', '2', '-ar', str(rate),
        '-f', 'f32le', 'pipe:1'], capture_output=True, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    pcm = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2).copy()
    if not np.isfinite(pcm).all() or len(pcm) < rate:
        raise ValueError('Invalid original audio')
    args.output.mkdir(parents=True)
    wavfile.write(args.output / 'input-float.wav', rate, pcm)
    identity = dict(source_commit=commit, source_files={p.name: sha(p) for p in args.source.glob('*.py')},
        config_sha256=sha(args.source/'config.yaml'), checkpoint=origin, original_audio_sha256=original['sha256'],
        input_float_sha256=sha(args.output/'input-float.wav'), runner_sha256=sha(__file__), ffmpeg_sha256=sha(ffmpeg),
        protocol_sha256=sha('docs/larsnet-protocol.md'), torch=torch.__version__, sample_rate=rate,
        samples=len(pcm), channels=2, threads=args.threads, wiener=False, batch_size=1)
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    shutil.copyfile(__file__, args.output / 'runner-snapshot.py')
    shutil.copyfile('docs/larsnet-protocol.md', args.output / 'protocol-snapshot.md')
    audio = torch.from_numpy(pcm.T.copy())[None]
    began = time.monotonic()
    stems = {}
    with torch.inference_mode():
        magnitude = phase = folded = None
        for index, (role, relative) in enumerate(cfg['inference_models'].items()):
            model = UNet(input_size=(2, cfg[role]['F'], cfg[role]['T']), device='cpu')
            checkpoint = torch.load(args.checkpoint / 'model' / relative, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()
            del checkpoint
            if magnitude is None:
                magnitude, phase = model.utils.batch_stft(audio)
                folded = model.utils.fold_unet_inputs(magnitude)
            if index == 0:
                batch, _ = model(folded[:2])
                singles = torch.cat([model(chunk[None])[0] for chunk in folded[:2]], dim=0)
                torch.testing.assert_close(batch, singles, atol=1e-5, rtol=1e-4)
                identity['batch_equivalence_max_error'] = float((batch-singles).abs().max())
                del batch, singles
            chunks = []
            for at, chunk in enumerate(folded):
                predicted, _ = model(chunk[None])
                if not torch.isfinite(predicted).all():
                    raise ValueError('Nonfinite separated spectrogram')
                chunks.append(predicted)
                print(json.dumps(dict(role=role, completed=at+1, total=len(folded), seconds=round(time.monotonic()-began, 1))), flush=True)
            separated = model.utils.unfold_unet_outputs(torch.cat(chunks), magnitude.size())
            output = model.utils.batch_istft(separated, phase, trim_length=len(pcm))[0].T.contiguous().numpy()
            if output.shape != pcm.shape or not np.isfinite(output).all():
                raise ValueError('Separated stem clock or values changed')
            filename = args.output / (role + '.wav')
            wavfile.write(filename, rate, output.astype(np.float32))
            stems[role] = dict(sha256=sha(filename), samples=len(output), peak=float(np.max(abs(output))),
                rms=float(np.sqrt(np.mean(output.astype(float)**2))))
            (args.output / 'partial-stems.json').write_text(json.dumps(stems, indent=2), encoding='utf-8')
            del model, chunks, separated, output, predicted
    report = dict(identity=identity, stems=stems, elapsed_seconds=time.monotonic()-began,
        enabled_in_application=False, notice='Frozen audio separation only; not authored selection, transcription, or LN duration validation.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(stems=stems, elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
