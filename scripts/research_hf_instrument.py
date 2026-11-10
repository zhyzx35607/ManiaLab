"""Local-only FiloBass/FiloSax research adapter; never changes app defaults."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from math import gcd
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_guitar_vocal_benchmark import read_json, require, sha, verified_hash

REVISION = 'b7bec65a2b860aca72856b0feef58b5df407b777'
SOURCE_COMMIT = '0226e74cbc805660e34bbd6a8fed2083890ebb88'
SOURCE_ARCHIVE = 'ac7062720a03feb965ffc9e48e6246497c415d90a60780ea98206fb89cbff7e7'
MODELS = {
    'bass': dict(filename='filobass_20000_iterations.pth', bytes=98882013, iteration=20000,
                 sha256='cbc792a1b6b6002a73ee0a433c0fe5e097ee728034fe2ea70b24bd06b5026e84'),
    'sax': dict(filename='filosax_25k.pth', bytes=99341469, iteration=25000,
                sha256='448cf2c8ea6d4b77f7435f5b9a496211ea60300c5c17fa9c754da764f75f3a6a'),
}
SOURCE_ROOT = ROOT/'test-data/piano-release-probe'
WEIGHTS = ROOT/'test-data/instrument-model-extra-search-2026-11-10/hf-weights'
OUTPUTS = ('reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output')
RATE, WINDOW, HOP, FPS = 16000, 160000, 80000, 100
THRESHOLDS = dict(onset_threshold=.3, offset_threshold=.3, frame_threshold=.1)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_source(root=SOURCE_ROOT):
    root = Path(root)
    identity = read_json(root/'verified-source.json')
    require(identity['commit'] == SOURCE_COMMIT and identity['archive_sha256'] == SOURCE_ARCHIVE,
            'Frozen Kong source identity differs')
    verified_hash(root/'source-verification.zip', SOURCE_ARCHIVE)
    prefix = 'piano_transcription_inference-'+SOURCE_COMMIT+'/'
    with zipfile.ZipFile(root/'source-verification.zip') as archive:
        files = {n[len(prefix):] for n in archive.namelist()
                 if n.startswith(prefix+'piano_transcription_inference/') and n.endswith('.py')}
        require(files == set(identity['python_files']), 'Source manifest is incomplete')
        for name, digest in identity['python_files'].items():
            verified_hash(root/'source'/name, digest)
            require((root/'source'/name).read_bytes() == archive.read(prefix+name), 'Archived source differs')
    return identity


def verify_checkpoint(model, checkpoint=None):
    expected = MODELS[model]
    path = Path(checkpoint) if checkpoint is not None else WEIGHTS/expected['filename']
    require(path.stat().st_size == expected['bytes'], 'Checkpoint length differs')
    verified_hash(path, expected['sha256'])
    return path.resolve()


def load_model(model, checkpoint=None, source_root=SOURCE_ROOT):
    """Strict state loading, with no wrapper download or permissive fallback."""
    source_root = Path(source_root).resolve()
    source = verify_source(source_root)
    checkpoint = verify_checkpoint(model, checkpoint)
    sys.path[:0] = [str(source_root/'source'), str(source_root/'dependencies'),
                   str(ROOT/'test-data/rosvot-runtime')]
    import numpy as np
    import torch
    from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN
    import piano_transcription_inference.models as model_module
    require(Path(model_module.__file__).resolve() == source_root/'source/piano_transcription_inference/models.py',
            'Another Kong package is already imported')
    torch.set_num_threads(2)
    # A single process may load bass then sax, but changing interop twice fails.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    allowed = [(np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               np.ndarray, np.dtype, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(saved['iteration'] == MODELS[model]['iteration'], 'Training iteration differs')
    network = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=FPS, classes_num=88)
    network.load_state_dict(saved['model'], strict=True)
    require(all(torch.equal(value, saved['model'][key]) for key, value in network.state_dict().items()),
            'State differs after strict load')
    network.eval().requires_grad_(False).to('cpu')
    identity = dict(model=model, revision=REVISION, checkpoint=str(checkpoint),
        checkpoint_sha256=MODELS[model]['sha256'], checkpoint_iteration=saved['iteration'],
        source=source, strict_weights=True, source_model='Regress_onset_offset_frame_velocity_CRNN',
        device='cpu', intra_threads=2, inter_threads=1, thresholds=THRESHOLDS,
        versions={name:importlib.metadata.version(name) for name in ('torch','torchlibrosa','librosa','numpy','scipy')},
        torchlibrosa_source_sha256=sha(sys.modules['torchlibrosa.stft'].__file__),
        adapter_sha256=sha(__file__), decoder_sha256=sha(ROOT/'manialab/piano.py'),
        normalization=False, timing_quantization=False)
    return network, identity


def stitch_windows(windows):
    """Author's overlap discard on the sample clock, retaining padded frames."""
    import numpy as np
    require(set(windows) == set(OUTPUTS), 'Unexpected heads')
    shape = np.asarray(windows[OUTPUTS[0]]).shape
    require(len(shape) == 3 and shape[0] > 0 and shape[1:] == (1001,88), 'Invalid window shape')
    require(all(np.asarray(v).shape == shape and np.isfinite(v).all() for v in windows.values()),
            'Invalid window activations')
    if shape[0] == 1:
        return {key:value[0].copy() for key,value in windows.items()}
    return {key:np.concatenate([value[0,:750], *[row[250:750] for row in value[1:-1]], value[-1,250:1000]])
            for key,value in windows.items()}


def physical_heads(heads, original_samples, original_rate):
    import numpy as np
    require(type(original_samples) is int and original_samples > 0 and type(original_rate) is int and original_rate > 0,
            'Invalid physical input clock')
    frames = len(heads[OUTPUTS[0]])
    times = np.arange(frames, dtype=np.float64)/FPS
    # Integer inequality avoids float errors at exactly representable endpoints.
    valid = np.arange(frames, dtype=np.int64)*original_rate < original_samples*FPS
    require(valid.any() and int(valid.sum()) == (original_samples*FPS+original_rate-1)//original_rate,
            'Activations do not cover the real waveform')
    return {key:value[valid] for key,value in heads.items()}, times, valid


def infer(network, pcm):
    import numpy as np
    import torch
    require(pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm)>0 and np.isfinite(pcm).all(),
            'Expected finite mono float32 PCM')
    padded = ((len(pcm)+WINDOW-1)//WINDOW)*WINDOW
    starts = list(range(0, padded-WINDOW+1, HOP))
    pieces = {key:[] for key in OUTPUTS}
    with torch.inference_mode():
        for start in starts:
            chunk = np.zeros(WINDOW, dtype=np.float32)
            current = pcm[start:start+WINDOW]
            chunk[:len(current)] = current
            result = network(torch.from_numpy(chunk[None]))
            require(set(result) == set(OUTPUTS), 'Unexpected network heads')
            for key in OUTPUTS:
                value = result[key].detach().cpu().numpy()
                require(value.shape == (1,1001,88) and np.isfinite(value).all(), 'Invalid network output')
                pieces[key].append(value[0].copy())
    return {key:np.stack(values) for key,values in pieces.items()}, starts


def transcribe(network, identity, audio, output):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from manialab.piano import decode
    from piano_transcription_inference.utilities import RegressionPostProcessor
    output, audio = Path(output), Path(audio).resolve()
    require(not output.exists(), 'Preserve existing inference: '+str(output))
    rate, wave = wavfile.read(audio)
    require(wave.dtype == np.int16 and wave.ndim == 1 and len(wave)>0, 'Expected original mono int16 PCM')
    samples = wave.astype(np.float32)/32768
    common = gcd(rate,RATE)
    pcm = resample_poly(samples,RATE//common,rate//common).astype(np.float32) if rate != RATE else samples
    require(len(pcm) == (len(wave)*RATE+rate-1)//rate, 'Resampling changed the physical clock')
    input_identity = dict(audio=str(audio), audio_sha256=sha(audio), original_samples=len(wave),
        original_sample_rate=rate, resampled_samples=len(pcm), sample_rate=RATE,
        resampler='scipy.signal.resample_poly/default' if rate != RATE else 'identity',
        pcm_conversion='int16/32768 float32',
        pcm_sha256=hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest(),
        model_input_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest())
    output.mkdir(parents=True)
    started = time.monotonic()
    windows, starts = infer(network, pcm)
    heads = stitch_windows(windows)
    physical, times, valid = physical_heads(heads,len(wave),rate)
    np.savez_compressed(output/'windows.npz', starts=np.asarray(starts), **windows)
    np.savez_compressed(output/'activations.npz', times=times, valid_frame=valid, **heads)
    processor = RegressionPostProcessor(FPS,classes_num=88,pedal_offset_threshold=.2,**THRESHOLDS)
    raw, _ = processor.output_dict_to_midi_events({key:value.copy() for key,value in physical.items()})
    raw = [{key:float(v) if key in ('onset_time','offset_time') else int(v) for key,v in n.items()} for n in raw]
    events = decode(physical, **THRESHOLDS)
    order = lambda n:(n['midi_note'],n['onset_time'],n['offset_time'])
    expected = [dict(onset_time=n['onset'],offset_time=n['offset'],midi_note=n['pitch'],velocity=n['velocity']) for n in events]
    require(len(raw) == len(expected), 'Provenance decoder changed the author event count')
    for a,b in zip(sorted(raw,key=order),sorted(expected,key=order)):
        require(a['midi_note']==b['midi_note'] and a['velocity']==b['velocity']
                and abs(a['onset_time']-b['onset_time'])<=1e-5 and abs(a['offset_time']-b['offset_time'])<=1e-5,
                'Provenance decoder changed a physical event')
    notes = [dict(n,program=-1,is_drum=False,voice=identity['model']) for n in events]
    write_new(output/'raw-notes.json',raw)
    write_new(output/'notes.json',notes)
    report = dict(identity=identity,input=input_identity,notes=len(notes),seconds=time.monotonic()-started,
        context_start_seconds=0.,context_end_seconds=len(wave)/rate,valid_frames=int(valid.sum()),
        padded_frames=len(times),raw_decoder_parity=True,release_reasons=dict(Counter(n['release_reason'] for n in notes)),
        notes_sha256=sha(output/'notes.json'),raw_notes_sha256=sha(output/'raw-notes.json'),
        activations_sha256=sha(output/'activations.npz'),windows_sha256=sha(output/'windows.npz'),
        enabled_in_application=False,unknown_tails_retained=True)
    require(sha(__file__)==identity['adapter_sha256'] and sha(ROOT/'manialab/piano.py')==identity['decoder_sha256'],
            'Inference source changed during run')
    write_new(output/'report.json',report)
    return notes, report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--model',choices=tuple(MODELS),required=True)
    cli.add_argument('--checkpoint',type=Path)
    cli.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    cli.add_argument('--audio',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    require(not args.output.exists(),'Preserve existing output')
    network, identity = load_model(args.model,args.checkpoint,args.source_root)
    _, report = transcribe(network,identity,args.audio,args.output)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
