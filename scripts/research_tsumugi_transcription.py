"""Run pinned per-instrument Tsumugi weights with raw pre-MIDI evidence.

Research adapter: complete bounded recordings, CPU 2/1 threads, no quantization
or application default changes. Run only when the serial neural slot is free.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

SOURCE_COMMIT = 'eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38'
MODEL_REVISION = 'c6ee8a22626cd7185031ee71ce7cea27ec862781'
MODELS = {
    'bass_v2': '807eedb409c77a710357d57c9e47d17821c4518db3b09c78d8dabd205a2b604a',
    'guitar_v1_5': '59c8335517d1faf7d735fef29e061181d87b96385f2e9a08332791e2b64affd4',
    'other_v1_5': 'e836d5b30a29ad2301726aedb9ae348db927fbb4813359cdfb2e304ba3084501',
    'vocal_harmony_v1_5': '9f068d21313039810c82ede5f01e30bf8bc25fe2ba42f2cad16f57a2095b5e97',
}
RUNTIME_PACKAGES = ('torch','torchaudio','numpy','scipy','soundfile','einops','pretty-midi','tqdm')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runtime_versions():
    return dict(python=sys.version, packages={name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES})


def verify_source(source):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != SOURCE_COMMIT or git('diff', 'HEAD', '--name-only'):
        raise ValueError('Tsumugi source differs from reviewed commit')
    return {name: sha(source/name) for name in git('ls-files', '*.py', '*.json').splitlines()}


@contextmanager
def trace_interval_construction(stitcher_type):
    """Observe V1 construction without changing arguments, output or decoder state."""
    original = stitcher_type._build_interval_note
    rows, object_rows = [], {}
    def observed(stitcher, **kwargs):
        note = original(stitcher, **kwargs)
        row = dict(arguments=copy.deepcopy(kwargs), hop_length=stitcher.hop_length,
            total_audio_samples=stitcher.total_audio_frames,
            constructed_note=None if note is None else asdict(note),
            v1_slot_index=None)
        rows.append(row)
        if note is not None:
            object_rows[id(note)] = len(rows)-1
        return note
    stitcher_type._build_interval_note = observed
    try:
        yield rows, object_rows
    finally:
        stitcher_type._build_interval_note = original


def convert_notes(raw, *, sample_rate, total_samples, classes):
    """Keep all decoded notes, slots and repeats without MIDI export pruning.

    finalize() forces offset flags and can synthesize gaps/trim overlaps. Until
    each final event has a traced pre-stitch source, none of these flags alone
    constitute observed release evidence for the LN planner.
    """
    if sample_rate <= 0 or total_samples <= 0:
        raise ValueError('Invalid output clock')
    result = []
    for note in raw:
        begin, end = note['start_sample'], note['end_sample']
        instrument, pitch = note['instrument_id'], note['pitch']
        if (type(begin) is not int or type(end) is not int or not 0 <= begin < end <= total_samples
                or type(instrument) is not int or not 0 <= instrument < len(classes)
                or type(pitch) is not int or not 0 <= pitch <= 127):
            raise ValueError('Invalid decoded note; do not silently drop or trim')
        name = classes[instrument]
        result.append(dict(onset=begin/sample_rate, offset=end/sample_rate, pitch=pitch,
            program=-1, is_drum=name == 'drums', instrument_class=name, instrument_class_id=instrument,
            program_semantics='broad_model_class_not_exact_gm_program',
            strength=1., strength_semantics='unit_weight_not_confidence',
            slot_index=note['slot_index'], model_has_onset=note['has_onset'],
            model_has_offset_after_stitching=note['has_offset'],
            instrument_candidates=note['instrument_candidates'],
            release_reason='unverified_stitched_boundary'))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--audio', type=Path, required=True)
    cli.add_argument('--model', choices=tuple(MODELS), required=True)
    cli.add_argument('--stem', choices=('all', 'bass', 'guitar', 'other', 'vocals'), default='all',
        help='Known separator output only; never infer this from benchmark annotation')
    cli.add_argument('--source', type=Path, default=Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi'))
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/rosvot-runtime'))
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    source, output, audio = args.source.resolve(), args.output.resolve(), args.audio.resolve()
    if output.exists():
        raise ValueError('Use a new output directory and preserve earlier evidence')
    sources = verify_source(source)
    checkpoint_path = source/'checkpoints'/('best_model_'+args.model+'.pth')
    if sha(checkpoint_path) != MODELS[args.model]:
        raise ValueError('Pinned Tsumugi checkpoint changed')
    if args.dependencies.exists():
        sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str(source))
    import numpy as np
    import soundfile as sf
    import torch
    from instrument_agnostic_amt.amt.cli.infer import resolve_inference_settings
    from instrument_agnostic_amt.amt.inference.audio import load_audio
    from instrument_agnostic_amt.amt.inference.windowed import decode_notes, WindowNoteStitcher
    from instrument_agnostic_amt.amt.inference.instruments import resolve_stem_instrument_class_ids, filter_supported_instrument_class_ids
    from instrument_agnostic_amt.amt.modeling.checkpoints import coerce_model_config, extract_model_config, extract_training_args, load_compatible_weights
    from instrument_agnostic_amt.amt.modeling.model import AudioSemiCRFTransformer
    from instrument_agnostic_amt.taxonomy.instrument_classes import INSTRUMENT_CLASSES
    from types import SimpleNamespace

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    pcm, original_rate = sf.read(audio, dtype='float32', always_2d=True)
    duration = len(pcm)/original_rate
    if not .1 <= duration <= 60 or pcm.shape[1] not in (1, 2) or not np.isfinite(pcm).all():
        raise ValueError('Expected a complete finite mono/stereo recording of 0.1–60 seconds')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = coerce_model_config(extract_model_config(checkpoint), for_inference=True)
    if config.semi_crf_version != 'v1':
        raise ValueError('This raw window evidence protocol has only reviewed V1 decoding')
    training = extract_training_args(checkpoint)
    model = AudioSemiCRFTransformer(config).eval()
    loaded = load_compatible_weights(model, checkpoint, prefer_ema=True, require_complete_backbone=True)
    if loaded.missing_keys or loaded.unexpected_keys or loaded.shape_mismatches:
        raise ValueError('Inspect incomplete weight load before inference: '+str(loaded))
    defaults = SimpleNamespace(window_ms=None, stride_ms=None, semi_crf_track_batch_size=None,
        window_batch_size=1, merge_gap_ms=None, merge_onset_ms=50., silence_gate_rms_dbfs=-72.,
        note_bias=0., disable_tqdm=True, no_boundary_head=False, semi_crf_backend='torch',
        semi_crf_sparse_decode=False, semi_crf_sparse_topk_per_start=16, semi_crf_sparse_score_threshold=None,
        semi_crf_sparse_max_span_ms=None, instrument_pair_infer_topk=256,
        instrument_pair_gate_threshold=-3., instrument_pair_max_pairs=512)
    settings = resolve_inference_settings(config, training, defaults)
    allowed = None if args.stem == 'all' else resolve_stem_instrument_class_ids(args.stem)
    settings = replace(settings, allowed_instrument_ids=filter_supported_instrument_class_ids(
        allowed, num_model_classes=config.num_instrument_classes))
    wave = load_audio(audio, target_sample_rate=config.sample_rate)
    expected_samples = math.ceil(len(pcm)*config.sample_rate/original_rate)
    if tuple(wave.shape) != (2, expected_samples) or not torch.isfinite(wave).all():
        raise ValueError('Unexpected upstream resampling/channel clock')
    identity = dict(source_commit=SOURCE_COMMIT, source_files=sources, runner_sha256=sha(__file__),
        model_revision=MODEL_REVISION, model=args.model, model_sha256=MODELS[args.model],
        audio=str(audio), audio_sha256=sha(audio), original_sample_rate=original_rate,
        original_samples=len(pcm), original_channels=pcm.shape[1],
        pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
        resampled_samples=wave.shape[-1], resampled_sample_rate=config.sample_rate,
        resampled_pcm_sha256=hashlib.sha256(wave.numpy().astype('<f4').tobytes()).hexdigest(),
        resampler='torchaudio.functional.resample/default', normalization=False,
        mono_policy='duplicate_to_stereo', inference_scope='full_original_audio',
        intra_threads=2, inter_threads=1, amp=False, compile=False, seed=42,
        stem_filter=args.stem, evidence_revision=2, settings=asdict(settings), model_config=asdict(config),
        versions=runtime_versions(), weight_load=asdict(loaded))
    output.mkdir(parents=True)
    def write(name, data):
        (output/name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    write('identity.json', identity)
    windows = []
    def record_window(stitcher, start):
        for notes in stitcher.notes_by_pair.values():
            for note in notes:
                # V1 fills slot_index after construction; capture it without
                # treating the earlier default zero as the model's pitch slot.
                constructions[object_rows[id(note)]]['v1_slot_index'] = note.slot_index
        windows.append(dict(window_start_sample=start,
            instrument_id_semantics='v1_stitch_placeholder_use_instrument_candidates_for_predictions',
            tracks={str(track):[asdict(n) for n in notes] for track, notes in stitcher.notes_by_pair.items()}))
        print(json.dumps(dict(stage='window', start_sample=start, windows=len(windows))), flush=True)
    print(json.dumps(dict(stage='loaded', model=args.model, duration_seconds=duration)), flush=True)
    started = time.monotonic()
    with trace_interval_construction(WindowNoteStitcher) as (constructions, object_rows):
        with torch.inference_mode():
            decoded, stats = decode_notes(model, config, wave, instrument_filter_id=None,
                device=torch.device('cpu'), amp_enabled=False, amp_dtype=torch.float32, settings=settings,
                velocity=100, on_window_consumed=record_window)
    if len(constructions) != stats['decoded_interval_count'] or any(
            row['constructed_note'] is not None and
            (type(row['v1_slot_index']) is not int or not 0 <= row['v1_slot_index'] < config.num_pitch_slots)
            for row in constructions):
        raise ValueError('Construction trace lost a decoded interval or pitch slot')
    raw = [asdict(note) for note in decoded]
    events = convert_notes(raw, sample_rate=config.sample_rate, total_samples=wave.shape[-1], classes=INSTRUMENT_CLASSES)
    write('raw-notes.json', raw)
    write('window-notes.json', windows)
    write('interval-constructions.json', constructions)
    write('notes.json', events)
    report = dict(identity=identity, context_start_seconds=0., context_end_seconds=duration,
        elapsed_seconds=time.monotonic()-started, notes=len(events), stats=stats,
        instrument_classes=dict(Counter(n['instrument_class'] for n in events)),
        release_reasons=dict(Counter(n['release_reason'] for n in events)),
        notes_sha256=sha(output/'notes.json'), raw_notes_sha256=sha(output/'raw-notes.json'),
        window_notes_sha256=sha(output/'window-notes.json'),
        interval_constructions_sha256=sha(output/'interval-constructions.json'), enabled_in_application=False,
        limitations=['Finalizer can force offset flags; all stitched tails remain unverified for LN use.',
            'Raw model class is not calibrated confidence or exact GM instrument truth.',
            'CPU uses installed Torch 2.9.1, not upstream pinned accelerator runtime.',
            'No MIDI minimum-duration pruning, overlap trimming or track-count remapping is applied.'])
    write('report.json', report)
    print(json.dumps(dict(stage='complete', notes=len(events), elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
