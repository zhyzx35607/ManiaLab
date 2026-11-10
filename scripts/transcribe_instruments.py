"""Optional portable Tsumugi worker with traced, explicitly selected model tails.

Uses the validated whole-song window path and never requires Git on the target.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_tsumugi_transcription import (
    convert_notes, runtime_versions, trace_interval_construction,
)

from manialab.instrument_bundle import MODELS, MODEL_REVISION, SOURCE_COMMIT, VOICE_MODELS, sha, verify_source, verify_model, runtime_identity
from scripts.audit_tsumugi_releases import audit_replay, plain
from scripts.tsumugi_candidate_adapter import tsumugi_candidates


def code_source(name):
    root = Path(sys._MEIPASS)/"source-evidence" if getattr(sys,"frozen",False) else Path(__file__).resolve().parents[1]
    return root/name


def validate_pcm(pcm, sample_rate):
    import numpy as np
    if (type(sample_rate) is not int or sample_rate <= 0 or pcm.ndim != 2
            or pcm.shape[1] not in (1, 2) or not np.isfinite(pcm).all()
            or not .1 <= len(pcm) / sample_rate <= 1200):
        raise ValueError('Expected finite mono/stereo PCM of 0.1–1200 seconds')
    return len(pcm) / sample_rate


def run(args):
    source, output, audio = (args.assets/"source").resolve(), args.output.resolve(), args.audio.resolve()
    model_name = VOICE_MODELS[args.voice]
    if output.exists():
        raise ValueError('Use a new output directory and preserve earlier evidence')
    sources = verify_source(source)
    checkpoint_path = verify_model(args.assets/"checkpoints",model_name)
    if args.dependencies.exists():
        sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str(source))
    import soundfile as sf
    import torch
    from instrument_agnostic_amt.amt.cli.infer import resolve_inference_settings
    from instrument_agnostic_amt.amt.inference.audio import load_audio
    from instrument_agnostic_amt.amt.inference.windowed import decode_notes, WindowNoteStitcher
    from instrument_agnostic_amt.amt.modeling.checkpoints import (
        coerce_model_config, extract_model_config, extract_training_args, load_compatible_weights,
    )
    from instrument_agnostic_amt.amt.modeling.model import AudioSemiCRFTransformer
    from instrument_agnostic_amt.taxonomy.instrument_classes import INSTRUMENT_CLASSES

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    pcm, original_rate = sf.read(audio, dtype='float32', always_2d=True)
    duration = validate_pcm(pcm, original_rate)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = coerce_model_config(extract_model_config(checkpoint), for_inference=True)
    if config.semi_crf_version != 'v1':
        raise ValueError('Only traced V1 decoding has been reviewed')
    training = extract_training_args(checkpoint)
    model = AudioSemiCRFTransformer(config).eval()
    loaded = load_compatible_weights(model, checkpoint, prefer_ema=True, require_complete_backbone=True)
    if loaded.missing_keys or loaded.unexpected_keys or loaded.shape_mismatches:
        raise ValueError('Incomplete model load: ' + str(loaded))
    defaults = SimpleNamespace(window_ms=None, stride_ms=None, semi_crf_track_batch_size=None,
        window_batch_size=1, merge_gap_ms=None, merge_onset_ms=50., silence_gate_rms_dbfs=-72.,
        note_bias=0., disable_tqdm=True, no_boundary_head=False, semi_crf_backend='torch',
        semi_crf_sparse_decode=False, semi_crf_sparse_topk_per_start=16,
        semi_crf_sparse_score_threshold=None, semi_crf_sparse_max_span_ms=None,
        instrument_pair_infer_topk=256, instrument_pair_gate_threshold=-3., instrument_pair_max_pairs=512)
    settings = replace(resolve_inference_settings(config, training, defaults), allowed_instrument_ids=None)
    wave = load_audio(audio, target_sample_rate=config.sample_rate)
    expected_samples = (len(pcm) * config.sample_rate + original_rate - 1) // original_rate
    if tuple(wave.shape) != (2, expected_samples) or not torch.isfinite(wave).all():
        raise ValueError('Unexpected upstream resampling/channel clock')
    identity = dict(source_commit=SOURCE_COMMIT, source_files=sources, runner_sha256=sha(code_source("scripts/transcribe_instruments.py")),
        shared_adapter_sha256=sha(code_source("scripts/research_tsumugi_transcription.py")),
        runtime_code_sha256=runtime_identity(code_source('')),
        model_revision=MODEL_REVISION, model=model_name, model_sha256=MODELS[model_name],
        audio=str(audio), audio_sha256=sha(audio), original_sample_rate=original_rate,
        original_samples=len(pcm), original_channels=pcm.shape[1],
        pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
        resampled_samples=wave.shape[-1], resampled_sample_rate=config.sample_rate,
        resampled_pcm_sha256=hashlib.sha256(wave.numpy().astype('<f4').tobytes()).hexdigest(),
        resampler='torchaudio.functional.resample/default', normalization=False,
        mono_policy='duplicate_to_stereo', inference_scope='full_original_audio',
        intra_threads=2, inter_threads=1, amp=False, compile=False, seed=42,
        stem_filter='all', evidence_revision=2, settings=asdict(settings), model_config=asdict(config),
        versions=runtime_versions(), weight_load=asdict(loaded), maximum_input_seconds=1200)
    output.mkdir(parents=True)
    def write(name, data):
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
    write('identity.json', identity)
    windows = []
    def record_window(stitcher, start):
        for notes in stitcher.notes_by_pair.values():
            for note in notes:
                constructions[object_rows[id(note)]]['v1_slot_index'] = note.slot_index
        windows.append(dict(window_start_sample=start,
            instrument_id_semantics='v1_stitch_placeholder_use_instrument_candidates_for_predictions',
            tracks={str(track):[asdict(n) for n in notes] for track, notes in stitcher.notes_by_pair.items()}))
        print(json.dumps(dict(stage='window', progress=min(94,5+round(89*start/max(1,wave.shape[-1]))),
            message='逐窗分析起音与音尾', start_sample=start, windows=len(windows)),ensure_ascii=True), flush=True)
    print(json.dumps(dict(stage='loaded', model=model_name, duration_seconds=duration)), flush=True)
    started = time.monotonic()
    with trace_interval_construction(WindowNoteStitcher) as (constructions, object_rows):
        with torch.inference_mode():
            decoded, stats = decode_notes(model, config, wave, instrument_filter_id=None,
                device=torch.device('cpu'), amp_enabled=False, amp_dtype=torch.float32, settings=settings,
                velocity=100, on_window_consumed=record_window)
    if len(constructions) != stats['decoded_interval_count'] or any(
            row['constructed_note'] is not None and (type(row['v1_slot_index']) is not int
            or not 0 <= row['v1_slot_index'] < config.num_pitch_slots) for row in constructions):
        raise ValueError('Construction trace lost a decoded interval or pitch slot')
    # Use the same JSON representation as saved research runs before replaying
    # the candidate bridge (instrument_candidates is a tuple upstream).
    raw = plain([asdict(note) for note in decoded])
    events = convert_notes(raw, sample_rate=config.sample_rate, total_samples=wave.shape[-1], classes=INSTRUMENT_CLASSES)
    for name, data in (('raw-notes.json',raw),('window-notes.json',windows),
                       ('interval-constructions.json',constructions),('notes.json',events)):
        write(name, data)
    report = dict(identity=identity, context_start_seconds=0., context_end_seconds=duration,
        elapsed_seconds=time.monotonic()-started, notes=len(events), stats=stats,
        instrument_classes=dict(Counter(n['instrument_class'] for n in events)),
        release_reasons=dict(Counter(n['release_reason'] for n in events)),
        **{key:sha(output/name) for name,key in (
            ('notes.json','notes_sha256'),('raw-notes.json','raw_notes_sha256'),
            ('window-notes.json','window_notes_sha256'),('interval-constructions.json','interval_constructions_sha256'))},
        enabled_in_application=False,
        limitations=['Stitched tails require source tracing before any sustained-expression hypothesis.',
            'Model classes and unit event weights are not verified timbres or calibrated confidence.',
            'Default stereo mean silence gate can suppress phase-inverted channels.',
            'Full-song execution is not an acceptance of timing or musical LN quality.'])
    write('report.json', report)
    audited = dict(identity=identity, **audit_replay(identity, plain(constructions), plain(windows), plain(raw), stats, WindowNoteStitcher))
    candidates, trace = tsumugi_candidates(audited,events,INSTRUMENT_CLASSES,voice=args.voice,
        allow_predicted_releases=args.allow_predicted_releases)
    write('candidates.json',[n.data() for n in candidates])
    write('candidate-trace.json',trace)
    write('result.json',dict(identity=identity,source_voice=args.voice,model=model_name,
        allow_predicted_releases=args.allow_predicted_releases,candidates=len(candidates),
        unobserved_onsets=sum(t['decision']=='unobserved_onset_not_keypress' for t in trace),
        input_start_active_intervals=sum(t['decision']=='candidate' and t['input_start_active_interval'] for t in trace),
        predicted_release_hypotheses=sum(n.release_reason=='model_predicted_offset' for n in candidates),
        evidence_sha256={name:sha(output/name) for name in ('notes.json','raw-notes.json',
            'window-notes.json','interval-constructions.json','candidates.json','candidate-trace.json','report.json')},
        source_semantics='unit_weight_not_confidence',musical_acceptance=False))
    print(json.dumps(dict(stage='complete', notes=len(events), elapsed_seconds=report['elapsed_seconds'])), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--audio', type=Path, required=True)
    cli.add_argument('--voice', choices=tuple(VOICE_MODELS), required=True)
    cli.add_argument('--assets', type=Path, required=True)
    cli.add_argument('--allow-predicted-releases', action='store_true')
    cli.add_argument('--dependencies', type=Path, default=Path('__no_extra_dependencies__'))
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args())


if __name__ == '__main__':
    from worker_lifetime import watch_parent
    watch_parent()
    main()
