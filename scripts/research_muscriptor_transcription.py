"""Run original MuScriptor CPU code with locally recovered public GGUF weights.

Research only. Preserve complete generated tokens, forced prefixes, decoded
actions, raw notes and all artificial endpoint causes. No MIDI, beat correction,
quantization, program filtering or cleanup is used for the research output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_mega53_separation import save, sha

SOURCE = ROOT / 'test-data/other-instrument-model-survey-2026-11-10/muscriptor'
COMMIT = '8a1ef88672cc23c26619adaff37ec48d82c0b906'
ASSETS = ROOT / 'test-data/muscriptor-cpp-review-v1'
GGUF_SHA = '925f55af65a20ebc4f8b45ceaf095a12b72493d436cb112623cd0041a1af23d4'
RECOVERED_SHA = '2111582d9f3df37619dfbc8b4f10071b4621f143bfa3f664f0efe7b293a208e8'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_source():
    git = lambda *args: subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()
    assert git('rev-parse', 'HEAD') == COMMIT and not git('diff', 'HEAD', '--name-only')
    return {str(p.relative_to(SOURCE)).replace('\\', '/'): sha(p)
            for p in (SOURCE / 'muscriptor').rglob('*.py')}


def replay_tokens(stream, tokenizer, actual_events):
    """Observe the unmodified upstream state machine and label closing causes."""
    from muscriptor.events import (OpenNoteTracker, ChunkBoundary, ProgressEvent,
        _StartNote, _EndNote, _DrumHit, decode_model_tokens)
    from muscriptor.transcription_model import _build_instrument_for_program
    tracker = OpenNoteTracker(tokenizer._vocab, tokenizer.frame_rate)
    instrument_for_program = _build_instrument_for_program(tokenizer)
    action_rows, notes, open_notes = [], [], {}
    next_id = 0

    def apply(actions, origin, item_index):
        nonlocal next_id
        action_rows.append(dict(item_index=item_index, origin=origin,
                               actions=[dict(type=type(a).__name__, **asdict(a)) for a in actions]))
        for action in actions:
            if isinstance(action, _StartNote):
                key = (action.program, action.pitch)
                assert key not in open_notes
                open_notes[key] = dict(index=next_id, program=action.program, pitch=action.pitch,
                    onset=action.time, is_drum=instrument_for_program(action.program) == 'drums',
                    onset_stream_index=item_index)
                next_id += 1
            elif isinstance(action, _EndNote):
                note = open_notes.pop((action.program, action.pitch))
                notes.append(note | dict(offset=action.time, endpoint_cause=origin,
                    offset_stream_index=item_index, authored_ln_verified=False))
            else:
                assert isinstance(action, _DrumHit)
                notes.append(dict(index=next_id, program=128, pitch=action.pitch,
                    onset=action.time, offset=action.time+.01, is_drum=True,
                    onset_stream_index=item_index, offset_stream_index=item_index,
                    endpoint_cause='drum_minimum_duration', authored_ln_verified=False))
                next_id += 1

    for index, item in enumerate(stream):
        if isinstance(item, ProgressEvent):
            continue
        if isinstance(item, ChunkBoundary):
            origin = 'malformed_previous_prologue_boundary'
        else:
            event = tokenizer._vocab[item]
            if tracker._in_prologue:
                origin = 'missing_tie_at_boundary' if event.type == 'tie' else 'malformed_prologue'
            elif event.type == 'pitch':
                origin = 'same_pitch_reattack' if tracker._velocity and tracker._velocity > 0 else 'predicted_note_off'
            else:
                origin = 'non_closing_token'
        apply(tracker.feed(item), origin, index)
    final_cause = 'malformed_final_prologue_boundary' if tracker._chunk_started and tracker._in_prologue else 'input_end_minimum_duration'
    apply(tracker.finish(), final_cause, len(stream))
    assert not open_notes
    notes.sort(key=lambda n: n['index'])
    # Check the independently assembled program-aware notes against the actual
    # public start/end stream. The latter lacks numeric program IDs.
    starts, public = {}, []
    for event in actual_events:
        if type(event).__name__ == 'ProgressEvent':
            continue
        if type(event).__name__ == 'NoteStartEvent':
            starts[event.index] = event
        else:
            start = starts.pop(event.start_event_index)
            public.append(dict(index=start.index, onset=start.start_time, offset=event.end_time,
                               pitch=start.pitch, is_drum=start.instrument == 'drums'))
    assert not starts
    public.sort(key=lambda n: n['index'])
    assert public == [{k:n[k] for k in ('index', 'onset', 'offset', 'pitch', 'is_drum')} for n in notes]
    assert all(all(math.isfinite(n[k]) for k in ('onset', 'offset', 'pitch', 'program')) for n in notes)
    return notes, action_rows


def run(audio, weights, output):
    import numpy as np
    from scipy.io import wavfile
    assert not output.exists(), 'Use a fresh output directory'
    assert sha(ASSETS / 'muscriptor-small-f16.gguf') == GGUF_SHA
    assert sha(weights) == RECOVERED_SHA
    source_files = verify_source()
    config = read(weights.parent / 'config.json')
    assert {k:config[k] for k in ('dim', 'num_heads', 'num_layers', 'card')} == dict(dim=768, num_heads=12, num_layers=14, card=1393)
    rate, wave = wavfile.read(audio)
    assert rate == 16000 and wave.ndim == 1 and wave.dtype in (np.float32, np.int16)
    pcm = wave if wave.dtype == np.float32 else wave.astype(np.float32)/32768.
    assert np.isfinite(pcm).all() and len(pcm) > 0
    identity = dict(source_commit=COMMIT, source_files=source_files, audio_sha256=sha(audio),
        pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(), samples=len(pcm),
        sample_rate=rate, channels=1, normalization=False, weights_sha256=sha(weights),
        recovered_config_sha256=sha(weights.parent/'config.json'), source_gguf_sha256=GGUF_SHA,
        upstream_original_fp32_weights=False, transformer_weight_source='GGUF F16 cast, loaded as CPU float32',
        runner_sha256=sha(__file__), device='cpu', intra_threads=2, inter_threads=1,
        seed=42, use_sampling=False, beam_size=1, cfg_coef=1., prelude_forcing=True,
        batch_size=1, max_gen_len=2000, no_eos_is_ok=True, instruments=None,
        source_code_unmodified=True, monkeypatch='read-only generator observers on this instance',
        beat_correction=False, midi_cleanup=False, enabled_in_application=False)
    save(output / 'plan.json', identity)
    for folder in (ROOT / 'test-data/rosvot-runtime', ASSETS/'deps', SOURCE):
        sys.path.insert(0, str(folder))
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(42)
    from safetensors.torch import load_file
    from muscriptor.transcription_model import TranscriptionModel, _build_model, _ModelConfig, _remap_single_codebook_keys
    from muscriptor.tokenizer.mt3 import MT3Tokenizer
    from muscriptor.events import ChunkBoundary, ProgressEvent
    net = _build_model(torch.device('cpu'), _ModelConfig(**{k:config[k] for k in ('dim', 'num_heads', 'num_layers', 'card')})).eval()
    state = _remap_single_codebook_keys(load_file(weights, device='cpu'))
    loaded = net.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert len(state) == len(net.state_dict()) == 122
    for name, actual in net.state_dict().items():
        assert torch.equal(actual, state[name].to(dtype=actual.dtype)), name
    model_identity = dict(identity, strict_load=True, tensors=122,
        parameter_count=sum(p.numel() for p in net.parameters()),
        dtype=str(next(net.parameters()).dtype),
        versions={p:importlib.metadata.version(p) for p in ('torch', 'numpy', 'einops', 'safetensors', 'huggingface-hub', 'mido')})
    save(output / 'model-identity.json', model_identity)
    del state
    tokenizer = MT3Tokenizer(instrument_vocabulary='MT3_FULL_PLUS', max_shift_steps=1001)
    transcriber = TranscriptionModel(net, tokenizer, torch.device('cpu'))
    original_generate, original_stream = net.generate, transcriber._generate_token_stream
    calls, stream = [], []

    def observed_generate(*args, **kwargs):
        assert kwargs['max_gen_len'] == 2000 and kwargs['use_sampling'] is False
        assert kwargs['beam_size'] == 1 and kwargs['forbidden_tokens'] is None
        assert len(kwargs['conditions']) == 1
        condition = kwargs['conditions'][0]
        assert condition.text == {'instrument_group': None, 'dataset_name': None}
        condition_pcm = condition.wav['self_wav'].wav.detach().cpu().numpy()
        assert condition_pcm.shape == (1, 1, 80000)
        call = dict(index=len(calls), prompt=None if kwargs['prompt'] is None else kwargs['prompt'].tolist(),
            condition_pcm_sha256=hashlib.sha256(condition_pcm.tobytes()).hexdigest(),
            condition_samples=80000, tokens=[], eos_id=tokenizer.eos_id, completed=False)
        calls.append(call)
        started = time.monotonic()
        for value in original_generate(*args, **kwargs):
            assert value.shape == (1,)
            call['tokens'].append(int(value.item()))
            yield value
        call.update(seconds=time.monotonic()-started, completed=True,
                    eos_observed=tokenizer.eos_id in call['tokens'])
        save(output / 'chunks' / f'{call["index"]:04}.json', call)
        print(json.dumps(dict(stage='chunk', index=call['index'], seconds=call['seconds'],
                              tokens=len(call['tokens']), eos=call['eos_observed'])), flush=True)

    def observed_stream(*args, **kwargs):
        for item in original_stream(*args, **kwargs):
            assert isinstance(item, (int, ChunkBoundary, ProgressEvent))
            stream.append(item)
            yield item

    net.generate = observed_generate
    transcriber._generate_token_stream = observed_stream
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as warning_rows, torch.inference_mode():
        warnings.simplefilter('always')
        events = list(transcriber.transcribe((torch.from_numpy(pcm.copy()), rate),
            use_sampling=False, cfg_coef=1., instruments=None, batch_size=1,
            no_eos_is_ok=True, beam_size=1, prelude_forcing=True))
    elapsed = time.monotonic()-started
    assert len(calls) == math.ceil(len(pcm)/80000) and all(c['completed'] for c in calls)
    boundaries = [x for x in stream if isinstance(x, ChunkBoundary)]
    assert [x.seek_time for x in boundaries] == [i*5 for i in range(len(calls))]
    stream_tokens = []
    for item in stream:
        if isinstance(item, ChunkBoundary):
            stream_tokens.append([])
        elif isinstance(item, int):
            stream_tokens[-1].append(item)
    for index, call in enumerate(calls):
        raw = call['tokens']; end = raw.index(tokenizer.eos_id) if tokenizer.eos_id in raw else len(raw)
        assert raw[:end] == stream_tokens[index]
        input_window = pcm[index*80000:(index+1)*80000]
        input_window = np.pad(input_window, (0, 80000-len(input_window)))
        assert hashlib.sha256(input_window.tobytes()).hexdigest() == call['condition_pcm_sha256']
    save(output / 'token-stream.json', [dict(type='token', value=x) if isinstance(x, int)
         else dict(type=type(x).__name__, **asdict(x)) for x in stream])
    save(output / 'events.json', [dict(type=type(e).__name__, **asdict(e)) for e in events])
    notes, actions = replay_tokens(stream, tokenizer, events)
    save(output / 'notes.json', notes)
    save(output / 'actions.json', actions)
    save(output / 'vocabulary.json', [asdict(x) for x in tokenizer._vocab])
    artifacts = {name:sha(output/name) for name in ('token-stream.json', 'events.json', 'notes.json', 'actions.json', 'vocabulary.json')}
    assert sha(__file__) == identity['runner_sha256'] and verify_source() == source_files
    save(output / 'report.json', dict(identity=model_identity, seconds=elapsed, notes=len(notes),
        neural_chunks=len(calls), missing_eos=sum(not c['eos_observed'] for c in calls),
        chunk_reports=[sha(output / 'chunks' / f'{c["index"]:04}.json') for c in calls],
        artifacts=artifacts, warnings=[str(w.message) for w in warning_rows],
        invalid_intervals=sum(n['offset'] <= n['onset'] for n in notes),
        onsets_outside_audio=sum(not 0 <= n['onset'] < len(pcm)/rate for n in notes),
        tails_at_or_after_audio_end=sum(n['offset'] >= len(pcm)/rate for n in notes),
        endpoint_causes=dict(Counter(n['endpoint_cause'] for n in notes)),
        context_start_seconds=0, context_end_seconds=len(pcm)/rate,
        musical_acceptance=False, default_changed=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--audio', type=Path, required=True)
    cli.add_argument('--weights', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    run(args.audio, args.weights, args.output)
