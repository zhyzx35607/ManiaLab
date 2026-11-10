"""Recover original AMT class logits on saved intervals, without decoding notes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_tsumugi_releases import audit_replay, load_official, read_json as read
from scripts.check_harmonica_transcription import BASELINE, BASELINE_SHA, MANIFEST, MANIFEST_SHA
from scripts.check_mega53_separation import save, sha
from scripts.research_tsumugi_transcription import MODELS, verify_source, runtime_versions
from scripts.tsumugi_instrument_scores import rebuild_intervals, note_decisions

SOURCE = ROOT/'test-data/other-instrument-model-survey-2026-11-10/tsumugi'
CODE = ('scripts/recover_tsumugi_instrument_scores.py', 'scripts/tsumugi_instrument_scores.py',
        'scripts/audit_tsumugi_releases.py', 'docs/instrument-score-evidence-protocol.md')


def prepare(output):
    if output.exists():
        raise ValueError('Preserve earlier evidence; use a new directory')
    assert sha(BASELINE/'comparison.json') == BASELINE_SHA and sha(MANIFEST) == MANIFEST_SHA
    baseline, manifest = read(BASELINE/'comparison.json'), read(MANIFEST)
    assert len(baseline['rows']) == len(manifest['cases']) == 35
    source_files = verify_source(SOURCE)
    checkpoint = SOURCE/'checkpoints/best_model_other_v1_5.pth'
    assert sha(checkpoint) == MODELS['other_v1_5']
    cases, inputs = [], {}
    for case, old in zip(manifest['cases'], baseline['rows']):
        assert case['name'] == old['name']
        folder = BASELINE/case['name']
        report = read(folder/'report.json')
        assert sha(folder/'report.json') == old['report_sha256']
        identity = report['identity']
        assert identity['source_files'] == source_files and identity['model'] == 'other_v1_5'
        assert identity['model_sha256'] == MODELS['other_v1_5']
        assert all(identity[k] == v for k, v in dict(stem_filter='all', amp=False, compile=False,
            seed=42, intra_threads=2, inter_threads=1, normalization=False, evidence_revision=2).items())
        assert all(identity['settings'][k] == v for k, v in dict(window_ms=8000., stride_ms=4000.,
            window_batch_size=1, allowed_instrument_ids=None, instrument_probability_mode='sigmoid',
            use_boundary_head=True).items())
        paths = {str(folder/'report.json'): old['report_sha256'], identity['audio']: identity['audio_sha256']}
        paths.update({str(folder/(key.replace('_', '-')+'.json')): report[key+'_sha256']
                      for key in ('notes', 'raw_notes', 'window_notes', 'interval_constructions')})
        for path, digest in paths.items():
            assert sha(path) == digest, path
        inputs.update(paths)
        cases.append(dict(name=case['name'], kind=case['kind'], instrument=case.get('instrument'),
                          original=str(folder), report_sha256=old['report_sha256']))
    plan = dict(revision=1, baseline_sha256=BASELINE_SHA, manifest_sha256=MANIFEST_SHA,
                source_files=source_files, checkpoint_sha256=sha(checkpoint), inputs=inputs, cases=cases,
                code_sha256={name: sha(ROOT/name) for name in CODE},
                original_intervals_only=True, labels_read=False, enabled_in_application=False)
    save(output/'plan.json', plan)
    for name in CODE:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    return plan, checkpoint


def run(output):
    plan, checkpoint_path = prepare(output)
    sys.path.insert(0, str(ROOT/'test-data/rosvot-runtime'))
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import soundfile as sf
    import torch
    from instrument_agnostic_amt.amt.inference.audio import load_audio
    from instrument_agnostic_amt.amt.inference.windowed import _slice_window
    from instrument_agnostic_amt.amt.inference.v1_windowed import _decode_instrument_map, _decode_boundary_map
    from instrument_agnostic_amt.amt.modeling.checkpoints import coerce_model_config, extract_model_config, load_compatible_weights
    from instrument_agnostic_amt.amt.modeling.model import AudioSemiCRFTransformer
    from instrument_agnostic_amt.taxonomy.instrument_classes import INSTRUMENT_CLASSES

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = coerce_model_config(extract_model_config(checkpoint), for_inference=True)
    model = AudioSemiCRFTransformer(config).eval()
    loaded = load_compatible_weights(model, checkpoint, prefer_ema=True, require_complete_backbone=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys and not loaded.shape_mismatches
    del checkpoint
    assert config.num_instrument_classes == len(INSTRUMENT_CLASSES) == 36
    assert config.num_pitch_slots == 3 and config.semi_crf_version == 'v1'
    runtime = runtime_versions()
    save(output/'runtime.json', dict(versions=runtime, model_config=asdict(config), weight_load=asdict(loaded),
                                    classes=list(INSTRUMENT_CLASSES), intra_threads=2, inter_threads=1, seed=42,
                                    amp=False, compile=False))
    official, _ = load_official(SOURCE, read(Path(plan['cases'][0]['original'])/'report.json')['identity'])
    completed, started = [], time.monotonic()
    for case_index, case in enumerate(plan['cases']):
        folder, target = Path(case['original']), output/case['name']
        report = read(folder/'report.json')
        identity = report['identity']
        assert identity['model_config'] == json.loads(json.dumps(asdict(config)))
        assert identity['versions'] == runtime
        for path, expected in plan['inputs'].items():
            if Path(path).parent == folder:
                assert sha(path) == expected, path
        audio = Path(identity['audio'])
        assert sha(audio) == identity['audio_sha256']
        pcm, rate = sf.read(audio, dtype='float32', always_2d=True)
        assert (rate, len(pcm), pcm.shape[1]) == (identity['original_sample_rate'],
            identity['original_samples'], identity['original_channels'])
        assert hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == identity['pcm_sha256']
        wave = load_audio(audio, target_sample_rate=config.sample_rate)
        assert tuple(wave.shape) == (2, identity['resampled_samples']) and torch.isfinite(wave).all()
        assert hashlib.sha256(wave.numpy().astype('<f4').tobytes()).hexdigest() == identity['resampled_pcm_sha256']
        constructions, windows, raw = [read(folder/name) for name in
            ('interval-constructions.json', 'window-notes.json', 'raw-notes.json')]
        audit = audit_replay(identity, constructions, windows, raw, report['stats'], official)
        assert len(raw) == len(read(folder/'notes.json'))
        target.mkdir(parents=True)
        logits = np.full((len(constructions), config.num_instrument_classes), np.nan, dtype=np.float32)
        boundaries = np.full((len(constructions), 4), np.nan, dtype=np.float32)
        rankings = np.full(logits.shape, -1, dtype=np.int64)
        entry_rows = np.full((len(constructions), 5), -1, dtype=np.int64)
        boundary_values = np.full((len(constructions), 4), np.nan, dtype=np.float64)
        coverage = np.zeros(len(constructions), dtype=np.int64)
        window_rows, local_started = [], time.monotonic()
        print(json.dumps(dict(stage='case', name=case['name'], index=case_index+1, total=35)), flush=True)
        for snapshot in windows:
            start = snapshot['window_start_sample']
            clip, valid = _slice_window(wave, start_frame=start,
                                       window_audio_frames=report['stats']['window_audio_frames'])
            indices = [i for i, row in enumerate(constructions) if row['arguments']['window_start_frame'] == start]
            assert all(constructions[i]['arguments']['valid_audio_frames'] == valid for i in indices)
            with torch.inference_mode():
                outputs = model(clip.unsqueeze(0), valid_audio_frames=torch.tensor([valid]),
                                include_aux_outputs=False, include_frame_instrument_logits=False)
                valid_frames = int(outputs['frame_valid_mask'].long().sum())
                decoded, expected_entries, keys = rebuild_intervals(constructions, indices,
                    slots=config.num_pitch_slots, classes=config.num_instrument_classes, valid_frames=valid_frames)
                scores, entries = model.predict_interval_instruments(outputs['instrument_features'], decoded,
                                                                      compute_dtype=torch.float32)
                ends, end_entries = model.predict_interval_boundaries(outputs['interval_features'], decoded,
                                                                      compute_dtype=torch.float32)
                assert entries == end_entries == expected_entries
                instruments = _decode_instrument_map(scores, entries, probability_mode='sigmoid',
                                                     allowed_instrument_ids=tuple(range(config.num_instrument_classes)))
                flags = _decode_boundary_map(ends, end_entries)
                assert torch.isfinite(scores).all() and torch.isfinite(ends).all()
                for row_index, entry in enumerate(entries):
                    key = tuple(entry[:3])
                    original_index = keys[key]
                    args = constructions[original_index]['arguments']
                    candidates = list(instruments[key][1])
                    flag = list(flags[key])
                    if candidates != args['instrument_candidates'] or flag[:2] != args['boundary_flag'][:2]:
                        save(target/'reproduction-failure.json', dict(construction=original_index,
                            expected=args, actual_candidates=candidates, actual_flag=flag))
                        raise ValueError('Original class order or boundary booleans did not reproduce')
                    if not np.allclose(flag[2:], args['boundary_flag'][2:], atol=1e-6, rtol=0):
                        save(target/'reproduction-failure.json', dict(construction=original_index,
                            expected=args, actual_flag=flag))
                        raise ValueError('Original boundary fractions did not reproduce')
                    logits[original_index] = scores[row_index].numpy()
                    boundaries[original_index] = ends[row_index].numpy()
                    rankings[original_index] = candidates
                    entry_rows[original_index] = entry
                    boundary_values[original_index] = flag
                    coverage[original_index] += 1
                del outputs, scores, ends
            window_rows.append(dict(start_sample=start, valid_samples=valid, valid_model_frames=valid_frames,
                                    constructions=indices))
        assert np.all(coverage == 1) and np.isfinite(logits).all() and np.isfinite(boundaries).all()
        decisions = note_decisions(audit['notes'], logits, rankings)
        np.savez_compressed(target/'evidence.npz', instrument_logits=logits, boundary_logits=boundaries,
                            rankings=rankings, entries=entry_rows, boundary_values=boundary_values,
                            coverage=coverage)
        save(target/'windows.json', window_rows)
        save(target/'lineage.json', [dict(note=n['note'], sources=n['sources'], onset_sources=n['onset_sources'],
                                         endpoint_sources=n['endpoint_sources']) for n in audit['notes']])
        save(target/'decisions.json', decisions)
        max_error = max((abs(boundary_values[i,j]-row['arguments']['boundary_flag'][j])
                         for i,row in enumerate(constructions) for j in (2,3)), default=0.)
        result = dict(name=case['name'], original_report_sha256=case['report_sha256'],
                      plan_sha256=sha(output/'plan.json'), runtime_sha256=sha(output/'runtime.json'),
                      evidence_sha256={name:sha(target/name) for name in
                          ('evidence.npz', 'windows.json', 'lineage.json', 'decisions.json')},
                      windows=len(windows), constructions=len(constructions), notes=len(raw),
                      changed_classes=sum(d['changed'] for d in decisions),
                      multiple_sources=sum(len(d['sources']) > 1 for d in decisions),
                      disagreeing_sources=sum(not d['sources_agree'] for d in decisions),
                      complete_class_rankings_reproduced=True, boundary_max_abs_error=max_error,
                      labels_read=False, elapsed_seconds=time.monotonic()-local_started)
        save(target/'report.json', result)
        completed.append(dict(name=case['name'], report_sha256=sha(target/'report.json')))
        print(json.dumps(dict(stage='case_complete', **{k:result[k] for k in
            ('name', 'windows', 'constructions', 'notes', 'changed_classes', 'elapsed_seconds')})), flush=True)
    assert all(sha(path) == digest for path, digest in plan['inputs'].items())
    assert all(sha(ROOT/path) == digest for path, digest in plan['code_sha256'].items())
    assert verify_source(SOURCE) == plan['source_files'] and sha(checkpoint_path) == plan['checkpoint_sha256']
    save(output/'inference-completed.json', dict(plan_sha256=sha(output/'plan.json'),
        runtime_sha256=sha(output/'runtime.json'), cases=completed, labels_read=False,
        elapsed_seconds=time.monotonic()-started, enabled_in_application=False))
    print(json.dumps(dict(stage='complete', output=str(output), cases=len(completed))), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output.resolve())
