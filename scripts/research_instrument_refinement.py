"""Pinned Tsumugi timbre grouping over saved audio-note predictions.

Research only. No MIDI round trip, timing changes, chart input or new notes.
Run only after the existing serial neural queue has completed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_tsumugi_transcription import (
    MODEL_REVISION, SOURCE_COMMIT, convert_notes, runtime_versions, sha, verify_source,
)

CHECKPOINT_SHA256 = '75b2846c492fbb7e6e36336c4eb0fa6ac48a9ec8f6de1c6a85f1c57b8e9639b9'
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))


def query_arrays(events, duration):
    """Preserve raw indices, chords, short notes and repeats; exclude only drums."""
    import numpy as np
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid input duration')
    for n in events:
        if (not all(np.isfinite(n[k]) for k in ('onset', 'offset'))
                or not 0 <= n['onset'] < n['offset'] <= duration
                or type(n['pitch']) is not int or not 0 <= n['pitch'] <= 127
                or type(n['is_drum']) is not bool):
            raise ValueError('Invalid original note; do not silently remove or repair')
    indices = sorted((i for i, n in enumerate(events) if not n['is_drum']),
                     key=lambda i: (events[i]['onset'], events[i]['pitch'], events[i]['offset'], i))
    starts = np.asarray([events[i]['onset'] for i in indices], dtype=np.float32)
    ends = np.asarray([events[i]['offset'] for i in indices], dtype=np.float32)
    if np.any(ends <= starts):
        raise ValueError('Query float32 conversion collapsed an original interval')
    return dict(original_indices=np.asarray(indices, dtype=np.int64), start_seconds=starts,
                end_seconds=ends, pitch=np.asarray([events[i]['pitch'] for i in indices], dtype=np.int64))


def observation_counts(starts, window_starts, sample_rate, window_seconds=8.):
    import numpy as np
    counts = np.zeros(len(starts), dtype=np.int64)
    for sample in window_starts:
        begin = sample / sample_rate
        counts += (starts >= begin) & (starts < begin + window_seconds)
    return counts


def run(args):
    source, output, folder = args.source.resolve(), args.output.resolve(), args.input.resolve()
    if output.exists():
        raise ValueError('Preserve previous evidence; choose a new directory')
    sources = verify_source(source)
    if sha(args.checkpoint) != CHECKPOINT_SHA256 or args.checkpoint.stat().st_size != 56225305:
        raise ValueError('Refinement checkpoint identity differs')
    report = read(folder/'report.json')
    identity = report['identity']
    if identity['source_commit'] != SOURCE_COMMIT or identity['model_revision'] != MODEL_REVISION:
        raise ValueError('Original transcription source differs')
    if identity['source_files'] != sources:
        raise ValueError('Original transcription source files differ')
    audio = Path(identity['audio'])
    if sha(audio) != identity['audio_sha256']:
        raise ValueError('Original audio changed')
    for filename, key in (('notes.json','notes_sha256'),('raw-notes.json','raw_notes_sha256')):
        if sha(folder/filename) != report[key]:
            raise ValueError('Original note evidence changed')
    events, raw = read(folder/'notes.json'), read(folder/'raw-notes.json')
    if args.dependencies.exists():
        sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from instrument_agnostic_amt.amt.inference.audio import load_audio
    from instrument_agnostic_amt.instrument_refinement.data.midi import RefinementNoteTable
    from instrument_agnostic_amt.instrument_refinement.modeling.model import (
        InstrumentRefinementConfig, InstrumentRefinementModel, _coerce_config_values,
    )
    from instrument_agnostic_amt.instrument_refinement.inference.refine import (
        _allowed_class_ids, _scan_windows, _cluster_reports,
    )
    from instrument_agnostic_amt.instrument_refinement.inference.aggregation import cluster_note_embeddings
    from instrument_agnostic_amt.taxonomy.instrument_classes import INSTRUMENT_CLASSES

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'instrument_refinement':
        raise ValueError('Wrong checkpoint task')
    config = InstrumentRefinementConfig(**_coerce_config_values(checkpoint['model_config']))
    model = InstrumentRefinementModel(config).eval()
    loaded = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Incomplete refinement weight load')
    if events != convert_notes(raw, sample_rate=identity['resampled_sample_rate'],
            total_samples=identity['resampled_samples'], classes=INSTRUMENT_CLASSES):
        raise ValueError('Original raw/events conversion differs')
    duration = identity['original_samples']/identity['original_sample_rate']
    query = query_arrays(events, duration)
    if np.any((query['pitch'] < config.pitch_min) | (query['pitch'] > config.pitch_max)):
        raise ValueError('Original pitch outside refinement support; no implicit clipping')
    table = RefinementNoteTable(start_seconds=query['start_seconds'], end_seconds=query['end_seconds'],
        pitch=query['pitch'], prior_class_id=np.full(len(query['pitch']), -1, dtype=np.int64),
        instrument_index=np.zeros(len(query['pitch']), dtype=np.int64), note_index=query['original_indices'])
    wave = load_audio(audio, target_sample_rate=config.sample_rate)
    expected_samples = (identity['original_samples']*config.sample_rate+identity['original_sample_rate']-1)//identity['original_sample_rate']
    pcm_hash = hashlib.sha256(wave.numpy().astype('<f4').tobytes()).hexdigest()
    if (tuple(wave.shape) != (2, expected_samples) or not torch.isfinite(wave).all()
            or config.sample_rate != identity['resampled_sample_rate']
            or pcm_hash != identity['resampled_pcm_sha256']):
        raise ValueError('Refinement and transcription PCM clocks or gains differ')
    allowed = _allowed_class_ids(None, use_stem_mask=False)
    output.mkdir(parents=True)
    def write(name, value):
        with (output/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    record = dict(source_commit=SOURCE_COMMIT, source_files=sources, model_revision=MODEL_REVISION,
        checkpoint_sha256=CHECKPOINT_SHA256, model_config=asdict(config), runner_sha256=sha(__file__),
        original_report_sha256=sha(folder/'report.json'), original_identity=identity,
        versions=runtime_versions(), resampled_pcm_sha256=pcm_hash, settings=dict(
            window_seconds=8., stride_seconds=4., window_batch_size=1, cluster_distance=.25,
            min_cluster_notes=2, global_logit_weight=0., stem_name=None, use_stem_mask=False,
            allowed_classes=list(allowed), intra_threads=2, inter_threads=1, seed=42,
            amp=False, normalization=False, prior_classes_used=False))
    write('identity.json', record)
    np.savez_compressed(output/'queries.npz', **query)
    started = time.monotonic()
    print(json.dumps(dict(stage='refinement_loaded', queries=table.note_count, duration_seconds=duration)), flush=True)
    with torch.inference_mode():
        scan = _scan_windows(model, config, wave, table, stem_name=None, allowed=allowed,
            window_seconds=8., stride_seconds=4., window_batch_size=1, device=torch.device('cpu'),
            disable_tqdm=True, amp_enabled=False, amp_dtype=torch.float32)
    counts = observation_counts(query['start_seconds'], scan.starts, config.sample_rate)
    if np.any(counts == 0):
        raise ValueError('Unobserved note used global fallback; preserve partial evidence for review')
    if any(not np.isfinite(value).all() for value in (scan.note_logits, scan.note_embeddings)):
        raise ValueError('Nonfinite model evidence')
    clusters = cluster_note_embeddings(scan.note_embeddings, distance_threshold=.25, min_cluster_notes=2)
    predicted, cluster_reports = _cluster_reports(table, scan, clusters, allowed=allowed,
        global_logit_weight=0., top_k=3)
    np.savez_compressed(output/'model-evidence.npz', window_starts=scan.starts,
        window_logits=scan.window_logits, window_note_counts=scan.window_note_counts,
        global_logits=scan.global_logits, note_logits=scan.note_logits,
        note_embeddings=scan.note_embeddings, observation_counts=counts, cluster_ids=clusters)
    by_index = {int(original):i for i,original in enumerate(query['original_indices'])}
    decisions = []
    for index, event in enumerate(events):
        item = dict(original_index=index, original_event=event)
        if index not in by_index:
            item.update(decision='unsupported_drum_prediction', refined_class=None)
        else:
            i = by_index[index]
            class_id = int(predicted[i])
            item.update(decision='classified', query_index=i, observations=int(counts[i]),
                query_start_seconds=float(table.start_seconds[i]), query_end_seconds=float(table.end_seconds[i]),
                cluster_id=int(clusters[i]), refined_class=INSTRUMENT_CLASSES[class_id],
                refined_class_id=class_id, ungrouped_class=INSTRUMENT_CLASSES[int(scan.note_logits[i].argmax())])
        decisions.append(item)
    write('decisions.json', decisions)
    write('clusters.json', cluster_reports)
    write('report.json', dict(identity=record, elapsed_seconds=time.monotonic()-started,
        original_notes=len(events), queried_notes=table.note_count, classified_notes=len(by_index),
        unsupported_drums=len(events)-len(by_index), clusters=len(cluster_reports),
        classes=dict(Counter(d['refined_class'] for d in decisions if d['decision']=='classified')),
        evidence_sha256={name:sha(output/name) for name in ('queries.npz','model-evidence.npz','decisions.json','clusters.json')},
        original_notes_modified=False, neural_inference=True, enabled_in_application=False,
        musical_acceptance=False, limits=[
            'Clustering does not add missing notes, repair timing, or produce independent waveforms.',
            'Existing predicted intervals are queries, not reference-label guidance.',
            'Fewer class switches do not prove correct instrument identity.',
            'Cluster candidate softmax values are not calibrated confidence.']))
    print(json.dumps(dict(stage='refinement_complete', notes=table.note_count, clusters=len(cluster_reports))),flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--source', type=Path, default=Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi'))
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/rosvot-runtime'))
    cli.add_argument('--checkpoint', type=Path, default=Path('test-data/tsumugi-instrument-refinement-v1/best_instrument_refinement.pth'))
    run(cli.parse_args())


if __name__ == '__main__':
    main()
