"""Inspect per-note dynamics without replacing confidence or selecting notes.

Pinned CPU experiment; preserves raw notes, explicit program assumptions,
unrounded outputs, silence controls and original PCM. No application changes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_tsumugi_transcription import (
    SOURCE_COMMIT, MODEL_REVISION, convert_notes, runtime_versions, sha, verify_source,
)

CHECKPOINT_SHA256 = 'df8b0877c8b71cb5c3c224d69d386d67520417ecbc03728f02a2951abc4d2ae1'
MODEL_STEMS = {'bass_v2': 'bass', 'guitar_v1_5': 'guitar',
               'other_v1_5': 'other', 'vocal_harmony_v1_5': 'vocals'}
GAINS = (-12., -6., 0., 6., None)
PAIRS = ((60, 67), (48, 69), (55, 72), (62, 73))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def pcm_sha(pcm):
    return hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()


def program_hypotheses(event, labels, extras):
    """Do not let the upstream embedding silently turn unknown -1 into piano."""
    name = event['instrument_class']
    if event['is_drum']:
        return [(0, 'drum_channel_program_placeholder')]
    if name in labels:
        programs = sorted({m['program_number'] for m in labels[name]['members']
                           if m.get('program_number') is not None})
        basis = 'assumed_member_of_predicted_broad_class'
    elif name in extras:
        programs, basis = [extras[name]], 'upstream_non_gm_representative_assumption'
    else:
        raise ValueError('No explicit program hypothesis for ' + str(name))
    if not programs or any(type(p) is not int or not 0 <= p <= 127 for p in programs):
        raise ValueError('Invalid program hypothesis')
    return [(p, basis) for p in programs]


def validate_queries(queries, duration, pitch_min=21, pitch_max=108):
    import numpy as np
    if not math.isfinite(duration) or duration <= 0 or not queries:
        raise ValueError('Invalid duration or empty query set')
    for q in queries:
        if (not all(math.isfinite(q[k]) for k in ('onset', 'offset'))
                or not 0 <= q['onset'] < q['offset'] <= duration
                or type(q['pitch']) is not int or not pitch_min <= q['pitch'] <= pitch_max
                or type(q['program']) is not int or not 0 <= q['program'] <= 127
                or type(q['is_drum']) is not bool
                or not q.get('program_basis')):
            raise ValueError('Unsupported query; no implicit clipping or unknown program')
    starts = np.array([q['onset'] for q in queries], dtype=np.float32)
    ends = np.array([q['offset'] for q in queries], dtype=np.float32)
    if np.any(ends <= starts) or np.any(starts >= duration):
        raise ValueError('Float32 query clock lost an interval')


def synthetic_cases(sample_rate):
    import numpy as np
    duration, onset, offset = 4., .5, 3.5
    samples = int(duration * sample_rate)
    begin, end = int(onset * sample_rate), int(offset * sample_rate)
    t = np.arange(end - begin, dtype=np.float64) / sample_rate
    envelope = np.minimum(1., t / .02) * np.minimum(1., (t[-1] - t) / .05)
    for pitches in PAIRS:
        atoms = []
        for pitch in pitches:
            f = 440. * 2 ** ((pitch - 69) / 12)
            atom = sum(np.sin(2 * np.pi * h * f * t + h * .13) / h
                       for h in (1, 2, 3, 4)) * envelope
            atom /= np.max(np.abs(atom))
            atoms.append(atom)
        for relation, amps in (('lower_stronger', (.25, .0625)),
                               ('higher_stronger', (.0625, .25)),
                               ('equal', (.125, .125))):
            components = []
            queries = []
            for i, (pitch, amplitude) in enumerate(zip(pitches, amps)):
                component = np.zeros(samples, dtype=np.float32)
                component[begin:end] = atoms[i] * amplitude
                components.append(component)
                for program in (0, 40):
                    queries.append(dict(original_index=i, onset=onset, offset=offset,
                        pitch=pitch, program=program, is_drum=False,
                        program_basis='controlled_program_assumption_not_instrument_truth'))
            wave = np.stack([sum(components)] * 2)
            yield dict(name=f'synthetic-{pitches[0]}-{pitches[1]}-{relation}', stem='other',
                queries=queries, wave=wave, identity=dict(kind='synthetic_control',
                pitches=pitches, relation=relation, amplitude=amps,
                component_rms=[float(np.sqrt(np.mean(c[begin:end] ** 2))) for c in components],
                component_sha256=[pcm_sha(c) for c in components],
                waveform='four_harmonics_inverse_amplitude_phase_h_times_0.13',
                attack_seconds=.02, release_seconds=.05), gains=(0.,))


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory')
    sources = verify_source(source)
    if sha(args.checkpoint) != CHECKPOINT_SHA256 or args.checkpoint.stat().st_size != 54996249:
        raise ValueError('Velocity checkpoint identity differs')
    if args.dependencies.exists():
        sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from instrument_agnostic_amt.amt.inference.audio import load_audio
    from instrument_agnostic_amt.velocity.modeling.model import VelocityModelConfig, VelocityPredictionModel
    from instrument_agnostic_amt.velocity.stems import STEM_CLASS_BY_NAME
    from instrument_agnostic_amt.taxonomy.instrument_classes import (
        INSTRUMENT_CLASSES, EXTRA_INSTRUMENT_CLASS_PROGRAMS,
    )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config_values = dict(checkpoint['model_config'])
    for k in ('harmonics', 'local_frame_offsets'):
        config_values[k] = tuple(config_values[k])
    config = VelocityModelConfig(**config_values)
    if config.predict_stem_gain or not config.use_absolute_velocity_energy:
        raise ValueError('Expected pinned absolute-energy velocity-only model')
    model = VelocityPredictionModel(config).eval()
    loaded = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Incomplete velocity model load')
    labels = read(source/'instrument_agnostic_amt/taxonomy/instrument_merge.json')['labels']
    cases = []
    unchanged = {str(args.checkpoint): CHECKPOINT_SHA256}
    for name, stem in MODEL_STEMS.items():
        folder = args.smoke/'outputs-traced'/name
        report = read(folder/'report.json')
        identity = report['identity']
        if (identity['source_commit'] != SOURCE_COMMIT or identity['source_files'] != sources
                or identity['model_revision'] != MODEL_REVISION):
            raise ValueError('Original transcription source differs')
        for filename, key in (('notes.json', 'notes_sha256'), ('raw-notes.json', 'raw_notes_sha256')):
            if sha(folder/filename) != report[key]:
                raise ValueError('Original note evidence changed')
            unchanged[str(folder/filename)] = report[key]
        unchanged[str(folder/'report.json')] = sha(folder/'report.json')
        audio = Path(identity['audio'])
        if sha(audio) != identity['audio_sha256']:
            raise ValueError('Original audio changed')
        unchanged[str(audio)] = identity['audio_sha256']
        events, raw = read(folder/'notes.json'), read(folder/'raw-notes.json')
        if events != convert_notes(raw, sample_rate=identity['resampled_sample_rate'],
                total_samples=identity['resampled_samples'], classes=INSTRUMENT_CLASSES):
            raise ValueError('Raw and derived notes differ')
        wave = load_audio(audio, target_sample_rate=config.sample_rate).numpy()
        if (config.sample_rate != identity['resampled_sample_rate']
                or tuple(wave.shape) != (2, identity['resampled_samples'])
                or not np.isfinite(wave).all()
                or pcm_sha(wave) != identity['resampled_pcm_sha256']):
            raise ValueError('Original PCM clock or level differs')
        queries = [dict(original_index=i, onset=n['onset'], offset=n['offset'], pitch=n['pitch'],
            program=p, is_drum=n['is_drum'], instrument_class=n['instrument_class'], program_basis=basis)
            for i, n in enumerate(events) for p, basis in program_hypotheses(n, labels, EXTRA_INSTRUMENT_CLASS_PROGRAMS)]
        cases.append(dict(name=name, stem=stem, wave=wave, queries=queries,
            identity=dict(kind='frozen_transcription', original_identity=identity,
                          original_report_sha256=sha(folder/'report.json'), original_notes=len(events)), gains=GAINS))
    cases.extend(synthetic_cases(config.sample_rate))
    for case in cases:
        validate_queries(case['queries'], case['wave'].shape[-1]/config.sample_rate,
                         config.pitch_min, config.pitch_max)
    output.mkdir(parents=True)
    write(output/'identity.json', dict(source_commit=SOURCE_COMMIT, source_files=sources,
        model_revision=MODEL_REVISION, checkpoint_sha256=CHECKPOINT_SHA256,
        model_config=asdict(config), weight_load=dict(missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys),
        runner_sha256=sha(__file__), protocol_sha256=sha(Path(__file__).resolve().parents[1]/'docs/note-dynamics-protocol.md'),
        versions=runtime_versions(), settings=dict(intra_threads=2, inter_threads=1, seed=42,
        amp=False, normalization=False, window_seconds=8., include_stem_gain=False)))
    summaries = []
    start = time.monotonic()
    with torch.inference_mode():
        for case in cases:
            folder = output/case['name']
            folder.mkdir()
            wave, queries = case['wave'], case['queries']
            original_pcm_hash = pcm_sha(wave)
            write(folder/'identity.json', dict(**case['identity'], stem=case['stem'],
                sample_rate=config.sample_rate, samples=wave.shape[-1], pcm_sha256=original_pcm_hash,
                gains_db=case['gains'], original_peak=float(np.max(np.abs(wave)))))
            write(folder/'queries.json', queries)
            starts = np.array([q['onset'] for q in queries], dtype=np.float32)
            ends = np.array([q['offset'] for q in queries], dtype=np.float32)
            outputs, windows = [], []
            for gain in case['gains']:
                changed = wave * np.float32(0. if gain is None else 10 ** (gain/20))
                velocity = np.full(len(queries), np.nan, dtype=np.float32)
                logits = np.full((len(queries), 127), np.nan, dtype=np.float32)
                counts = np.zeros(len(queries), dtype=np.int64)
                for sample in range(0, wave.shape[-1], 8 * config.sample_rate):
                    begin = sample/config.sample_rate
                    indices = np.flatnonzero((starts >= begin) & (starts < begin + 8))
                    if not len(indices):
                        continue
                    chunk = torch.from_numpy(changed[:, sample:sample+8*config.sample_rate])[None, None]
                    kwargs = dict(note_start_seconds=torch.from_numpy(starts[indices]-begin)[None],
                        note_end_seconds=torch.from_numpy(ends[indices]-begin)[None],
                        note_stem_index=torch.zeros((1,len(indices)),dtype=torch.long),
                        stem_class_id=torch.tensor([[STEM_CLASS_BY_NAME[case['stem']]]]), include_stem_gain=False)
                    for field in ('pitch', 'program', 'is_drum'):
                        kwargs['note_'+field] = torch.tensor([[queries[i][field] for i in indices]],dtype=torch.long)
                    predicted = model(chunk, **kwargs)
                    velocity[indices] = predicted['velocity_expected'][0].float().numpy()
                    logits[indices] = predicted['velocity_logits'][0].float().numpy()
                    counts[indices] += 1
                    windows.append(dict(gain_db=gain, start_sample=sample, queries=len(indices)))
                if (not np.isfinite(velocity).all() or not np.isfinite(logits).all()
                        or not np.all(counts == 1) or np.any((velocity<1)|(velocity>127))):
                    raise ValueError('Missing, duplicate or invalid model output')
                tag = 'silence' if gain is None else f'gain-{gain:g}'
                np.savez_compressed(folder/(tag+'.npz'), expected_velocity=velocity,
                    logits=logits, observations=counts)
                row = dict(gain_db=gain, pcm_sha256=pcm_sha(changed), peak=float(np.max(np.abs(changed))),
                    samples_abs_gt_one=int(np.count_nonzero(np.abs(changed)>1)),
                    velocity_min=float(velocity.min()), velocity_max=float(velocity.max()),
                    velocity_median=float(np.median(velocity)), evidence_sha256=sha(folder/(tag+'.npz')))
                outputs.append(row)
                print(json.dumps(dict(case=case['name'], gain_db=gain, queries=len(queries),
                                      elapsed_seconds=round(time.monotonic()-start,2))),flush=True)
            if pcm_sha(wave) != original_pcm_hash:
                raise ValueError('Input PCM was mutated')
            write(folder/'report.json',dict(queries=len(queries),outputs=outputs,windows=windows))
            summaries.append(dict(case=case['name'], report_sha256=sha(folder/'report.json'),
                identity_sha256=sha(folder/'identity.json'), queries_sha256=sha(folder/'queries.json')))
    for filename, digest in unchanged.items():
        if sha(filename) != digest:
            raise ValueError('Input evidence changed during inference: '+filename)
    if verify_source(source) != sources:
        raise ValueError('Source changed during inference')
    write(output/'report.json',dict(cases=summaries,elapsed_seconds=time.monotonic()-start,
        inputs_verified_unchanged=unchanged,enabled_in_application=False,musical_acceptance=False,
        semantics='conditional_MIDI_velocity_not_presence_confidence_or_melodic_role'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--smoke', type=Path, required=True)
    cli.add_argument('--dependencies', type=Path, default=Path('__no_extra_dependencies__'))
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args())


if __name__ == '__main__':
    main()
