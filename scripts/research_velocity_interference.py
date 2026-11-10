"""Fixed real-instrument gain intervention with frozen transcription queries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_tsumugi_velocity import (
    CHECKPOINT_SHA256, SOURCE_COMMIT, MODEL_REVISION, read, write, sha, pcm_sha,
    verify_source, convert_notes, runtime_versions, program_hypotheses, validate_queries,
)
from scripts.summarize_note_dynamics import distribution

PAIRS = (('12-1-vn', '12-3-vc'), ('31-1-tpt', '31-3-hn'))
CONDITIONS = {'both': (0., 0.), 'attenuate_a': (-12., 0.), 'attenuate_b': (0., -12.),
              'a_only': (0., None), 'b_only': (None, 0.)}


def run(args):
    import numpy as np
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Preserve earlier evidence; use a new directory')
    sources = verify_source(source)
    if sha(args.checkpoint) != CHECKPOINT_SHA256 or args.checkpoint.stat().st_size != 54996249:
        raise ValueError('Velocity checkpoint identity differs')
    sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str(source))
    import torch
    from instrument_agnostic_amt.amt.inference.audio import load_audio
    from instrument_agnostic_amt.velocity.modeling.model import VelocityModelConfig, VelocityPredictionModel
    from instrument_agnostic_amt.velocity.stems import STEM_CLASS_BY_NAME
    from instrument_agnostic_amt.taxonomy.instrument_classes import INSTRUMENT_CLASSES, EXTRA_INSTRUMENT_CLASS_PROGRAMS
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    values = dict(checkpoint['model_config'])
    for key in ('harmonics', 'local_frame_offsets'):
        values[key] = tuple(values[key])
    config = VelocityModelConfig(**values)
    if config.predict_stem_gain or not config.use_absolute_velocity_energy:
        raise ValueError('Unexpected velocity configuration')
    model = VelocityPredictionModel(config).eval()
    loaded = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Incomplete model load')
    labels = read(source/'instrument_agnostic_amt/taxonomy/instrument_merge.json')['labels']
    unchanged, cases = {str(args.checkpoint): CHECKPOINT_SHA256}, []
    for pair in PAIRS:
        waves, queries, identities = [], [], []
        for source_index, name in enumerate(pair):
            folder = args.inputs/name
            report = read(folder/'report.json')
            identity = report['identity']
            if (identity['source_commit'] != SOURCE_COMMIT or identity['source_files'] != sources
                    or identity['model_revision'] != MODEL_REVISION
                    or report['context_start_seconds'] != 0 or report['context_end_seconds'] != 40):
                raise ValueError('Unexpected original source or 0-40 second clock')
            for filename, key in (('notes.json', 'notes_sha256'), ('raw-notes.json', 'raw_notes_sha256')):
                if sha(folder/filename) != report[key]:
                    raise ValueError('Changed note evidence')
                unchanged[str(folder/filename)] = report[key]
            unchanged[str(folder/'report.json')] = sha(folder/'report.json')
            audio = Path(identity['audio'])
            if sha(audio) != identity['audio_sha256']:
                raise ValueError('Changed original PCM file')
            unchanged[str(audio)] = identity['audio_sha256']
            events, raw = read(folder/'notes.json'), read(folder/'raw-notes.json')
            if events != convert_notes(raw, sample_rate=identity['resampled_sample_rate'],
                    total_samples=identity['resampled_samples'], classes=INSTRUMENT_CLASSES):
                raise ValueError('Original raw note conversion differs')
            wave = load_audio(audio, target_sample_rate=config.sample_rate).numpy()
            if (tuple(wave.shape) != (2, 40 * config.sample_rate)
                    or config.sample_rate != identity['resampled_sample_rate']
                    or pcm_sha(wave) != identity['resampled_pcm_sha256']):
                raise ValueError('Original PCM clock or level differs')
            waves.append(wave)
            identities.append(identity)
            queries.extend(dict(original_index=i, source_index=source_index, source_case=name,
                onset=n['onset'], offset=n['offset'], pitch=n['pitch'], is_drum=n['is_drum'],
                program=p, program_basis=basis, instrument_class=n['instrument_class'],
                model_has_onset=n['model_has_onset'], release_reason=n['release_reason'])
                for i,n in enumerate(events) for p,basis in program_hypotheses(n,labels,EXTRA_INSTRUMENT_CLASS_PROGRAMS))
        validate_queries(queries, 40., config.pitch_min, config.pitch_max)
        cases.append(dict(name='__'.join(pair), waves=waves, queries=queries, identities=identities))
    output.mkdir(parents=True)
    write(output/'identity.json', dict(source_commit=SOURCE_COMMIT, source_files=sources,
        model_revision=MODEL_REVISION, checkpoint_sha256=CHECKPOINT_SHA256, model_config=asdict(config),
        runner_sha256=sha(__file__), protocol_sha256=sha(Path(__file__).resolve().parents[1]/'docs/velocity-interference-protocol.md'),
        helpers_sha256={name:sha(Path(__file__).with_name(name)) for name in (
            'research_tsumugi_velocity.py', 'research_tsumugi_transcription.py','summarize_note_dynamics.py')},
        versions=runtime_versions(), settings=dict(window_seconds=8., intra_threads=2, inter_threads=1,
            normalization=False, clipping=False, amp=False, include_stem_gain=False),
        query_source_semantics='known_separated_source_not_identity_inferred_from_mix'))
    started = time.monotonic()
    comparisons = []
    with torch.inference_mode():
        for case in cases:
            folder = output/case['name']
            folder.mkdir()
            waves, queries = case['waves'], case['queries']
            hashes = [pcm_sha(wave) for wave in waves]
            write(folder/'queries.json', queries)
            write(folder/'sources.json', case['identities'])
            starts = np.array([q['onset'] for q in queries], dtype=np.float32)
            ends = np.array([q['offset'] for q in queries], dtype=np.float32)
            predictions, records = {}, []
            for name, gains in CONDITIONS.items():
                wave = sum(w * np.float32(0. if db is None else 10 ** (db/20)) for w,db in zip(waves,gains))
                velocity = np.full(len(queries), np.nan, dtype=np.float32)
                logits = np.full((len(queries),127), np.nan, dtype=np.float32)
                counts = np.zeros(len(queries), dtype=np.int64)
                for sample in range(0,wave.shape[-1],8*config.sample_rate):
                    begin = sample/config.sample_rate
                    indices = np.flatnonzero((starts>=begin)&(starts<begin+8))
                    if not len(indices):
                        continue
                    kwargs = dict(note_start_seconds=torch.from_numpy(starts[indices]-begin)[None],
                        note_end_seconds=torch.from_numpy(ends[indices]-begin)[None],
                        note_stem_index=torch.zeros((1,len(indices)),dtype=torch.long),
                        stem_class_id=torch.tensor([[STEM_CLASS_BY_NAME['other']]]), include_stem_gain=False)
                    for field in ('pitch','program','is_drum'):
                        kwargs['note_'+field] = torch.tensor([[queries[i][field] for i in indices]],dtype=torch.long)
                    predicted = model(torch.from_numpy(wave[:,sample:sample+8*config.sample_rate])[None,None],**kwargs)
                    velocity[indices] = predicted['velocity_expected'][0].float().numpy()
                    logits[indices] = predicted['velocity_logits'][0].float().numpy()
                    counts[indices] += 1
                if not np.isfinite(velocity).all() or not np.isfinite(logits).all() or not np.all(counts==1):
                    raise ValueError('Missing, repeated or nonfinite query output')
                np.savez_compressed(folder/(name+'.npz'),expected_velocity=velocity,logits=logits,observations=counts)
                predictions[name] = velocity
                records.append(dict(condition=name, gains_db=gains, pcm_sha256=pcm_sha(wave),
                    peak=float(np.max(np.abs(wave))), samples_abs_gt_one=int(np.count_nonzero(np.abs(wave)>1)),
                    output_sha256=sha(folder/(name+'.npz'))))
                print(f'{case["name"]} {name} queries={len(queries)} elapsed={time.monotonic()-started:.1f}s',flush=True)
            summaries = []
            for source_index in (0,1):
                indices = np.array([i for i,q in enumerate(queries) if q['source_index']==source_index])
                own = 'attenuate_a' if source_index==0 else 'attenuate_b'
                other = 'attenuate_b' if source_index==0 else 'attenuate_a'
                absent = 'b_only' if source_index==0 else 'a_only'
                own_delta = predictions['both'][indices]-predictions[own][indices]
                other_delta = predictions['both'][indices]-predictions[other][indices]
                summaries.append(dict(source_index=source_index,queries=len(indices),
                    own_attenuation_response=distribution(own_delta),
                    own_attenuation_wrong_direction_queries=int(np.sum(own_delta < -1e-5)),
                    other_attenuation_response=distribution(other_delta),
                    other_attenuation_absolute_response=distribution(np.abs(other_delta)),
                    absent_source_velocity=distribution(predictions[absent][indices]),
                    baseline_velocity=distribution(predictions['both'][indices])))
            if [pcm_sha(wave) for wave in waves] != hashes:
                raise ValueError('Original PCM mutated')
            row=dict(case=case['name'],conditions=records,summary=summaries,
                queries_sha256=sha(folder/'queries.json'),sources_sha256=sha(folder/'sources.json'))
            write(folder/'report.json',row)
            comparisons.append(row)
    for filename,digest in unchanged.items():
        if sha(filename)!=digest:
            raise ValueError('Input evidence changed during analysis')
    if verify_source(source)!=sources:
        raise ValueError('Source changed during analysis')
    write(output/'report.json',dict(cases=comparisons,elapsed_seconds=time.monotonic()-started,
        inputs_verified_unchanged=unchanged, enabled_in_application=False,musical_acceptance=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','dependencies','inputs','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    run(cli.parse_args())
