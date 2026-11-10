"""Send completed Nova/Tsumugi song evidence through the real 6–10K arranger.

Controlled local comparison, with frozen section plan and manual chart clock.
Raw model durations are hypotheses, not automatically validated LN gestures.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.io import wavfile
from manialab.audio import SR, source_evidence
from manialab.arrange import parameters, assign_lanes, layout_metrics
from manialab.domain import UNOBSERVED_RELEASES, note_from, validate, export_osu, parse_osu
from manialab.expression import propose_expression, expression_evidence, select_expression, signature
from manialab.timing import chart_clock_notes, align_candidates, grid_report

read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_analysis(analysis, roles, uniform_weights=False):
    result = copy.deepcopy(analysis)
    if uniform_weights:
        # This is a symmetrical arrangement ablation, not new confidence.
        # Retain unmodified source analyses and record this choice per case.
        for collection in ('candidates', 'phrase_candidates'):
            for note in result.get(collection, []):
                if note['voice'] in roles:
                    note['strength'] = 1.
    return result


def frozen_expression(analysis, settings, reference_plan):
    result = propose_expression(analysis, settings)
    assert len(result['sections']) == len(reference_plan['sections'])
    for target, source in zip(result['sections'], reference_plan['sections']):
        assert (target['start_ms'], target['end_ms']) == (source['start_ms'], source['end_ms'])
        target['main_voice'] = source['main_voice']
        target['voices'] = {voice:copy.deepcopy(source['voices'].get(voice, dict(mode='omit', role='support')))
                            for voice in target['voices']}
        # New model identities inherit the same voice policy. Never reuse
        # instrument IDs from another model or fabricate a baseline policy.
        target['instrument_modes'] = {voice:{identity:'inherit' for identity in modes}
                                      for voice,modes in target['instrument_modes'].items()}
        target['reason'] = 'Controlled comparison: frozen baseline voice expression.'
    result['evidence_signature'] = signature(analysis)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, default=Path('test-data/sakura-tsumugi-fullsong-v1'))
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve completed and partial runs; choose a new destination')
    root = args.input
    plan = read(root/'execution-plan.json')
    completed = read(root/'completed.json')
    assert completed['song_inputs'] == len(completed['results']) == len(plan['sources']) == 4
    assert completed['plan_sha256'] == sha(root/'execution-plan.json')
    assert completed['parity_sha256'] == sha(root/'parity.json')
    assert sha(root/'application-analysis.json') == plan['application_analysis_sha256']
    assert sha('scripts/tsumugi_candidate_adapter.py') == plan['bridge_sha256']
    assert all(sha(path)==digest for path,digest in plan['application_sources'].items()), 'Prepared application sources changed'
    baseline = read(root/'application-analysis.json')
    revised = copy.deepcopy(baseline)
    roles = [row['voice'] for row in plan['sources']]
    assert sorted(roles) == ['bass','guitar','other','vocals']
    mix_path = Path(plan['sources'][0]['audio']).with_name('mix.wav')
    assert sha(mix_path) == plan['mix_sha256']
    sr, mix = wavfile.read(mix_path)
    assert sr == SR and mix.dtype == np.float32 and mix.ndim == 1
    replacements, traces, report_hashes = [], {}, {}
    onset_replacements, onset_traces = [], {}
    for row in plan['sources']:
        voice = row['voice']
        source = Path(row['audio'])
        assert sha(source) == row['audio_sha256']
        rate, signal = wavfile.read(source)
        assert rate == sr and signal.dtype == np.float32 and signal.ndim == 1 and len(signal) == len(mix)
        bridge_path = root/(voice+'-predicted-release.json')
        bridge = read(bridge_path)
        model_dir = root/('model-'+voice)
        model_report = read(model_dir/'report.json')
        completion = next(r for r in completed['results'] if r['voice']==voice)
        assert completion['report_sha256'] == sha(model_dir/'report.json') == bridge['report_sha256']
        assert bridge['identity'] == model_report['identity']
        assert bridge['identity']['audio_sha256'] == row['audio_sha256']
        assert bridge['source_voice'] == voice and bridge['allow_predicted_releases'] is True
        assert bridge['recover_filtered_attacks'] is False and bridge['baseline_modified'] is False
        assert bridge['runner_sha256'] == plan['bridge_sha256']
        for filename, digest in bridge['evidence_sha256'].items():
            assert sha(model_dir/filename) == digest
        assert bridge['baseline_notes'] == read(model_dir/'notes.json')
        raw = [note_from(n) for n in bridge['candidates']]
        assert all(n.voice==voice and n.end<=round(len(signal)*1000/sr) for n in raw)
        selected, rejected = source_evidence(raw, signal, mix)
        replacements.extend(n.data() for n in selected)
        kept = {n.id for n in selected}
        trace = copy.deepcopy(bridge['trace'])
        for item in trace:
            if item['decision']=='candidate' and item['id'] not in kept:
                item['decision']='below_source_energy_floor'
        traces[voice] = dict(trace=trace, raw_candidates=len(raw), accepted=len(selected), rejected_low_energy=rejected)
        report_hashes[voice] = sha(bridge_path)
        onset_path = root/(voice+'-onsets-only.json')
        onset_bridge = read(onset_path)
        assert onset_bridge['identity']==bridge['identity'] and onset_bridge['evidence_sha256']==bridge['evidence_sha256']
        assert onset_bridge['allow_predicted_releases'] is False and onset_bridge['recover_filtered_attacks'] is False
        assert onset_bridge['runner_sha256']==plan['bridge_sha256'] and onset_bridge['baseline_notes']==bridge['baseline_notes']
        onset_raw = [note_from(n) for n in onset_bridge['candidates']]
        assert [n.id for n in onset_raw]==[n.id for n in raw]
        assert all((a.start,a.end,a.pitch,a.voice,a.strength,a.instrument_id)==
                   (b.start,b.end,b.pitch,b.voice,b.strength,b.instrument_id) for a,b in zip(raw,onset_raw))
        onset_selected = [n for n in onset_raw if n.id in kept]
        assert all(n.release_reason in UNOBSERVED_RELEASES for n in onset_selected)
        onset_replacements.extend(n.data() for n in onset_selected)
        onset_traces[voice] = dict(bridge_sha256=sha(onset_path), candidates=len(onset_selected))
        for summary in revised['voices']:
            if summary['voice']==voice:
                summary.update(candidates=len(selected), detector='tsumugi-research', cached=True,
                    rejected_low_energy_pitch=rejected,
                    sustains_over_800ms=sum(n.end-n.start>=800 for n in selected),
                    release_refinement=dict(model_hypotheses=True, bridge_sha256=sha(bridge_path)))
    revised['candidates'] = sorted([n for n in revised['candidates'] if n['voice'] not in roles]+replacements,
        key=lambda n:(n['start'],n['pitch'],n['voice']))
    # No frame activations accompany these new source events. Do not recycle
    # phrase candidates constructed from the old model's unrelated intervals.
    revised['phrase_candidates'] = [n for n in revised['phrase_candidates'] if n['voice'] not in roles]
    revised['instrument_engine'] = 'research-tsumugi'
    # Cross-source bleed was not recomputed by preparation or this comparison.
    # Inherited diagnostics must not masquerade as checks of the new events.
    for analysis in (baseline,revised):
        analysis['source_bleed_status'] = 'not_recomputed_for_comparison'
        analysis['prior_source_bleed_rejections'] = analysis.pop('source_bleed_rejections', [])
        for summary in analysis['voices']:
            if 'rejected_cross_source_bleed' in summary:
                summary['prior_rejected_cross_source_bleed'] = summary.pop('rejected_cross_source_bleed')
    onsets_only = copy.deepcopy(revised)
    onsets_only['candidates'] = sorted([n for n in revised['candidates'] if n['voice'] not in roles]+onset_replacements,
                                      key=lambda n:(n['start'],n['pitch'],n['voice']))
    for summary in onsets_only['voices']:
        if summary['voice'] in roles:
            summary['release_refinement'] = dict(model_hypotheses=False,
                bridge_sha256=onset_traces[summary['voice']]['bridge_sha256'])
    args.output.mkdir(parents=True)
    def save(name, value):
        with (args.output/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    save('analysis.json', revised)
    save('onsets-only-analysis.json', onsets_only)
    save('candidate-trace.json', traces)
    config = plan['frozen_generation']
    clock = config['generation_clock']
    cases, chart_paths = [], []
    analyses = {'current-basic':baseline,'tsumugi-onsets':onsets_only,'tsumugi-releases':revised}
    reference_plan = propose_expression(baseline,parameters(config['settings']|{'keys':10}))
    suite = [(engine,keys,False) for keys in range(6,11) for engine in analyses]
    suite += [(engine,10,True) for engine in analyses]
    for engine, keys, controlled in suite:
        analysis = comparison_analysis(analyses[engine],roles,controlled)
        settings = parameters(config['settings']|{'keys':keys})
        assert settings['vocals_weight']==0
        expression = frozen_expression(analysis,settings,reference_plan) if controlled else propose_expression(analysis, settings)
        evidence, metadata, decisions = expression_evidence(analysis, expression, settings)
        aligned, alignment = align_candidates(chart_clock_notes(evidence,clock['audio_shift_ms']), clock, confirmed=True)
        selected, selection = select_expression(aligned,metadata,decisions,settings,60000/clock['bpm'])
        placed = assign_lanes(selected,settings,beat_ms=60000/clock['bpm'])
        original = {n.id:n for n in selected}
        assert len(placed)==len(selected)
        assert all((n.start,n.end,n.voice,n.pitch)==(original[n.id].start,original[n.id].end,
                   original[n.id].voice,original[n.id].pitch) for n in placed)
        assert all(n.voice!='vocals' and (not n.hold or n.release_reason not in UNOBSERVED_RELEASES) for n in placed)
        structural, grid = validate(placed,keys), grid_report(placed,clock)
        assert structural['valid'] and grid['off_grid']==0
        name = f'{engine}-{keys}k'+('-unit-fixed-expression' if controlled else '')
        content = export_osu(placed,keys,Path('vendor/mapperatorinator/template.osu'),
            title='Sakura instrument comparison',audio='audio.mp3',bpm=clock['bpm'],offset=clock['offset'],
            version=name+' research',grid_division=clock['division'])
        parsed = parse_osu(content)
        assert validate(parsed['notes'],keys)['valid'] and len(parsed['notes'])==len(placed)
        path = args.output/(name+'.osu')
        path.write_text(content,encoding='utf-8')
        chart_paths.append(path)
        save(name+'-notes.json',[n.data() for n in placed])
        save(name+'-expression.json',expression)
        save(name+'-selection.json',selection)
        cases.append(dict(name=name,keys=keys,notes=len(placed),ln=sum(n.hold for n in placed),
            equal_unit_weights_for_replaced_voices=controlled, voice_expression_frozen=controlled,
            weight_semantics='unit_weight_for_replaced_voices' if controlled else 'native_model_weights_not_comparable_confidence',
            section_main_voices=[s['main_voice'] for s in expression['sections']],
            section_modes=[{v:p['mode'] for v,p in s['voices'].items()} for s in expression['sections']],
            source_candidates=dict(Counter(n['voice'] for n in analysis['candidates'])),
            source_counts=dict(Counter(n.voice for n in placed)),
            source_holds=dict(Counter(n.voice for n in placed if n.hold)),
            large_attack_shifts=dict(Counter(n.voice for n in placed if abs(n.start-n.evidence_start)>30)),
            all_expression_attack_shifts_over_30ms=sum(abs(n.start-n.evidence_start)>30 for n in aligned),
            source_bleed_status='not_recomputed_for_comparison',
            structural=structural,grid=grid,alignment=alignment,omissions=selection['omission_reasons'],
            layout=layout_metrics(placed,keys,60000/clock['bpm'])))
        print(json.dumps(dict(case=name,notes=len(placed),source_counts=cases[-1]['source_counts'])),flush=True)
    original_audio = Path('test-data/sakura-benchmark/piano-full-pipeline-v1/audio.mp3')
    assert sha(original_audio) == config['mix_sha256']
    shutil.copyfile(original_audio,args.output/'audio.mp3')
    with zipfile.ZipFile(args.output/'sakura-instrument-comparison.osz','x',zipfile.ZIP_DEFLATED) as archive:
        archive.write(original_audio,'audio.mp3')
        for path in chart_paths:
            archive.write(path,path.name)
    save('report.json',dict(cases=cases,source_plan_sha256=sha(root/'execution-plan.json'),
        source_completed_sha256=sha(root/'completed.json'),bridges=report_hashes,onset_bridges=onset_traces,runner_sha256=sha(__file__),
        application_sources={p:sha(p) for p in ('manialab/audio.py','manialab/arrange.py','manialab/expression.py',
            'manialab/domain.py','manialab/timing.py')},
        generation_clock=clock,automatic_timing_acceptance=False,reference_chart_read=False,
        baseline_coverage_keys=list(range(6,11)), source_filter='same_local_energy_floor',
        processing_differences=['Basic Pitch retains its bass suppression, release refinement and transient fusion.',
            'Tsumugi retains its traced model events without generic transient fusion.',
            'Native weights have different semantics; three 10K cases use equal unit weights and frozen voice modes.',
            'No cross-source bleed rerun; native expression recommendations are recomputed per analysis.',
            'Forced grid checks do not evaluate automatic timing or acoustic correctness.'],
        enabled_in_application=False,portable_runtime_verified=False,musical_acceptance=False,
        source_audio_sha256=sha(original_audio),osz_sha256=sha(args.output/'sakura-instrument-comparison.osz')))


if __name__=='__main__':
    main()
