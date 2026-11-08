"""Publish measurements/protocol only; keep authored events, audio and weights local."""
import argparse
import json
from pathlib import Path


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,default=Path('docs/sakura-source-research.json'))
    args=cli.parse_args();root=args.root
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    manifest=read('manifest.json')
    result=dict(revision=6,status='development_not_accepted',date='2026-11-08',
        reference=dict(beatmapset_id=manifest['beatmapset_id'],archive_sha256=manifest['archive_sha256'],
            charts={role:{k:v for k,v in row.items() if k in ('beatmap_id','official_checksum','chart_sha256','version','validation')}
                    for role,row in manifest['charts'].items()},annotations=manifest['annotations']),
        source_detection=read('detector-baseline/report.json'),piano_comparison=read('piano-specialist/comparison.json'),
        drum_classes=read('drum-classifier/report.json'),piano_layers=read('piano-layers/report.json'),
        specialist_piano_layers=read('piano-specialist-layers/report.json'),
        drum_pretrained=read('drum-adtof/comparison.json'),
        drum_pretrained_selection=read('drum-classifier-adtof/report.json'),
        authored_expression=read('expression-study.json'),source_free_generation=read('source-free-215853-12/source-comparison.json'),
        blind_mix_probes={})
    for voice in ('mix','accompaniment'):
        row=read(f'mapper-6664-26/{voice}/result.json')
        result['blind_mix_probes'][voice]={k:v for k,v in row.items() if k not in ('notes','raw_file','warnings')}
        result['blind_mix_probes'][voice]['incomplete_hold_warnings']=sum('Incomplete hold' in w for w in row['warnings'])
    training=[('source_training','source-lora'),('expression_training','expression-lora-v2')]
    cold_path=root/'expression-lora-cold-v2/report.json'
    if cold_path.exists() and 'completed_steps' in read('expression-lora-cold-v2/report.json'):
        training.append(('expression_cold_start_training','expression-lora-cold-v2'))
    prefix_path=root/'expression-lora-prefix-v1/report.json'
    if prefix_path.exists() and 'completed_steps' in read('expression-lora-prefix-v1/report.json'):
        training.append(('expression_prefix_weighted_training','expression-lora-prefix-v1'))
    for role,folder in training:
        row=read(folder+'/report.json');samples=read(folder+'/samples.json')
        result[role]={k:row[k] for k in ('identity','runner_sha256','steps_requested','completed_steps','trainable_parameters',
            'validation_scope','key_schedule','adapter_sha256','baseline','after')}
        result[role]['sample_windows']=samples['charts']
    result['expression_free_generation']={}
    for experiment in ('expression-fixed-215853-12','expression-cold-215853-12','expression-prefix-215853-12',
            'expression-prefix-241798-12','expression-constrained-v2-215853-12'):
        result['expression_free_generation'][experiment]={}
        for variant in ('baseline','adapter'):
            if not (root/experiment/variant/'result.json').exists():continue
            row=read(f'{experiment}/{variant}/result.json')
            result['expression_free_generation'][experiment][variant]={k:v for k,v in row.items()
                if k not in ('notes','raw_file','warnings')}
            result['expression_free_generation'][experiment][variant]['incomplete_hold_warnings']=sum(
                'Incomplete hold' in warning for warning in row['warnings'])
    result['cold_start_aborted']=read('expression-lora-cold/aborted.json')
    causal=root/'expression-lora-cold-v2/causal-prefix-report.json'
    if causal.exists():
        row=read('expression-lora-cold-v2/causal-prefix-report.json')
        result['causal_prefix_check']=dict(passed=row['causal'],comparisons=len(row['rows']),
            max_logit_difference=max(r['max_logit_difference'] for r in row['rows']),
            notice='Full teacher-forced context and prefix-only logits agree; no causal-mask leakage found in these checks. First-event generation still fails.')
    real_prefix=root/'expression-lora-prefix-v1/real-prefix-report.json'
    if real_prefix.exists():
        row=read('expression-lora-prefix-v1/real-prefix-report.json')
        result['real_song_prefix_check']=dict(passed=row['all_equal'],comparisons=len(row['rows']),
            notice='Real cached cold training windows match inference prompts from both the original sketch and the application wrapper context. No finished events enter inference.')
    pipeline=root/'provided-pipeline/report.json'
    if pipeline.exists():
        result['application_provided_pipeline']={k:v for k,v in read('provided-pipeline/report.json').items() if k!='expression'}
    structure=root/'structure-comparison.json'
    if structure.exists():result['structure_model_comparison']=read('structure-comparison.json')
    transcription=root/'instrument-transcription-comparison.json'
    if transcription.exists():result['instrument_transcription_comparison']=read('instrument-transcription-comparison.json')
    drum_selection=root/'drum-classifier-yourmt3/report.json'
    if drum_selection.exists():result['drum_instrument_selection']=read('drum-classifier-yourmt3/report.json')
    result['instrument_source_layers']={}
    for role in ('piano','other'):
        name=role+'-yourmt3-layers/report.json'
        if (root/name).exists():result['instrument_source_layers'][role]=read(name)
    transfer='piano-layer-transfer-v1/report.json'
    if (root/transfer).exists():result['instrument_source_layer_transfer']=read(transfer)
    controls=Path('test-data/yourmt3/control-comparison.json')
    if controls.exists():result['instrument_transcription_controls']=json.loads(controls.read_text())
    result['instrument_release_audits']={}
    for folder in ('other-yourmt3-full','drums-yourmt3-full','piano-yourmt3-full',
            'piano-yourmt3-215853-12','drums-yourmt3-215853-12'):
        if (root/folder/'release-audit.json').exists():
            result['instrument_release_audits'][folder]=read(folder+'/release-audit.json')
    failed=root/'expression-constrained-215853-12/adapter/failure.json'
    if failed.exists():result['event_constraint_aborted']=read('expression-constrained-215853-12/adapter/failure.json')
    grammar=root/'expression-lora-prefix-v1/event-grammar-check.json'
    if grammar.exists():result['event_grammar_check']=read('expression-lora-prefix-v1/event-grammar-check.json')
    timing=Path('docs/timing-grid-support-validation.json')
    if timing.exists():
        row=json.loads(timing.read_text())
        result['timing_grid_fix']=dict(summary=row['summary'],sakura=next(r for r in row['rows'] if r['group']=='sakura'))
    check=Path('test-data/mapper-sketch-fixture/report.json')
    if check.exists():result['sketch_parser_integration']=json.loads(check.read_text())
    result['decoder_trace_diagnosis']=dict(
        observed='Previous expression adapter starts with a partial group, then a time at 12360ms, beyond 9825.6ms lookahead cutoff.',
        hypothesis='Cold-start supervision alone failed. First-event weighting restored generation but overlaps and unclosed holds remain. Event-state constraints collapsed LN generation and still left a snapped overlap; rejected.',
        private_token_traces_retained=True)
    result['invalid_expression_attempt']=dict(reason='Upstream ignored unrecognized GD input; not an arrangement evaluation.',
        raw_result_retained_locally=True,reported_as_success=False)
    result['limits']=['One author-labelled song already used for development; later sections contain repeated musical material.',
        'Reference piano taps supervise onset/layer choices, not acoustic offsets or MIDI-pitch truth.',
        'Author Nova stems were supplied in the mapset; this is not application separation quality.',
        'No research adapter is enabled by default. Both source and expression quality remain unaccepted.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(args.output),flush=True)


if __name__=='__main__':main()
