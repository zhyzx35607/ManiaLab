"""Publish aggregate acoustic-head adaptation results, without paired audio."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.train_piano_onset_head import pooled, eligible
from scripts.piano_match_coverage import changes, summarize_changes


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cache',type=Path,required=True)
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    cache,run=args.cache.resolve(),args.run.resolve()
    source,report=read(cache/'report.json'),read(run/'report.json')
    if args.output.exists():
        raise ValueError('Preserve earlier aggregate')
    if report['identity']['cache_report_sha256']!=sha256(cache/'report.json'):
        raise ValueError('Training/cache identities differ')
    for name,path in [('scoring','evaluate_piano_acoustics.py'),('runner','train_piano_onset_head.py')]:
        if report['identity']['code'][name]!=sha256(ROOT/'scripts'/path):
            raise ValueError('Current helper differs from the training snapshot: '+path)
    for folder,data in [(cache,source),(run,report)]:
        for name,digest in data['artifacts_sha256'].items():
            if sha256(folder/name)!=digest:
                raise ValueError('Saved evidence changed: '+name)
    if len(report['epochs'])!=3 or len(source['records'])!=10 or not report['frozen_model_unchanged']:
        raise ValueError('Incomplete run')
    output=[]
    baseline=report['epochs'][0]['validation']
    for epoch in report['epochs']:
        number=epoch['epoch']
        if eligible(baseline,epoch['validation'])!=epoch['selection']:
            raise ValueError('Stored selection is inconsistent')
        differences={label:dict(recovered=0,new_missing=0,unmatched_predictions=0,
                               unavoidable_recovered=0,unavoidable_new_missing=0,
                               before_ambiguous_coverage=0,after_ambiguous_coverage=0)
                     for label in ('50ms','40ms')}
        by_recording=[]
        for row in epoch['validation']:
            name=row['id']
            scope=lambda ns:[n for n in ns if 0<=n['onset']<170]
            ref=scope(read(cache/(name+'-labels.json'))['notes'])
            old=scope(read(run/(name+'-epoch0-events.json'))['notes'])
            new=scope(read(run/(name+f'-epoch{number}-events.json'))['notes'])
            delta={}
            for label,strict in [('50ms',False),('40ms',True)]:
                delta[label]=summarize_changes(changes(ref,old,new,strict=strict),len(new))
                for key,value in delta[label].items():
                    differences[label][key]+=value
            by_recording.append(dict(**row,differences=delta))
        output.append(dict(epoch=number,head_sha256=epoch['head_sha256'],training_bce=epoch['loss'],
            supervised_cells=epoch['supervised_cells'],selection=epoch['selection'],differences=differences,
            pooled={semantics:{label:{field:pooled(epoch['validation'],semantics,label,field)
                         for field in ('onsets','intervals')} for label in ('50ms','40ms')}
                         for semantics in ('key','sustain')},recordings=by_recording))
    selected=max((e for e in output if e['selection']['eligible']),
                 key=lambda e:(e['selection']['pooled_onsets']['f1'],-e['epoch']))['epoch']
    if selected!=report['selected_epoch']:
        raise ValueError('Stored epoch does not follow the fixed selection rule')
    result=dict(source=dict(cache_report_sha256=sha256(cache/'report.json'),
        training_report_sha256=sha256(run/'report.json'),summarizer_sha256=sha256(Path(__file__)),
        matching_coverage_sha256=sha256(ROOT/'scripts/piano_match_coverage.py'),
        selection_sha256=source['identity']['selection_sha256'],code=report['identity']['code'],
        model=report['identity']['model']),
        configuration={key:value for key,value in report['identity'].items()
                       if key not in ('cache_report_sha256','model','code')},
        feature_preparation=dict(recordings=len(source['records']),windows=len(source['windows']),
            seconds=sum(r['elapsed_seconds'] for r in source['windows']),
            seven_output_epoch0_replay_identical=source['all_epoch0_outputs_identical']),
        epochs=output,selected_epoch=selected,frozen_model_unchanged=True,
        test_accessed=False,training_performed=True,application_enabled=False,
        direct_listening_performed=False,musical_acceptance=False,
        differences_interpretation='recovered/new_missing are deterministic assignment changes; '
            'unavoidable_* holds under every maximum-cardinality onset matching, '
            'conservatively ignoring the secondary minimum-time-error objective.',
        interpretation='Small final-head fit on a public development subset; not an unseen-pretraining benchmark or LN acceptance.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(selected_epoch=selected,epochs=[dict(epoch=e['epoch'],selection=e['selection'],
                                                             differences=e['differences']) for e in output])),flush=True)


if __name__=='__main__':
    main()
