"""Export aggregate research evidence; never publish audio, charts or cache paths."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def per_key(windows):
    groups=defaultdict(lambda:defaultdict(list))
    for row in windows:
        for metric, stats in row['stats'].items():
            if stats['tokens']:
                groups[(row['keys'],metric)][row['id']].append(stats)
    output={}
    for (key,metric),songs in groups.items():
        output.setdefault(str(key),{})[metric]={
            name:sum(sum(s[name] for s in values)/len(values) for values in songs.values())/len(songs)
            for name in ('nll','accuracy')}
    return output


def summarize_run(path):
    report=read(path/'report.json')
    if 'completed_steps' not in report:
        raise ValueError(f'Incomplete run: {path.name}')
    summary={key:report[key] for key in ('identity','trainable_parameters','completed_steps','training_seconds','adapter_sha256','enabled_in_application')}
    summary['charts_seen_in_optimizer']=len(report['charts_seen_in_optimizer'])
    summary['steps_by_keys']=dict(Counter(r['keys'] for r in report['training']))
    summary['steps_by_band']=dict(Counter(r['band'] for r in report['training']))
    for stage in ('baseline','after'):
        summary[stage]=dict(macro=report[stage]['macro'],by_key=per_key(report[stage]['windows']))
    samples=read(path/'samples.json')
    summary['windows']=sum(len(c['windows']) for c in samples['charts'])
    summary['capacity_rejections']=sum(c.get('audit',{}).get('over_capacity',0) for c in samples['charts'])
    summary['complete_candidate_windows']=sum(c.get('audit',{}).get('complete_windows',0) for c in samples['charts'])
    if (path/'audio-evaluation.json').exists():
        audio=read(path/'audio-evaluation.json')
        summary['context_audit']=audio['summary']
        summary['audio_counterfactual']={stage:{condition:values['macro'] for condition,values in groups.items()}
                                         for stage,groups in audio['results'].items()}
    return summary


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data'))
    cli.add_argument('--output',type=Path,default=Path('docs/mapper-audio-research.json'))
    args=cli.parse_args();training=args.root/'mapper-training'
    report=dict(revision=1,date='2026-11-08',default_model_changed=False,new_release_published=False,
                data_counts=read(training/'mmrs/manifest.json')['counts'],runs={},free_generation=[],stem_probe=[])
    for name in ('lora','lora-context'):
        report['runs'][name]=summarize_run(training/name)
    for path in sorted((args.root/'mapper-adapter-probe').glob('*/report.json')):
        probe=read(path)
        names=[name for name in ('baseline','adapter') if name in probe]
        if not names:continue
        report['free_generation'].append(dict(experiment=path.parent.name,
            variants={name:{key:probe[name][key] for key in ('identity','elapsed_seconds','raw_sha256','comparison','validation')}
                      | dict(incomplete_holds=sum('Incomplete hold' in w for w in probe[name]['warnings']))
                      for name in names}))
    report['user_audio_probe']={}
    for name in ('baseline','adapter'):
        path=args.root/'mapper-mix-condition'/name/'mix/result.json'
        if path.exists():
            row=read(path)
            report['user_audio_probe'][name]={key:row[key] for key in ('context','elapsed_seconds','raw_sha256','validation')}
            report['user_audio_probe'][name]['incomplete_holds']=sum('Incomplete hold' in w for w in row['warnings'])
    for row in read(args.root/'mapper-stem-probe/report.json'):
        report['stem_probe'].append({key:row[key] for key in ('voice','context','elapsed_seconds','raw_sha256','validation')}
            | dict(incomplete_holds=sum('Incomplete hold' in w for w in row['warnings'])))
    report['limitations']=[
        'Teacher-forced metrics do not measure free-generation quality.',
        'Human authored events are one interpretation, not exhaustive instrument transcription truth.',
        'Fine-tuning validation songs may have appeared in base-model pretraining.',
        'Separated instruments and section plans are not joint model training inputs yet.',
        'No Nova private weights; public Demucs stems only.',
        'No new research adapter is enabled in the application.',
    ]
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),runs=list(report['runs']),free_generation=len(report['free_generation']))))


if __name__=='__main__':main()
