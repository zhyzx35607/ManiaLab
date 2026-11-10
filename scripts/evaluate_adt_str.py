"""Score frozen 26-class drum onsets against the author's source selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import sha,event_similarity
from scripts.research_source_attack_classes import drum_event_counts
from scripts.audit_adt_tokens import audit_run


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--audio',type=Path,help='Exact inference input, otherwise the recorded legacy preview')
    args=cli.parse_args();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    if args.output.exists():raise ValueError('Preserve previous evaluations')
    report=read(args.run/'report.json');events=read(args.run/'events.json');manifest=read(args.root/'manifest.json')
    audio=args.audio or args.root/'bundled/drums/preview.wav'
    if (report['events_sha256']!=sha(args.run/'events.json') or report['tokens_sha256']!=sha(args.run/'tokens.npz')
            or report['identity']['audio_sha256']!=sha(audio)):
        raise ValueError('Inference or source identity changed')
    audit=audit_run(args.run)
    if not audit['suitable_for_event_scoring']:raise ValueError('Review malformed upstream decoding before accepting scores: '+str(audit['issues']))
    from scipy.io import wavfile
    rate,pcm=wavfile.read(audio)
    if abs(len(pcm)/rate*1000-report['duration_ms'])>.1:raise ValueError('Inference input duration changed')
    path=args.root/'reference.osu'
    if sha(path)!=manifest['charts']['reference']['chart_sha256']:raise ValueError('Reference changed')
    counts=drum_event_counts(parse_osu(path.read_text(encoding='utf-8-sig'))['notes'])
    groups={'kick':({35,36},'kick'),'snare_strict':({38,40},'snare'),
            'snare_with_stick_clap':({37,38,39,40},'snare'),'hat_strict':({42,43,44},'hi-hat'),
            'hat_tom_diagnostic':({41,42,43,44,45,47},'hi-hat'),
            'remaining_percussion_diagnostic':(set(range(41,61)),'hi-hat')}
    scores={}
    for name,start,end in [('whole',0,report['duration_ms']),('fit',0,162340),('calibration',162340,188286),
                            ('later_sections',188286,report['duration_ms'])]:
        scores[name]={}
        for label,(pitches,kind) in groups.items():
            target=[(t,) for t,c in counts if c==kind and start<=t<end]
            proposed=[(n['onset']*1000-22.5,) for n in events
                      if n['custom_pitch'] in pitches and start<=n['onset']*1000-22.5<end]
            scores[name][label]=event_similarity(target,proposed,[40])
        target=[(t,) for t in sorted({t for t,c in counts if start<=t<end})]
        # A chord may contain multiple classes; unique timestamps score onset
        # groups while the per-class rows retain duplicate observations.
        times=sorted({n['onset']*1000-22.5 for n in events if start<=n['onset']*1000-22.5<end})
        scores[name]['all_onset_groups']=event_similarity(target,[(t,) for t in times],[40])
    result=dict(identity=report['identity'],counts=report['counts'],token_issues=audit['issues'],independent_audit=audit,
        events=report['events'],duplicate_rows=report['duplicate_rows'],scores=scores,
        reference_sha256=sha(path),report_sha256=sha(args.run/'report.json'),evaluator_sha256=sha(__file__),
        enabled_in_application=False,
        limits=['No chart conditioning in transcription. Fixed 22.5ms chart anticipation and 40ms onset tolerance.',
            'Fourth author column contains multiple drum types; diagnostic unions are not hi-hat identification accuracy.',
            'Reference records selection, not exhaustive drum truth. Extra observations are not necessarily nonexistent sound.',
            'Offsets are 100ms placeholders. No LN duration evidence is provided by this model.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(whole=scores['whole'],later=scores['later_sections'],counts=report['counts'])),flush=True)


if __name__=='__main__':main()
