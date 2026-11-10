"""Compare source-aware lane assignment on already frozen gesture intervals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.anchor_gestures import arrange_gestures
from manialab.arrange import layout_metrics
from manialab.domain import note_from, parse_osu, export_osu, validate
from scripts.probe_mapper_adapter import sha, write_review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve original assignments')
    original=parse_osu((args.input/'planned.osu').read_text(encoding='utf-8-sig'))
    notes=[note_from(n) for n in json.loads((args.input/'learned-events.json').read_text(encoding='utf-8'))]
    # Events saved for structural checking already have temporary colored
    # lanes. Recover original source columns by immutable input IDs.
    from scripts.train_anchor_gestures import source_chart
    root=Path('test-data/sakura-benchmark')
    sketch=source_chart(root/'detected-onset-sketch-v1/sketch.osu')
    source={n.id:n for n in sketch['notes']}
    from dataclasses import replace
    notes=[replace(n, lane=source[n.id].lane, voice=source[n.id].voice) for n in notes]
    arranged=arrange_gestures(notes, 10, original['timing'], dict(release_gap=0, chord_size=10, max_held=10))
    if Counter((n.start,n.end) for n in arranged)!=Counter((n.start,n.end) for n in original['notes']):
        raise ValueError('Relayout changed an attack or release')
    args.output.mkdir(parents=True)
    output=export_osu(arranged,10,Path('vendor/mapperatorinator/template.osu'),
        original=(args.input/'planned.osu').read_text(encoding='utf-8-sig'), version='Source-aware experimental gestures')
    (args.output/'planned.osu').write_text(output,encoding='utf-8')
    def details(values):
        stats=validate(values,10);stats['left_objects']=sum(n.lane<5 for n in values)
        stats['right_objects']=len(values)-stats['left_objects']
        stats['layout_metrics']=layout_metrics(values,10,timing_points=original['timing'])
        return stats
    report=dict(input_events_sha256=sha(args.input/'learned-events.json'), previous_chart_sha256=sha(args.input/'planned.osu'),
        source_sketch_sha256=sha(root/'detected-onset-sketch-v1/sketch.osu'),
        chart_sha256=sha(args.output/'planned.osu'), runner_sha256=sha(__file__),
        before=details(original['notes']), after=details(arranged), intervals_preserved=True,
        enabled_in_application=False, musical_acceptance=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    human=parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
    start,end=215853,227853
    selected=lambda values:dict(notes=[n.data() for n in values if n.start<end and n.end>=start])
    write_review(args.output,root/'mapper-6664-26/accompaniment.wav',start,end,
        dict(finished=selected(human),before=selected(original['notes']),after=selected(arranged)),
        labels=dict(finished='作者成品',before='旧调用：将声部列号当作演奏位置',after='修正调用：声部身份与演奏位置分开'),
        sources={v:dict(label='Nova '+v,path=root/f'bundled/{v}/preview.wav') for v in ('drums','piano','other')},
        notice='相同的已冻结起止事件，仅重新分配轨道；这项检查不能改善采音或 LN 长度。未通过音乐验收。')
    page=args.output/'review.html'
    page.write_text(page.read_text(encoding='utf-8').replace('保留原始事件，未量化、未重新定轨。',
        '保持全部起止时间，只比较定轨调用修正前后。'),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
