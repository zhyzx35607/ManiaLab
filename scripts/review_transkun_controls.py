"""Build a complete audio/timeline comparison for the frozen 91 controls."""
import argparse
import html
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transkun_runtime import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def timeline(expected,old,new,old_score,new_score):
    """Compare event onsets; dashed tails are predictions, not release labels."""
    pitches=sorted({r['pitch'] for r in [*expected,*old,*new]})
    duration=max(10.,max((e['offset'] for e in [*old,*new]),default=0.))
    width,height=1060,max(130,55+len(pitches)*50)
    x=lambda t: 65+t/duration*(width-85)
    y=lambda pitch: 35+pitches.index(pitch)*50
    pieces=[f'<svg viewBox="0 0 {width} {height}" role="img" '
            'aria-label="两模型钢琴事件对照；黑色标记为加入声源">']
    if duration>10:
        pieces.append(f'<rect x="{x(10):.3f}" y="15" width="{x(duration)-x(10):.3f}" '
                      f'height="{height-40}" fill="#fff0df"/>')
    for t in range(math.ceil(duration)+1):
        if t>duration:continue
        pieces.append(f'<path d="M{x(t):.3f},15 V{height-25}" stroke="#e0e4e9"/>'
                      f'<text x="{x(t):.3f}" y="{height-6}" text-anchor="middle">{t}</text>')
    for pitch in pitches:
        pieces.append(f'<text x="12" y="{y(pitch)+4}">{pitch}</text>')
    for events,score,shift,color,label in [(old,old_score,0,'#466fd5','Kong'),
                                          (new,new_score,16,'#09845c','Transkun')]:
        unmatched=set(score['extra_pred_indices'])
        for index,event in enumerate(events):
            yy=y(event['pitch'])+shift
            start,end=x(event['onset']),x(event['offset'])
            title=html.escape(f'{label} pitch {event["pitch"]}: {event["onset"]:.6f}–{event["offset"]:.6f}s')
            pieces.append(f'<g><title>{title}</title><path d="M{start:.3f},{yy} H{end:.3f}" '
                          f'stroke="{color}" stroke-dasharray="3,3"/>')
            if index in unmatched:
                pieces.append(f'<path d="M{start-3:.3f},{yy-3} l6,6 m0,-6 l-6,6" stroke="#b61e35"/>')
            else:
                pieces.append(f'<circle cx="{start:.3f}" cy="{yy}" r="3" fill="{color}"/>')
            pieces.append('</g>')
    for event in expected:
        xx,yy=x(event['anchor_seconds']),y(event['pitch'])
        pieces.append(f'<path d="M{xx:.3f},{yy-8} V{yy+24}" stroke="#101820" stroke-width="2"/>')
    pieces.append('</svg>')
    return ''.join(pieces)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    run,output=args.run.resolve(),args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local test-data directory')
    report=read(run/'report.json')
    if report['unique_control_inferences']!=91 or not report['wrapper_verification']['identical']:
        raise ValueError('Expected a completed verified run')
    for name,digest in report['artifacts_sha256'].items():
        if sha256(run/name)!=digest:
            raise ValueError('Run artifact changed')
    plans=read(run/'plans.json')
    output.mkdir(parents=True)
    shutil.copyfile(__file__,output/'review-snapshot.py')
    tables,cards,rows=[],[],[]
    for plan in plans:
        name=plan['name']
        if sha256(plan['file'])!=plan['wav_sha256']:
            raise ValueError('Input audio changed')
        result=read(run/(name+'-result.json'))
        baseline=read(Path(plan['baseline_result']))
        old,new=baseline['events'],result['trace']['piano_events']
        a,b=baseline['score'],result['score']
        new_missing=set(b['missing_ref_indices'])-set(a['missing_ref_indices'])
        recovered=set(a['missing_ref_indices'])-set(b['missing_ref_indices'])
        diff='有新漏音' if new_missing else '找回起音' if recovered else '匹配数相同'
        safe=html.escape(name)
        tables.append(f'<tr><td><a href="#{safe}">{safe}</a></td><td>{b["expected"]}</td>'
                      f'<td>{a["matched"]} / {a["extra"]}</td><td>{b["matched"]} / {b["extra"]}</td>'
                      f'<td>{diff}</td></tr>')
        notes=dict(name=name,expected=result['expected'],kong=old,transkun=new,
                   pedals=result['trace']['pedal_events'],kong_score=a,transkun_score=b)
        (output/(name+'.json')).write_text(json.dumps(notes,indent=2,allow_nan=False),encoding='utf-8')
        shutil.copyfile(plan['file'],output/(name+'.wav'))
        diagrams=timeline(result['expected'],old,new,a,b)
        cards.append(f'<section id="{safe}"><h2>{safe}</h2><p>{diff}；新增踏板控制事件 '
                     f'{len(notes["pedals"])} 个（单列，不计钢琴音符）。</p>'
                     f'<audio controls preload="none" src="{safe}.wav"></audio> '
                     f'<a href="{safe}.json">完整事件与匹配</a>{diagrams}</section>')
        rows.append(dict(name=name,kong_matched=a['matched'],transkun_matched=b['matched'],
                         kong_extra=a['extra'],transkun_extra=b['extra'],
                         new_missing=len(new_missing),recovered=len(recovered)))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>钢琴声学模型对照</title>
<style>body{font:16px system-ui;max-width:1160px;margin:auto;padding:20px;background:#fafafa;color:#18202c}
section{padding:14px;border:1px solid #d8dee5;margin:24px 0;background:white}svg{width:100%;font:12px system-ui}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}
.table{overflow:auto}audio{max-width:100%;vertical-align:middle}a{color:#234d91}h2{overflow-wrap:anywhere}
</style><h1>同一批声音，两种钢琴模型</h1>
<p>全部91例均列出。表格为“匹配 / 额外预测”，不是音乐质量评分。黑色竖线是加入声源的包络锚点，
蓝色为Kong、绿色为Transkun；红叉是未匹配预测。虚线音尾仅为模型输出，两模型音尾语义不同，不能当作LN放手比较。</p>
<p>每个音频保留实际原控制音量。图的横轴为物理秒，音高相同时上行为Kong、下行为Transkun。
输入超过10秒的区域用浅橙色显示，所有预测均保留。本页面提供人工回放入口，未声称已完成听感验收。</p>
<div class="table"><table><tr><th>输入</th><th>加入数</th><th>Kong</th><th>Transkun</th><th>匹配变化</th></tr>
'''+''.join(tables)+'</table></div>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    saved=dict(source_report_sha256=sha256(run/'report.json'),runner_sha256=sha256(Path(__file__)),
               cases=rows,offsets_scored=False,direct_listening_performed=False,
               artifacts_sha256={p.name:sha256(p) for p in output.iterdir()})
    (output/'report.json').write_text(json.dumps(saved,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(cases=len(rows),page=str(output/'index.html'))),flush=True)


if __name__=='__main__':
    main()
