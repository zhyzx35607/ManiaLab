"""Compare real autoregressive generation with/without a research adapter.

Same mixed audio, full reference TimingPoints, difficulty, LN condition and seed.
Reference hit objects never enter inference. No snapping or rearrangement is
applied. Similarity to one authored chart is diagnostic, not musical ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters
from manialab.domain import parse_osu, validate
from manialab.upstream import MapperEngine


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def event_similarity(reference, predicted, tolerances):
    """One-to-one maximum matching; columns intentionally do not affect timing."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    pairs = []
    if reference and predicted:
        delta = np.abs(np.asarray(reference)[:, None, :] - np.asarray(predicted)[None, :, :])
        allowed = (delta <= np.asarray(tolerances)).all(axis=2)
        # An extra match must always win over the combined timing tie-breaks.
        tie_break = (delta / np.asarray(tolerances)).mean(axis=2) / (len(reference) + 1)
        costs = np.concatenate((np.where(allowed, tie_break, 1e6), np.ones((len(reference), len(reference)))), axis=1)
        rows, cols = linear_sum_assignment(costs)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if j < len(predicted) and allowed[i, j]]
    count = len(pairs)
    precision = count / len(predicted) if predicted else 0.
    recall = count / len(reference) if reference else 0.
    errors = [[abs(reference[i][k] - predicted[j][k]) for i, j in pairs] for k in range(len(tolerances))]
    return dict(reference=len(reference), predicted=len(predicted), matched=count,
                precision=precision, recall=recall, f1=2*count/(len(reference)+len(predicted)) if reference or predicted else 0.,
                mean_absolute_error_ms=[sum(values)/len(values) if values else None for values in errors])


def timing_comparison(reference, predicted, start, end):
    heads = lambda notes: [(t,) for t in sorted({n.start for n in notes if start <= n.start < end})]
    tails = lambda notes: [(t,) for t in sorted({n.end for n in notes if n.hold and start <= n.end < end})]
    holds = lambda notes: [(n.start, n.end) for n in notes if n.hold and start <= n.start and n.end < end]
    return dict(onset_groups=event_similarity(heads(reference), heads(predicted), [30]),
                release_groups=event_similarity(tails(reference), tails(predicted), [60]),
                complete_hold_spans=event_similarity(holds(reference), holds(predicted), [30, 60]),
                notice='Similarity to one human interpretation, not instrument transcription accuracy or playability.')


def write_review(output, source, start, end, results, *, labels=None, sources=None, notice=None, keys=10):
    """Local shared-audio timeline; works without osu! or a server."""
    import re
    from manialab.audition import render_audition
    labels = labels or dict(reference='人工谱', baseline='原模型', adapter='实验微调')
    if not results or set(results)-set(labels):
        raise ValueError('Every chart requires an explicit provenance label')
    sources = {'mix': dict(label='原混音', path=source), **(sources or {})}
    if any(not re.fullmatch(r'[a-z][a-z0-9_-]*', name) for name in [*results, *sources]):
        raise ValueError('Invalid review artifact name')
    output.mkdir(parents=True, exist_ok=True)
    audio_sources = []
    for voice, track in sources.items():
        prefix = 'review' if voice == 'mix' else f'review-source-{voice}'
        original, info = render_audition(track['path'], start, end-start, [], music_volume=1., click_volume=0.)
        if abs(info['samples'] / info['sample_rate'] * 1000 - (end-start)) > 1:
            raise ValueError('Review sources must cover the same complete time interval')
        (output/f'{prefix}.wav').write_bytes(original.getvalue())
        files = {'music': f'{prefix}.wav'}
        for name, result in results.items():
            events = [(n['start'], 'head') for n in result['notes']]
            events += [(n['end'], 'tail') for n in result['notes'] if n['end'] > n['start']]
            preview, info = render_audition(track['path'], start, end-start, events)
            filename = f'{prefix}-{name}.wav'
            (output/filename).write_bytes(preview.getvalue())
            (output/f'{prefix}-{name}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
            files[name] = filename
        audio_sources.append(dict(name=voice, label=track['label'], files=files))
    charts = dict(start=start, end=end, keys=keys, sources=audio_sources,
        notice=notice or '人工谱只是其中一种创作；人工谱相似度不能代替音乐质量判断。',
        charts=[dict(name=name, label=labels[name], notes=result['notes'],
                     warnings=result.get('warnings', [])) for name, result in results.items()])
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mapper 原始生成对照</title>
<style>body{background:#10151f;color:#e7edf6;font:16px system-ui;margin:24px;line-height:1.6}audio{width:100%;max-width:900px}
canvas{display:block;width:100%;height:auto;background:#1b2433;border-radius:8px}h2{font-size:18px}.hint{color:#b4c3d8}
select{max-width:100%;font:inherit;background:#1b2433;color:inherit;padding:6px;border:1px solid #63758d;border-radius:5px}
.controls{display:flex;gap:16px;flex-wrap:wrap}.controls label{max-width:100%}.warning{color:#ffc88c;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}</style>
<h1>同一音频、同一 Timing：原始生成对照</h1>
<p class="hint">绿色为 LN，蓝色为短键。各图共用同一个播放时钟；点击谱图可跳转。保留原始事件，未量化、未重新定轨。</p>
<p id="notice" class="hint"></p>
<div class="controls"><label>声音 <select id="source"></select></label><label>起止提示 <select id="track"></select></label></div>
<p class="hint">高音提示按下，低音提示放开；提示与声音已混入同一音频。切换声部只改变试听声音，不改变模型输出或给音符自动标注乐器。</p>
<audio id="audio" controls src="review.wav"></audio><div id="charts"></div>
<script>const data=CHART_DATA; const audio=document.getElementById('audio');const canvases=[];
const sourceSelect=document.getElementById('source'),trackSelect=document.getElementById('track');
document.getElementById('notice').textContent=data.notice;
for(const s of data.sources)sourceSelect.add(new Option(s.label,s.name));
trackSelect.add(new Option('无提示','music'));for(const c of data.charts)trackSelect.add(new Option(c.label,c.name));
let pendingSeek=null;
function changeTrack(){const state=pendingSeek||{position:audio.currentTime,playing:!audio.paused};pendingSeek=state;
audio.pause();audio.src=data.sources.find(s=>s.name===sourceSelect.value).files[trackSelect.value];audio.load();}
audio.onloadedmetadata=()=>{if(!pendingSeek)return;const state=pendingSeek;pendingSeek=null;audio.currentTime=Math.min(state.position,audio.duration);if(state.playing)audio.play().catch(()=>{});};
sourceSelect.onchange=changeTrack;trackSelect.onchange=changeTrack;
for(const chart of data.charts){const title=document.createElement('h2');title.textContent=chart.label;
const c=document.createElement('canvas');c.width=1400;c.height=data.keys*18+25;c.style.height=c.height+'px';c.setAttribute('aria-label',chart.label+'音符时间轴');
document.getElementById('charts').append(title,c);canvases.push([c,chart]);
if(chart.warnings.length){const details=document.createElement('details'),summary=document.createElement('summary'),w=document.createElement('p');
const incomplete=chart.warnings.filter(s=>s.includes('Incomplete hold')).length;
details.className='warning';summary.textContent=incomplete?'模型报告 '+incomplete+' 条未闭合 LN · 查看原始日志':'查看运行提示';
w.textContent=chart.warnings.join('\\n');details.append(summary,w);document.getElementById('charts').append(details);}
c.onclick=e=>{const time=Math.max(0,Math.min(audio.duration,(e.clientX-c.getBoundingClientRect().left)/c.clientWidth*(data.end-data.start)/1000));if(pendingSeek)pendingSeek.position=time;else audio.currentTime=time;};}
function draw(){for(const[c,chart]of canvases){if(c.width!==Math.round(c.clientWidth))c.width=Math.round(c.clientWidth);const g=c.getContext('2d'),w=c.width;g.clearRect(0,0,w,c.height);
const x=t=>(t-data.start)/(data.end-data.start)*w;g.strokeStyle='#344258';g.fillStyle='#a9bbd2';g.font='13px system-ui';const bottom=data.keys*18+5;
const interval=w<600?2000:1000;for(let t=Math.ceil(data.start/interval)*interval;t<data.end;t+=interval){g.beginPath();g.moveTo(x(t),0);g.lineTo(x(t),bottom);g.stroke();g.fillText((t/1000).toFixed(0)+'s',x(t)+3,c.height-4);}
for(const n of chart.notes){g.fillStyle=n.end>n.start?'#6bd8b5':'#73aefb';g.fillRect(x(n.start),n.lane*18+4,Math.max(3,x(n.end)-x(n.start)),12);}
g.fillStyle='#fff';g.fillRect(x(data.start+audio.currentTime*1000),0,2,bottom);}requestAnimationFrame(draw);}draw();</script></html>'''
    (output/'review.html').write_text(html.replace('CHART_DATA', json.dumps(charts).replace('<', '\\u003c')), encoding='utf-8')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', type=Path, default=Path('test-data/mapper-training/mmrs'))
    cli.add_argument('--adapter', type=Path, default=Path('test-data/mapper-training/lora/adapter'))
    cli.add_argument('--engine-root', type=Path, default=Path('test-data'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--beatmap-id', type=int, default=4606010)
    cli.add_argument('--start-ms', type=int, default=24000)
    cli.add_argument('--seconds', type=int, default=12)
    cli.add_argument('--variants', nargs='+', choices=['baseline','adapter'], default=['baseline','adapter'])
    cli.add_argument('--conditioning', type=Path, help='JSON token flags; applied identically to baseline and adapter')
    cli.add_argument('--upstream-defaults', action='store_true', help='Explicitly reproduce the original inference prefix')
    args = cli.parse_args()
    if args.upstream_defaults and args.conditioning:
        raise ValueError('Choose declared training conditions or upstream defaults, not both')
    if args.conditioning and not args.conditioning.is_file():
        raise ValueError('Explicit conditioning file does not exist')
    condition_file=args.conditioning or args.adapter/'conditioning.json'
    conditioning=json.loads(condition_file.read_text(encoding='utf-8')) if not args.upstream_defaults and condition_file.exists() else None
    if not 1 <= args.seconds <= 30:
        raise ValueError('Audition probes must be 1–30 seconds')
    manifest = json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    row = next(r for r in manifest['records'] if r['beatmap_id'] == args.beatmap_id)
    if row['split'] != 'validation':
        raise ValueError('This probe requires a validation song')
    folder = args.dataset/'validation'/'data'/str(args.beatmap_id)
    audio, chart = folder/'audio.wav', folder/'reference.osu'
    if sha(audio) != row['decoded_audio_sha256'] or sha(chart) != row['prepared_chart_sha256']:
        raise ValueError('Validation pair changed')
    engine = MapperEngine(args.engine_root.resolve(), Path(__file__).resolve().parents[1])
    start, end = args.start_ms, args.start_ms+args.seconds*1000
    human = parse_osu(chart.read_text(encoding='utf-8'))['notes']
    args.output.mkdir(parents=True, exist_ok=True)
    selected = lambda notes: [n for n in notes if n.start < end and n.end >= start]
    results = dict(reference=dict(notes=[n.data() for n in selected(human)]))
    for name, adapter in [('baseline', None), ('adapter', args.adapter.resolve())]:
        if name not in args.variants:
            continue
        output = args.output/name
        saved_path = output/'result.json'
        identity = dict(audio=sha(audio), chart=sha(chart), adapter=sha(adapter/'adapter_model.safetensors') if adapter else None,
                        adapter_config=sha(adapter/'adapter_config.json') if adapter else None,
                        source_commit=engine.status()['source_commit'], model_revision=engine.status()['model_revision'],
                        start_ms=start, seconds=args.seconds, keys=row['keys'], seed=42, ln_ratio=row['ln_ratio'], difficulty=row['difficulty'])
        if conditioning is not None:
            identity['conditioning']=conditioning
        if saved_path.exists():
            saved = json.loads(saved_path.read_text(encoding='utf-8'))
            if saved['identity'] != identity or sha(output/saved['raw_file']) != saved['raw_sha256']:
                raise ValueError('Probe cache differs; use a new output directory')
        else:
            last = [0.]
            def progress(percent, message):
                if time.monotonic()-last[0] > 30:
                    print(json.dumps(dict(variant=name, message=message)), flush=True)
                    last[0] = time.monotonic()
            began = time.monotonic()
            result = engine.generate(audio, output, parameters(dict(keys=row['keys'], seed=42, ln_ratio=row['ln_ratio'])),
                progress, lambda: False, seconds=args.seconds, start_ms=start, timing_reference=chart,
                lora_path=adapter, difficulty=row['difficulty'], conditioning=conditioning)
            notes = parse_osu(result.read_text(encoding='utf-8-sig'))['notes']
            saved = dict(identity=identity, raw_file=result.relative_to(output).as_posix(), raw_sha256=sha(result),
                         elapsed_seconds=round(time.monotonic()-began, 2), warnings=engine.last_warnings,
                         validation=validate(notes, row['keys']), notes=[n.data() for n in selected(notes)],
                         comparison=timing_comparison(human, notes, start, end))
            saved_path.write_text(json.dumps(saved, indent=2), encoding='utf-8')
        results[name] = saved
        (args.output/'report.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(dict(variant=name, comparison=saved['comparison'], warnings=saved['warnings'])), flush=True)
    if 'baseline' in results and 'adapter' in results:
        write_review(args.output, audio, start, end, results, keys=row['keys'])


if __name__ == '__main__':
    main()
