"""Render fixed audio intervals and local connection hypotheses; no new inference."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.io import wavfile
from scripts.build_provenance import sha256

MODELS=('original','clean_adapted','overlap_adapted')
LABELS=('Original weights','Clean-adapted weights','Overlap-adapted weights')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def edge_visibility(edges,physical,begin,end):
    """Account for snapped-core membership displayed on the raw audio clock."""
    result=dict(fully_visible=0,boundary_clipped=0,outside_clip=0)
    for a,b in edges:
        first,last=physical[a]['start']/1000,physical[b]['start']/1000
        if max(first,last)<begin or min(first,last)>=end:
            result['outside_clip']+=1
        elif begin<=first<end and begin<=last<end:
            result['fully_visible']+=1
        else:
            result['boundary_clipped']+=1
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--plot-dependencies',type=Path,default=ROOT/'test-data/audio-review-deps')
    args=cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior reviews')
    sys.path.insert(0,str(args.plot_dependencies.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    report=read(args.run/'report.json')
    for name,digest in report['artifacts_sha256'].items():
        if sha256(args.run/name)!=digest:
            raise ValueError('Saved inference artifact changed')
    source=read(args.run/'source-map.json')
    physical={n['id']:n['physical'] for n in source}
    graph={n['id']:n['graph'] for n in source}
    predictions={model:read(args.run/(model+'-predictions.json')) for model in MODELS}
    path=ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav'
    if sha256(path)!=report['identity']['pcm_sha256']:
        raise ValueError('PCM differs from fixed inference selection')
    rate,pcm=wavfile.read(path)
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'renderer-snapshot.py')
    cards=[];clips=[]
    for selection in report['selection']:
        chunk=selection['chunk'];name=f'core-{chunk:03d}'
        left=max(0,round(selection['begin_ms']*rate/1000));right=min(len(pcm),round(selection['end_ms']*rate/1000))
        begin,end=left/rate,right/rate
        signal=pcm[left:right]
        if not len(signal):
            raise ValueError('Empty chosen audio')
        gain=min(1.,.95/max(float(np.abs(signal).max()),1e-12))
        rendered=np.rint(signal.astype(np.float64)*gain*32767).astype(np.int16)
        wavfile.write(args.output/(name+'.wav'),rate,rendered)
        check_rate,check=wavfile.read(args.output/(name+'.wav'))
        assert check_rate==rate and np.array_equal(check,rendered)
        chosen=[n for n in physical.values() if n['start']/1000<end and n['end']/1000>begin]
        core_ids={r['id'] for r in graph.values() if r['tick']//7680==chunk}
        edge_sets={model:{tuple(pair) for pair in predictions[model]['local_edges'] if pair[1] in core_ids}
                   for model in MODELS}
        visibility={model:edge_visibility(edges,physical,begin,end) for model,edges in edge_sets.items()}
        low=min((n['pitch'] for n in chosen),default=40)-3
        high=max((n['pitch'] for n in chosen),default=90)+3
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
        fig,axes=plt.subplots(4,1,figsize=(16,11),sharex=True,sharey=True)
        for panel,ax in enumerate(axes):
            for note in chosen:
                x=max(begin,note['start']/1000);stop=min(end,note['end']/1000)
                ax.add_patch(Rectangle((x,note['pitch']-.25),stop-x,.5,
                                      facecolor='#94a3b8',edgecolor='none',alpha=.45))
                if begin<=note['start']/1000<end:
                    ax.plot(note['start']/1000,note['pitch'],'|',color='#111827',ms=5)
            if panel:
                edges=edge_sets[MODELS[panel-1]]
                for a,b in sorted(edges):
                    first,last=physical[a],physical[b]
                    ax.plot([first['start']/1000,last['start']/1000],[first['pitch'],last['pitch']],
                            color='#087f8c',alpha=.30,lw=.8)
                visible=visibility[MODELS[panel-1]]
                title=(f'{LABELS[panel-1]} | {len(edges)} same-bar positive edges (hypotheses)'
                       f' | clipped {visible["boundary_clipped"]}, outside {visible["outside_clip"]}')
            else:
                title='Frozen original acoustic intervals | endpoints are not authored LN releases'
            ax.set_title(title,loc='left');ax.set_ylabel('MIDI pitch');ax.set_ylim(low,high)
            ax.set_xlim(begin,end);ax.grid(alpha=.15)
        axes[-1].set_xlabel('Original audio time (s); nodes use raw candidate onsets, not snapped display positions')
        fig.suptitle(f'Sakura Nova piano | core {chunk} | same input, three weight sets',x=.06,ha='left')
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(args.output/(name+'.png'),dpi=130);plt.close(fig)
        union=set().union(*edge_sets.values())
        rows=[]
        for a,b in sorted(union,key=lambda pair:(physical[pair[0]]['start'],physical[pair[1]]['start'],pair)):
            first,last=physical[a],physical[b]
            cells=''.join('<td>'+('✓' if (a,b) in edge_sets[m] else '—')+'</td>' for m in MODELS)
            rows.append(f'<tr><td>{first["start"]/1000:.3f}s / {first["pitch"]}</td>'
                        f'<td>{last["start"]/1000:.3f}s / {last["pitch"]}</td>{cells}</tr>')
        row=dict(chunk=chunk,first_sample=left,stop_sample=right,sample_rate=rate,source_gain=gain,
            source_intervals_overlapping=len(chosen),source_attacks_in_clip=sum(begin<=n['start']/1000<end for n in chosen),
            positive_edges={m:len(v) for m,v in edge_sets.items()},
            edge_visibility=visibility,
            changed_from_original={m:len(edge_sets[m]^edge_sets['original']) for m in MODELS[1:]})
        clips.append(row)
        cards.append(f'''<section><h2>{begin:.3f}–{end:.3f} 秒 · Core {chunk}</h2>
<audio controls preload="metadata" src="{name}.wav"></audio>
<p>回放为同一条原Nova钢琴轨。灰色是原采音区间，黑色短线是原起音，青线是模型提出的同小节连接。三张连接图没有重新采音或改变音符。</p>
<p>连接按量化小节归属，实际起音可能略越过片段边界；图中注明被裁剪和完全位于片段外的连接数，表格保留全部归属连接。</p>
<img src="{name}.png" alt="原音符和三权重局部连接对照">
<details><summary>查看全部局部连接假设（{len(union)} 对）</summary><table>
<thead><tr><th>来源时间 / MIDI</th><th>目标时间 / MIDI</th><th>原模型</th><th>Clean适配</th><th>重叠适配</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></details></section>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ManiaLab · 钢琴局部连接对照</title><style>
body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;background:#f5f7fb;color:#152137}
section{background:white;padding:24px;margin:24px 0;border:1px solid #dce3ee;border-radius:12px}img{width:100%;height:auto}audio{width:min(650px,100%)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:6px 12px;border-bottom:1px solid #e2e8f0;text-align:left}details{margin-top:16px}
.note{padding:16px 20px;background:#fff3d6;border-radius:8px}summary{cursor:pointer}</style>
<h1>同一段钢琴，三个声部连接模型</h1>
<p class="note">这些是需要审阅的局部连接假设，不是正确声部标签，也不是生成的LN。跨小节分数未验证，因此没有画成旋律线。
模型输入沿用人工74 BPM / Offset178ms / 22.5ms提前量的量化诊断坐标；图中音符仍用原始采音时间。此页面不能证明自动对音正确。</p>
<p>按原音源时长的1/8、3/8、5/8、7/8于推理前选段，并对重复片段去重。原始音轨回放、物理音符区间和三模型结果分开呈现；没有声部真值，不能把连线多或少当成优劣。</p>
'''+f'<p>实际展示 {len(clips)} 个片段。</p>'+''.join(cards)+'</html>'
    (args.output/'index.html').write_text(page,encoding='utf-8')
    result=dict(inference_report_sha256=sha256(args.run/'report.json'),renderer_sha256=sha256(__file__),
                clips=clips,model_rerun=False,manual_listening_performed=False,musical_acceptance=False,
                libraries=dict(numpy=np.__version__,matplotlib=matplotlib.__version__))
    result['artifacts_sha256']={p.name:sha256(p) for p in args.output.iterdir()}
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
