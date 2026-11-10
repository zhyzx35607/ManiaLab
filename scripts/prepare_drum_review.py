"""Prepare fixed, shared-clock drum excerpts for source-by-source inspection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import sha, write_review
from scripts.research_source_attack_classes import CLASSES, drum_event_counts


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();root=args.root
    if args.output.exists():raise ValueError('Preserve prior review')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    separation=root/'drums-larsnet-v1';raw=root/'drums-larsnet-onsets-v1';selected=root/'drums-larsnet-selection-v1'
    report=read(separation/'report.json');raw_report=read(raw/'report.json');selected_report=read(selected/'report.json')
    if sha(raw/'events.json')!=raw_report['events_sha256'] or sha(selected/'events.json')!=selected_report['events_sha256']:
        raise ValueError('Saved onset output changed')
    reference=root/'reference.osu'
    if sha(reference)!=raw_report['reference_sha256']:raise ValueError('Reference changed')
    for role,info in report['stems'].items():
        if sha(separation/(role+'.wav'))!=info['sha256']:raise ValueError('Separated audio changed')
    if sha(separation/'input-float.wav')!=report['identity']['input_float_sha256']:raise ValueError('Separation input changed')
    def note(t,c):return dict(start=float(t),end=float(t),lane=c)
    human=[note(t+22.5,CLASSES.index(c)) for t,c in drum_event_counts(parse_osu(reference.read_text(encoding='utf-8-sig'))['notes'])]
    predicted=[note(e['start_ms'],{'kick':0,'snare':1,'hihat':2}[e['kind']]) for e in read(raw/'events.json') if e['kind'] in ('kick','snare','hihat')]
    selection=[note(e['start_ms']+22.5,CLASSES.index(e['kind'])) for e in read(selected/'events.json')]
    with np.load(root/'drum-mixture-v1/linear-float-control.npz',allow_pickle=False) as data:
        baseline=[note(t,c) for i,t in enumerate(data['times']) for c in range(3) if data['evidence'][i,c]>=data['thresholds'][c]]
    names=dict(kick='模型kick轨',snare='模型snare轨',toms='模型tom轨',hihat='模型hi-hat轨',cymbals='模型镲片轨')
    sources={'mix':dict(label='原Nova鼓轨',path=separation/'input-float.wav'),
        **{role:dict(label=label,path=separation/(role+'.wav')) for role,label in names.items()}}
    for tag,start in [('early',6664),('later',215853)]:
        end=start+12000
        results={name:dict(notes=[n for n in notes if start<=n['start']<end]) for name,notes in
            [('reference',human),('linear',baseline),('separated',predicted),('selected',selection)]}
        write_review(args.output/tag,separation/'input-float.wav',start,end,results,
            labels=dict(reference='作者选择（仅起音）',linear='原float局部分类',separated='分离后直接采音',selected='分离证据学习取舍（失败对照）'),
            sources=sources,keys=3,title='鼓声采音逐轨对照',
            intro='三行依次为kick、snare、作者第4列。这里只检查起音，军鼓双押合并；不表示可玩谱面或LN。各图共用播放时钟，点击可跳转。',
            notice='作者第4列包含hi-hat、tom等，直接采音行只显示模型hi-hat类，不能视为完整第4列真值。作者起音整体加22.5ms用于原音频时钟对照，模型起音保持原物理时间；没有拍线吸附。模型轨名不是已经核实的音色，切换音轨可检查串音。')
    index='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>鼓声对照索引</title><style>body{font:18px system-ui;background:#10151f;color:#eee;margin:40px;line-height:1.8}a{color:#8fcaff}</style><h1>鼓声采音对照</h1><p>固定前段和后段各12秒，可在原鼓轨与五条模型分离轨间切换。失败结果保留，用于定位漏音、串音和作者取舍差异。</p><p><a href="early/review.html">前段：6.664–18.664秒</a></p><p><a href="later/review.html">后段：215.853–227.853秒</a></p></html>'''
    (args.output/'index.html').write_text(index,encoding='utf-8')
    print(str((args.output/'index.html').resolve()),flush=True)


if __name__=='__main__':main()
