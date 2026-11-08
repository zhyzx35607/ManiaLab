"""Plot aggregate LN counts and occupancy, without redistributing chart events."""
import argparse
import json
from pathlib import Path


def run(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    data=json.loads(source.read_text(encoding='utf-8'))
    selected=[(4201381,'学猫叫 · 7K'),(4438589,'大浪淘沙 · Easy'),(4035790,'大浪淘沙 · Meng'),
              (2473984,'夜曲 · 7K'),(5848413,'童话 · 10K（用户参考）')]
    rows=[next(r for r in data['rows'] if r['beatmap_id']==bid) for bid,_ in selected]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained');y=np.arange(len(rows))
    a=np.array([r['patterns']['ln_ratio'] for r in rows])*100
    b=np.array([r['patterns']['occupied_key_fraction'] for r in rows])*100
    ax.barh(y-.17,a,.32,label='LN 数量比例',color='#2b997d')
    ax.barh(y+.17,b,.32,label='全曲占轨比例',color='#5176b6')
    for pos,value in zip(y-.17,a):ax.text(value+.7,pos,f'{value:.1f}%',va='center',fontsize=10)
    for pos,value in zip(y+.17,b):ax.text(value+.7,pos,f'{value:.1f}%',va='center',fontsize=10)
    ax.set_yticks(y,[label for _,label in selected],fontproperties=font)
    ax.set_xlim(0,95);ax.invert_yaxis();ax.set_xlabel('百分比；两者分母不同，不是难度评分',fontproperties=font)
    ax.set_title('LN 数量相近，持续占轨可以很不同',fontproperties=font,pad=15)
    ax.legend(prop=font,loc='upper right');ax.spines[['top','right']].set_visible(False)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=150);plt.close(fig)
    summary=dict(selection_sha256=data['selection_sha256'],rows=[dict(beatmap_id=r['beatmap_id'],sha256=r['sha256'],
        keys=r['keys'],group=r['group'],ln_ratio=r['patterns']['ln_ratio'],occupancy=r['patterns']['occupied_key_fraction']) for r in rows])
    output.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
