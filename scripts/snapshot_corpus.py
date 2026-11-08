"""Merge frozen local chart inventories into a verifiable training snapshot.

Later roots supersede earlier IDs, including exclusions. Only native 6-10K
Ranked/Approved/Loved charts enter this core snapshot. Exact layouts are deduped
within keycount; all revisions and rejection reasons remain in the audit report.
Downloaded originals are never changed and must not be redistributed.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu,validate


def build(roots, output):
    if (output/'manifest-snapshot.jsonl').exists():
        raise ValueError('Use a new snapshot directory; do not overwrite an experiment')
    selected={};versions=0;aliases=defaultdict(lambda:dict(songs=set(),audio=set()));official={}
    for root in roots:
        for path in sorted((root/'official').glob('*.json')):
            value=json.loads(path.read_text(encoding='utf-8'))
            official[value['id']]={d['id']:d for d in value.get('beatmaps',[])}
        for path in sorted(root.glob('manifest-*.jsonl')):
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                row=json.loads(line);versions+=1
                bid=int(row['beatmap_id'])
                selected[bid]=(row,root/'maps'/f'{bid}.osu')
                if row.get('song_key'):aliases[bid]['songs'].add(row['song_key'])
                if row.get('audio_hash'):aliases[bid]['audio'].add(row['audio_hash'])
                aliases[bid]['songs'].update(row.get('related_song_keys',[]))
                aliases[bid]['audio'].update(row.get('related_audio_hashes',[]))
    output.mkdir(parents=True,exist_ok=True);(output/'maps').mkdir(exist_ok=True)
    included=[];excluded=[];seen={}
    for index,(bid,(row,path)) in enumerate(sorted(selected.items())):
        if index%250==0:print(json.dumps(dict(stage='snapshot',visited=index,total=len(selected),included=len(included))),flush=True)
        reason=None
        try:
            difficulty=None
            if row.get('beatmapset_id') in official:
                difficulty=official[row['beatmapset_id']].get(bid,{})
                if difficulty.get('status') not in ('ranked','approved','loved') or difficulty.get('convert'):
                    raise ValueError('official_difficulty_status_excluded')
            if row.get('approved')==4 and not difficulty and row.get('status_scope')!='beatmap':
                raise ValueError('loved_difficulty_status_unverified')
            # A labelled inverse/converted chart can still teach lane geometry.
            # Keep that role explicit instead of using its tails as evidence
            # of how long an instrument really sustained.
            layout_only=bool(row.get('suspected_conversion'))
            if (not row.get('eligible') and not layout_only) or row.get('approved') not in (1,2,4):
                raise ValueError('ineligible_or_not_ranked_loved')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('hash_mismatch')
            expected_md5=difficulty.get('checksum') if difficulty else row.get('official_md5')
            if expected_md5 and hashlib.md5(raw).hexdigest()!=expected_md5:
                raise ValueError('official_checksum_mismatch')
            chart=parse_osu(raw.decode('utf-8-sig'));keys=chart['keys']
            if not 6<=keys<=10 or keys!=row['keys']:raise ValueError('keycount')
            report=validate(chart['notes'],keys)
            if not report['valid'] or len(chart['notes'])<100:raise ValueError('structure_or_placeholder')
            label=' '.join(str(row.get(k,'')) for k in ('title','version','tags')).casefold()
            if keys==10 and re.search(r'\b(?:4k?\s*\+\s*6k?|6k?\s*\+\s*4k?)\b',label):raise ValueError('asymmetric_10k')
            if keys==8 and re.search(r'\b(?:7k?\s*\+\s*1k?|1k?\s*\+\s*7k?|iidx)\b',label):raise ValueError('scratch_8k')
            layout=hashlib.sha256(json.dumps([keys,sorted((n.start,n.end,n.lane) for n in chart['notes'])]).encode()).hexdigest()
            if layout in seen:raise ValueError(f'duplicate_layout:{seen[layout]}')
            seen[layout]=bid
            (output/'maps'/f'{bid}.osu').write_bytes(raw)
            row=dict(row,eligible=True,reference_role='layout_only' if layout_only else 'expression_and_layout',
                     related_song_keys=sorted(aliases[bid]['songs']),related_audio_hashes=sorted(aliases[bid]['audio']),
                     layout_sha256=layout,report=report,snapshot_source=str(path.parent.parent),
                     ln_class='ln_majority' if report['ln']/report['notes']>=.5 else 'hybrid' if report['ln']/report['notes']>=.15 else 'tap_majority')
            included.append(row)
        except (ValueError,OSError) as error:reason=str(error)
        if reason:excluded.append(dict(id=bid,reason=reason))
    manifest=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in included)
    (output/'manifest-snapshot.jsonl').write_text(manifest,encoding='utf-8',newline='\n')
    report=dict(protocol=__doc__,source_records=versions,unique_ids=len(selected),included=len(included),
        keycounts=dict(Counter(str(r['keys']) for r in included)),
        reference_roles=dict(Counter(r['reference_role'] for r in included)),
        key_ln_classes={str(k):dict(Counter(r['ln_class'] for r in included if r['keys']==k)) for k in range(6,11)},
        exclusions=dict(Counter(r['reason'].split(':')[0] for r in excluded)),excluded=excluded,
        manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('protocol','excluded')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.roots,a.output)
