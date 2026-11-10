"""Describe authored expression without assigning ambiguous LN sources."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, sections
from scripts.probe_mapper_adapter import sha


def describe(reference, finished, tolerance=30):
    sources=defaultdict(set)
    for note in reference:
        sources[note.start].add('drums' if note.lane<4 else 'piano' if note.lane<6 else 'other')
    times=sorted(sources)
    def nearby(time):
        return times[bisect_left(times,time-tolerance):bisect_right(times,time+tolerance)]
    heads=Counter();same_source=0;head_matched=tail_matched=both_matched=0
    holds=[n for n in finished if n.hold]
    for note in holds:
        head_times,tail_times=nearby(note.start),nearby(note.end)
        roles=set().union(*(sources[t] for t in head_times)) if head_times else set()
        end_roles=set().union(*(sources[t] for t in tail_times)) if tail_times else set()
        heads['+'.join(sorted(roles)) or 'unmatched']+=1
        head_matched+=bool(head_times);tail_matched+=bool(tail_times)
        both_matched+=bool(head_times and tail_times)
        same_source+=bool(roles&end_roles)
    releases=Counter(n.end for n in holds)
    return dict(reference_objects=len(reference),reference_lns=sum(n.hold for n in reference),
        finished_objects=len(finished),finished_lns=len(holds),tolerance_ms=tolerance,
        ln_head_candidate_sources=dict(heads),ln_heads_near_reference_attacks=head_matched,
        ln_tails_near_reference_attacks=tail_matched,both_endpoints_near_reference_attacks=both_matched,
        endpoints_with_any_common_candidate_source=same_source,
        release_group_multiplicity=dict(sorted(Counter(releases.values()).items())),
        lns_in_shared_release_groups=sum(count for count in releases.values() if count>=2),
        note='Candidate sources are ambiguous when instruments coincide. Musical release alignment is not proof of acoustic duration or source identity.')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    args=cli.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    maps={r:parse_osu((root/(r+'.osu')).read_text(encoding='utf-8-sig'))['notes'] for r in ('reference','finished')}
    result=describe(maps['reference'],maps['finished'])
    result['chart_hashes']={r:sha(root/(r+'.osu')) for r in maps}
    result['sections']=[dict(start_ms=s['start_ms'],end_ms=s['end_ms'],
        **describe(*[[n for n in maps[r] if s['start_ms']<=n.start<s['end_ms']] for r in ('reference','finished')])) for s in manifest['segments']]
    (root/'expression-study.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sections'}),flush=True)


if __name__=='__main__':main()
