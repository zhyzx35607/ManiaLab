"""Describe repeated attacks before choosing their chart expression.

This is an editable arrangement policy, not finger-release transcription or
melody/hand separation. Never infer new attacks or fabricate a release time.
"""
from __future__ import annotations

from statistics import median

from .domain import UNOBSERVED_RELEASES


def attack_runs(notes):
    """Group source-local attacks without chaining nearly simultaneous notes."""
    groups=[]
    for note in sorted(notes,key=lambda n:(n.start,n.pitch,n.id)):
        if not groups or note.start-groups[-1][0].start>30:
            groups.append([note])
        else:
            groups[-1].append(note)
    gaps=[b[0].start-a[0].start for a,b in zip(groups,groups[1:])]
    pause=max(500.,min(1600.,median(gaps)*2.5)) if gaps else 500.
    runs=[]
    for group in groups:
        if not runs or group[0].start-runs[-1][-1][0].start>=pause:
            runs.append([group])
        else:
            runs[-1].append(group)
    return runs,pause


def articulation_policy(notes):
    """Return expression choices keyed by original ID; do not mutate notes."""
    runs,pause=attack_runs(notes)
    result={}
    for run_index,groups in enumerate(runs):
        overlapping=0
        for group,next_group in zip(groups,groups[1:]):
            ends=[n.end for n in group if n.source=='pitch' and n.end>n.start
                  and n.release_reason not in UNOBSERVED_RELEASES]
            overlapping+=bool(ends and median(ends)>next_group[0].start+80)
        articulated=len(groups)>=3 and overlapping*2>=len(groups)-1
        for group_index,group in enumerate(groups):
            body=articulated and group_index<len(groups)-1
            for note in group:
                result[note.id]=dict(tap=body,run=run_index,run_start=groups[0][0].start,
                    run_end_attack=groups[-1][0].start,attack_groups=len(groups),
                    overlap_pairs=overlapping,pause_threshold_ms=pause,
                    articulation='attack_run_body' if body else
                        'attack_run_tail' if articulated else 'independent_sustain')
    return result


def coverage(intervals,begin,end):
    """Exact union occupancy, clipped to a section rather than coarse bins."""
    stop=begin;total=0
    for left,right in sorted((max(begin,a),min(end,b)) for a,b in intervals if b>begin and a<end):
        if right>max(left,stop):total+=right-max(left,stop)
        stop=max(stop,right)
    return total/max(1,end-begin)


def source_activity(notes,begin,end):
    attacks=[(n.start,n.start+250) for n in notes]
    reliable=[(n.start,n.end) for n in notes if n.source=='pitch' and n.end>n.start
              and n.release_reason not in UNOBSERVED_RELEASES]
    return dict(activity_coverage=coverage(attacks+reliable,begin,end),
        attack_coverage=coverage(attacks,begin,end),observed_interval_coverage=coverage(reliable,begin,end),
        uncertain_tails=sum(n.end>n.start and n.release_reason in UNOBSERVED_RELEASES for n in notes),
        candidate_count=len(notes))
