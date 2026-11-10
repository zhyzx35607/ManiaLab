"""Describe repeated attacks before choosing their chart expression.

This is an editable arrangement policy, not finger-release transcription or
melody/hand separation. Never infer new attacks or fabricate a release time.
"""
from __future__ import annotations

from statistics import median
from collections import defaultdict

from .domain import UNOBSERVED_RELEASES


def _overlap_pairs(groups):
    overlapping=0
    for group,next_group in zip(groups,groups[1:]):
        ends=[n.end for n in group if n.source=='pitch' and n.end>n.start
              and n.release_reason not in UNOBSERVED_RELEASES]
        overlapping+=bool(ends and median(ends)>next_group[0].start+80)
    return overlapping


def attack_runs(notes):
    """Group attacks, respecting gaps between observed source intervals.

    This is a gap in candidate coverage, not proof of acoustic silence. Keep
    earlier overlapping tails in view; an uncertain end cannot establish a
    new interval gap within the current onset-based context. Split only runs
    already subject to short-key expression, so protection cannot add taps.
    """
    groups=[]
    for note in sorted(notes,key=lambda n:(n.start,n.pitch,n.id)):
        if not groups or note.start-groups[-1][0].start>30:
            groups.append([note])
        else:
            groups[-1].append(note)
    gaps=[b[0].start-a[0].start for a,b in zip(groups,groups[1:])]
    pause=max(500.,min(1600.,median(gaps)*2.5)) if gaps else 500.
    onset_runs=[]
    for group in groups:
        if not onset_runs or group[0].start-onset_runs[-1][-1][0].start>=pause:
            onset_runs.append([group])
        else:
            onset_runs[-1].append(group)
    runs=[]
    latest_end=None
    for original in onset_runs:
        articulated=len(original)>=3 and _overlap_pairs(original)*2>=len(original)-1
        complete_intervals=True
        for index,group in enumerate(original):
            interval_gap=(articulated and latest_end is not None and complete_intervals
                          and group[0].start-latest_end>=80)
            if index==0 or interval_gap:
                runs.append([group])
            else:
                runs[-1].append(group)
            latest_end=max(latest_end if latest_end is not None else group[0].start,
                           max(n.end for n in group))
            complete_intervals=complete_intervals and all(
                n.source=='pitch' and n.end>n.start and n.release_reason not in UNOBSERVED_RELEASES
                for n in group)
    return runs,pause


def articulation_policy(notes, *, observed_until_ms=None):
    """Describe attacks with full source context, separately from chart sections.

    An observed gap is absence of another detected attack, not proof of a
    musical phrase ending. Unknown right context cannot establish a run tail.
    Callers omitting the observation boundary retain the legacy tap policy.
    """
    streams=defaultdict(list)
    for note in notes:
        streams[(note.voice,note.instrument_id)].append(note)
    if len(streams)>1:
        result={}
        for stream in streams.values():
            result.update(articulation_policy(stream,observed_until_ms=observed_until_ms))
        return result
    runs,pause=attack_runs(next(iter(streams.values()), ()))
    result={}
    for run_index,groups in enumerate(runs):
        overlapping=_overlap_pairs(groups)
        articulated=len(groups)>=3 and overlapping*2>=len(groups)-1
        for group_index,group in enumerate(groups):
            if group_index<len(groups)-1:
                continuation='next_attack'
            elif run_index<len(runs)-1:
                next_start=runs[run_index+1][0][0].start
                continuation=('observed_gap' if next_start-group[0].start>=pause
                              else 'observed_interval_gap')
            elif (observed_until_ms is not None
                    and observed_until_ms-groups[-1][0].start>=pause):
                continuation='observed_gap'
            else:
                continuation='unknown'
            body=articulated and (group_index<len(groups)-1 or
                (observed_until_ms is not None and continuation=='unknown'))
            for note in group:
                result[note.id]=dict(tap=body,run=run_index,run_start=groups[0][0].start,
                    run_end_attack=groups[-1][0].start,attack_groups=len(groups),
                    overlap_pairs=overlapping,pause_threshold_ms=pause,
                    continuation=continuation,
                    articulation='attack_run_open' if articulated and body and continuation=='unknown' else
                        'attack_run_body' if body else
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
