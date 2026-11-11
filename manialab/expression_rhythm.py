"""Explicit, constant-clock rhythmic subsets for expression research.

This is an arrangement choice, not an accompaniment or melody detector. It
does not regroup candidates, score them, or alter note/LN endpoints. Callers
apply these decisions within their original complete selection families.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math

from .timing import DIVISIONS


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _scope(section, voice, instrument_id):
    if (type(section) is not int or section < 0 or
            not isinstance(voice, str) or not voice or
            not isinstance(instrument_id, str)):
        raise ValueError('节奏子集范围必须包含有效段落、声部和乐器标识')
    return section, voice, instrument_id


def plan_rhythm(candidates, metadata, *, clock=None, confirmed=False, policies=()):
    """Return candidate decisions and an explicit-policy report, or ({}, None).

    A policy fixes one ``stride`` and ``phase`` for (section, voice,
    instrument_id). Tick zero is the supplied chart offset, never a section
    or a group's first note. The clock and candidates must already share the
    chart time convention; audio_shift_ms is deliberately not applied here.
    Nonempty timing_points are unsupported, including mixed red/green lines.

    Only the scoped candidates need to be on this grid. An omitted onset is
    an arrangement omission, not an attack represented by another LN. No
    fallback note is invented when the requested subset is empty.
    """
    if not isinstance(policies, (list, tuple)):
        raise ValueError('节奏子集策略必须是列表')
    if not policies:
        return {}, None
    if confirmed is not True or not isinstance(clock, Mapping):
        raise ValueError('节奏子集需要明确确认的恒速拍格')
    points = clock.get('timing_points')
    if points is not None and not (isinstance(points, (list, tuple)) and not points):
        raise ValueError('节奏子集只支持显式恒速拍格，不接受 timing_points')
    bpm, offset, division = (clock.get(key) for key in ('bpm', 'offset', 'division'))
    if not _number(bpm) or not 10 <= bpm <= 1000:
        raise ValueError('节奏子集 BPM 必须在10至1000之间')
    if not _number(offset) or abs(offset) > 600000:
        raise ValueError('节奏子集 Offset 必须是有效毫秒数')
    if type(division) is not int or division not in DIVISIONS:
        raise ValueError('节奏子集需要支持的整数拍格细分')

    rules = {}
    fields = {'section', 'voice', 'instrument_id', 'stride', 'phase'}
    for policy in policies:
        if not isinstance(policy, Mapping) or set(policy) != fields:
            raise ValueError('节奏子集策略必须包含范围、stride和phase')
        scope = _scope(policy['section'], policy['voice'], policy['instrument_id'])
        if scope in rules:
            raise ValueError('同一段落声部和乐器只能指定一个节奏子集')
        stride, phase = policy['stride'], policy['phase']
        if type(stride) is not int or stride <= 0 or division % stride:
            raise ValueError('节奏子集 stride 必须是拍格细分的正整数约数')
        if type(phase) is not int or not 0 <= phase < stride:
            raise ValueError('节奏子集 phase 必须是0至stride-1的整数')
        rules[scope] = stride, phase

    if not isinstance(metadata, Mapping):
        raise ValueError('节奏子集缺少候选段落信息')
    groups = defaultdict(list)
    identities = set()
    for note in candidates:
        if not isinstance(note.id, str) or not note.id or note.id in identities:
            raise ValueError('节奏子集候选标识缺失或重复')
        identities.add(note.id)
        meta = metadata.get(note.id)
        if not isinstance(meta, Mapping):
            raise ValueError('节奏子集缺少候选段落信息')
        scope = _scope(meta.get('section'), note.voice, note.instrument_id)
        if scope in rules:
            groups[scope].append(note)
    if set(rules) != set(groups):
        raise ValueError('节奏子集范围没有对应候选音符')

    tick_ms = 60000 / bpm / division
    decisions, rows = {}, []
    for scope, (stride, phase) in rules.items():
        descriptor = dict(zip(('section', 'voice', 'instrument_id'), scope))
        attacks, kept_attacks, kept = set(), set(), 0
        for note in groups[scope]:
            if type(note.start) is not int or not 0 <= note.start <= 24 * 3600 * 1000:
                raise ValueError('节奏子集候选起音必须是有效整数毫秒')
            tick = round((note.start - offset) / tick_ms)
            if abs(note.start - (offset + tick * tick_ms)) > .501:
                raise ValueError('节奏子集候选没有落在已确认拍格上')
            keep = tick % stride == phase
            decisions[note.id] = dict(keep=keep, tick=tick, scope=dict(descriptor),
                                     stride=stride, phase=phase)
            attacks.add(note.start)
            kept += keep
            if keep:
                kept_attacks.add(note.start)
        rows.append(dict(descriptor, stride=stride, phase=phase,
            candidates=len(groups[scope]), kept_candidates=kept,
            omitted_candidates=len(groups[scope])-kept, attack_groups=len(attacks),
            kept_attack_groups=len(kept_attacks),
            omitted_attack_groups=len(attacks)-len(kept_attacks)))
    kept = sum(decision['keep'] for decision in decisions.values())
    return decisions, dict(revision=1, research_only=True,
        semantics='explicit_chart_grid_subset', automatic_role_inference=False,
        clock=dict(bpm=bpm, offset=offset, division=division), clock_confirmed=True,
        candidates=len(decisions), kept_candidates=kept,
        omitted_candidates=len(decisions)-kept, policies=rows)
