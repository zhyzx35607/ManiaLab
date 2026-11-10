"""Experimental stream hints for expression, independent of instrument identity.

Group labels are hypotheses, not physical hands or new sound sources. Local
components cannot establish silence after their last member. A change requires
agreement with at least one fully observed source-wide grouping hypothesis.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

from .expression_articulation import articulation_policy


def candidate_signature(notes):
    data = [n.data() for n in sorted(notes, key=lambda n: n.id)]
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_stream_evidence(value, notes, observed_until_ms):
    if not isinstance(value, dict) or value.get('revision') != 1:
        raise ValueError('声音流证据版本无效')
    voice = value.get('voice')
    source = [n for n in notes if n.voice == voice]
    if not isinstance(voice, str) or not source or len({n.id for n in source}) != len(source):
        raise ValueError('声音流证据必须对应一个现有声部和唯一音符标识')
    if (type(observed_until_ms) is not int or observed_until_ms <= 0
            or value.get('observed_until_ms') != observed_until_ms
            or type(value.get('observed_until_ms')) is not int
            or max(n.start for n in source) >= observed_until_ms):
        raise ValueError('声音流证据与音频观测时长不一致')
    if value.get('candidates_sha256') != candidate_signature(source):
        raise ValueError('声音流证据与当前采音不一致，请重新分析')
    hypotheses = value.get('hypotheses')
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError('声音流证据缺少分组假设')
    names = set()
    ids = {n.id for n in source}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            raise ValueError('声音流分组格式无效')
        method = hypothesis.get('method')
        if not isinstance(method, str) or not method or method in names:
            raise ValueError('声音流分组方法必须具备唯一标识')
        names.add(method)
        if hypothesis.get('identity_scope') not in ('source', 'fragment'):
            raise ValueError('声音流身份范围无效')
        digest = hypothesis.get('prediction_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('声音流证据缺少预测来源')
        groups = hypothesis.get('groups')
        if not isinstance(groups, dict) or set(groups) != ids:
            raise ValueError('声音流分组必须覆盖本声部所有候选，未知身份请明确留空')
        if any(group is not None and (not isinstance(group, str) or not group or len(group) > 128)
               for group in groups.values()):
            raise ValueError('声音流分组标识无效')
    return source


def stream_articulation_policy(notes, evidence, *, observed_until_ms):
    """Return full-source decisions; apply section/manual modes afterwards.

    Unknown assignments are never combined into an invented stream. They also
    prevent a source-wide hypothesis from claiming complete identity context.
    Agreement is an expression heuristic, not independent acoustic validation.
    """
    source = validate_stream_evidence(evidence, notes, observed_until_ms)
    eligible = [n for n in source if n.strength >= .35]
    baseline = articulation_policy(eligible, observed_until_ms=observed_until_ms)
    observations = defaultdict(list)
    for hypothesis in evidence['hypotheses']:
        groups = defaultdict(list)
        complete = {}
        for note in eligible:
            key = (note.voice, note.instrument_id)
            complete[key] = complete.get(key, True) and hypothesis['groups'][note.id] is not None
            group = hypothesis['groups'][note.id]
            if group is not None:
                groups[(*key, group)].append(note)
        for key, members in groups.items():
            full = hypothesis['identity_scope'] == 'source' and complete[key[:2]]
            # No evidence beyond a fragment's last attack establishes a gap
            # in this identity. Do not use the end of the audio as proof.
            boundary = observed_until_ms if full else max(n.start for n in members)
            choices = articulation_policy(members, observed_until_ms=boundary)
            for note in members:
                choice = choices[note.id]
                observations[note.id].append(dict(method=hypothesis['method'], group=key[2],
                    identity_scope=hypothesis['identity_scope'], complete_source_context=full,
                    right_identity_context=choice['continuation'] if full or choice['continuation'] == 'next_attack' else 'unknown_identity',
                    prediction_sha256=hypothesis['prediction_sha256'], choice=choice))
    result = {}
    for note in eligible:
        original = baseline[note.id]
        rows = observations[note.id]
        complete = [r for r in rows if r['complete_source_context']]
        if len(rows) != len(evidence['hypotheses']):
            status = 'unknown_assignment'
        elif not complete:
            status = 'unknown_source_context'
        elif len({r['choice']['tap'] for r in rows}) > 1:
            status = 'grouping_disagreement'
        else:
            status = 'groupings_agree'
        chosen = complete[0]['choice'] if status == 'groupings_agree' else original
        result[note.id] = dict(chosen, stream=dict(status=status, baseline_tap=original['tap'],
            changed=chosen['tap'] != original['tap'], candidates_sha256=evidence['candidates_sha256'],
            observations=rows))
    return result
