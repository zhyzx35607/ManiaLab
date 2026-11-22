"""Choose source phrases before assigning lanes, preserving their expression.

This layer does not claim to transcribe undetected notes or learn from chart
geometry. Every decision refers to an existing audio candidate ID.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math

from .arrange import parameters, settings_at
from .solver import solve_binary as _solve_binary
from .domain import UNOBSERVED_RELEASES,note_from
from .expression_articulation import articulation_policy
from .expression_activity import section_activity
from .expression_selection import (
    _attack_value as _selection_attack_value,
    _deduplicate_pitches as _selection_deduplicate_pitches,
    _group_fits as _selection_group_fits,
    _select_drum_anchors as _selection_drum_anchors,
    chord_layer_options as _selection_chord_layer_options,
    drum_group_anchor as _selection_drum_group_anchor,
    resolve_phrase_decisions as _selection_resolve_phrase_decisions,
    select_expression as _select_expression,
)
from .structure import detect_structure
from .voices import reconcile_phrase_sources, select_phrase_subset

MODES = ('sustain', 'articulation', 'onsets', 'phrase', 'omit')


def _section_instruments(notes, begin, end):
    instruments = defaultdict(set)
    for note in notes:
        identity = note.get('instrument_id', '')
        if begin <= note['start'] < end and isinstance(identity, str) and identity:
            instruments[note.get('voice', 'mix')].add(identity)
    return instruments


def _instrument_mode(section, voice, instrument_id):
    return section.get('instrument_modes', {}).get(voice, {}).get(instrument_id, 'inherit')


def _expression_mode(section, voice, instrument_id):
    mode = _instrument_mode(section, voice, instrument_id)
    return section['voices'][voice]['mode'] if mode == 'inherit' else mode


def _phrase_eligible(phrase, anchor, section, min_length):
    return (
        anchor is not None
        and phrase.voice == anchor.voice
        and phrase.instrument_id == anchor.instrument_id
        and phrase.support_coverage >= .95
        and phrase.end <= section['end_ms']
        and phrase.release_reason not in UNOBSERVED_RELEASES
        and phrase.source == 'pitch'
        and phrase.end - phrase.start >= min_length
        and phrase.end > anchor.end
        and anchor.strength >= .35
    )


def signature(analysis):
    payload = [analysis.get('structure'), analysis.get('candidates'), analysis.get('phrase_candidates'),
               analysis.get('duration')]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def propose_expression(analysis, settings, segment_preferences=(), *,
                      candidate_evidence=None, phrase_evidence=None):
    p = parameters(settings)
    structure = analysis.get('structure') or detect_structure(analysis)
    notes = analysis.get('candidates', []) if candidate_evidence is None else candidate_evidence
    identities = [note.get('id') for note in notes]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError('候选音符标识缺失或无效')
    if len(set(identities)) != len(identities):
        raise ValueError('候选音符标识重复')
    anchors = {n.get('id'): note_from(n) for n in notes}
    phrase_rows = analysis.get('phrase_candidates', []) if phrase_evidence is None else phrase_evidence
    phrases = sorted([note_from(n) for n in phrase_rows], key=lambda n: n.start)
    sections = []
    voices = sorted({n.get('voice', 'mix') for n in notes})
    sources = {voice: [note_from(n) for n in notes if n.get('voice', 'mix') == voice and n.get('strength', 0) >= .35]
               for voice in voices}
    for segment in structure['segments']:
        a, b = segment['start_ms'], segment['end_ms']
        local_settings = settings_at(p, a, segment_preferences)
        evidence = {voice: section_activity(sources[voice], a, b) for voice in voices}
        active = [voice for voice in voices if evidence[voice]['present']
                  and local_settings.get(voice+'_weight', 1.) > 0]
        local = {identity: anchor for identity, anchor in anchors.items() if a <= anchor.start < b}
        phrase_voices = {phrase.voice for phrase in phrases
            if (_phrase_eligible(phrase, local.get(phrase.id), segment,
                                 settings_at(p, phrase.start, segment_preferences)['min_ln'])
                and settings_at(p, phrase.start, segment_preferences)['phrase_ln'] > 0)}
        melodic = [voice for voice in active if voice != 'drums']
        # Coverage, attack counts and model classes do not identify a lead.
        # Only suggest a priority when no other melodic source is present.
        # In an ensemble the user's explicit choice can still grant priority.
        main = melodic[0] if len(melodic) == 1 else None
        status = 'single_melodic_source' if main else 'multiple_melodic_sources' if melodic else 'no_melodic_source'
        policies = {}
        for voice in voices:
            if voice not in active:
                mode = 'omit'
            elif voice == 'drums' or local_settings['ln_ratio'] == 0:
                mode = 'onsets'
            elif voice == 'vocals' and local_settings['phrase_ln'] > 0 and voice in phrase_voices:
                mode = 'phrase'
            elif voice in ('piano', 'guitar') and local_settings['ln_ratio'] < .75:
                mode = 'articulation'
            else:
                mode = 'sustain'
            policies[voice] = dict(mode=mode, role='primary' if voice == main else 'support',
                activity=evidence[voice])
        instruments = _section_instruments(notes, a, b)
        sections.append(dict(start_ms=a, end_ms=b, label=segment.get('label', ''),
            similarity_group=segment.get('similarity_group'), main_voice=main,
            priority_evidence=dict(status=status, eligible_melodic_sources=melodic,
                                   salience_estimated=False),
            voices=policies,
            instrument_modes={voice: {identity: 'inherit' for identity in sorted(identities)}
                              for voice, identities in sorted(instruments.items())},
            reason='多声部同时存在时先不指定主声部，可试听后手动突出；持续时间不等于旋律重要性。各声部的长短表达独立选择，跨段延续沿用起音所在段的方案。'))
    return dict(revision=3, source='automatic', evidence_signature=signature(analysis),
        sections=sections, notice='各声部独立选择长短表达，不强凑全曲 LN 比例。倾向为零时只表示起音；低于 0.75 时钢琴和吉他优先连续起音，达到 0.75 时保留完整持续音。短音、未决尾和占轨限制仍会影响实际结果。')


def validate_expression(value, analysis):
    if not isinstance(value, dict) or value.get('evidence_signature') != signature(analysis):
        raise ValueError('采音或段落已经改变，请重新生成声部方案')
    sections = value.get('sections')
    expected = (analysis.get('structure') or detect_structure(analysis))['segments']
    voices = {n.get('voice', 'mix') for n in analysis.get('candidates', [])}
    if not isinstance(sections, list) or len(sections) != len(expected):
        raise ValueError('声部方案必须覆盖全部音乐段落')
    for row, segment in zip(sections, expected):
        if not isinstance(row, dict) or any(row.get(k) != segment[k] for k in ('start_ms', 'end_ms')):
            raise ValueError('声部方案边界与歌曲分段不一致')
        policies = row.get('voices')
        if not isinstance(policies, dict) or set(policies) != voices:
            raise ValueError('声部方案必须包含当前分析的全部声部')
        for policy in policies.values():
            if not isinstance(policy, dict) or policy.get('mode') not in MODES or policy.get('role') not in ('primary', 'support'):
                raise ValueError('声部表达只能选择持续音、连续起音优先、起音、连贯乐句或不采')
        phrase_ln = row.get('phrase_ln')
        if phrase_ln is not None and (
                isinstance(phrase_ln, bool) or not isinstance(phrase_ln, (int, float))
                or not math.isfinite(phrase_ln) or not 0 <= phrase_ln <= 1):
            raise ValueError('本段人声乐句比例必须在0到1之间')
        overrides = row.get('instrument_modes', {})
        if not isinstance(overrides, dict) or not set(overrides) <= voices:
            raise ValueError('乐器表达必须按当前分析的声部分组')
        instruments = _section_instruments(analysis.get('candidates', []), segment['start_ms'], segment['end_ms'])
        for voice, modes in overrides.items():
            if not isinstance(modes, dict):
                raise ValueError('乐器表达必须是乐器标识与表达方式的对应表')
            for identity, mode in modes.items():
                if not isinstance(identity, str) or not identity or identity not in instruments.get(voice, set()):
                    raise ValueError('乐器表达包含本段不存在的乐器标识')
                if mode not in ('inherit', *MODES):
                    raise ValueError('乐器表达只能选择继承、持续音、连续起音优先、起音、连贯乐句或不采')
        active_primary = [v for v, policy in policies.items() if policy['role'] == 'primary'
            and (policy['mode'] != 'omit' or any(mode not in ('inherit', 'omit')
                 for mode in overrides.get(v, {}).values()))]
        if len(active_primary) > 1:
            raise ValueError('每个段落最多指定一个主声部')
    return value


def expression_evidence(analysis, plan, settings, *, stream_evidence=None, segment_preferences=(),
                        candidate_evidence=None, phrase_evidence=None, onsets_only=False):
    """Apply an explicit source policy, without LN quotas or lane decisions."""
    validate_expression(plan, analysis)
    p = parameters(settings)
    candidate_rows = analysis['candidates'] if candidate_evidence is None else candidate_evidence
    raw = [note_from(n) for n in candidate_rows]
    if (any(not isinstance(n.id,str) or not n.id for n in raw)
            or len({n.id for n in raw})!=len(raw)):
        raise ValueError('采音结果的音符标识缺失或重复，请重新分析音频')
    phrase_rows = analysis.get('phrase_candidates', []) if phrase_evidence is None else phrase_evidence
    phrases = sorted([note_from(n) for n in phrase_rows], key=lambda n: n.start)
    phrases = reconcile_phrase_sources(phrases, raw)

    def effective_mode(section, voice, instrument_id):
        mode = _expression_mode(section, voice, instrument_id)
        return 'onsets' if onsets_only and mode != 'omit' else mode

    output, trace, metadata = [], {}, {}
    # Analyse the source's attack context once. Cutting chart sections through
    # a chord or run must not turn its last local object into a phrase ending.
    articulation = {}
    articulation_streams = {(n.voice, n.instrument_id) for section in plan['sections'] for n in raw
        if section['start_ms'] <= n.start < section['end_ms']
        and effective_mode(section, n.voice, n.instrument_id) == 'articulation'}
    duration = analysis.get('duration')
    if articulation_streams and (isinstance(duration,bool) or not isinstance(duration,(int,float))
            or not math.isfinite(duration) or duration<=0):
        raise ValueError('连续起音表达需要有效的音频时长，请重新分析音频')
    if articulation_streams:
        articulation = articulation_policy(
            [n for n in raw if (n.voice, n.instrument_id) in articulation_streams and n.strength>=.35],
            observed_until_ms=round(duration*1000))
    if stream_evidence is not None:
        from .expression_streams import stream_articulation_policy
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('声音流表达需要有效的音频观测时长')
        # Research callers may supply fixed grouping evidence. The ordinary
        # application has no automatic model/grouping switch at this stage.
        stream_choices = stream_articulation_policy(raw, stream_evidence, observed_until_ms=round(duration*1000))
        articulation.update(stream_choices)
    for index, section in enumerate(plan['sections']):
        local = [n for n in raw if section['start_ms'] <= n.start < section['end_ms']]
        by_id = {n.id: n for n in local}
        replacements, covered = {}, {}
        phrase_groups = defaultdict(list)
        for phrase in phrases:
            anchor = by_id.get(phrase.id)
            phrase_settings = settings_at(p, phrase.start, segment_preferences)
            if (not _phrase_eligible(phrase, anchor, section, phrase_settings['min_ln']) or
                    effective_mode(section, anchor.voice, anchor.instrument_id) != 'phrase'):
                continue
            amount = section.get('phrase_ln', phrase_settings['phrase_ln'])
            phrase_groups[(anchor.voice, anchor.instrument_id, amount)].append(phrase)
        for (_, _, amount), candidates in phrase_groups.items():
            for phrase in select_phrase_subset(candidates, amount):
                anchor = by_id[phrase.id]
                if phrase.id in covered:
                    continue
                replacements[phrase.id] = phrase
            # Only strictly interior supported attacks are represented by the
            # gesture. Never swallow a new attack at its end or another voice.
                for identity in phrase.support:
                    n = by_id.get(identity)
                    if (n and n.voice == phrase.voice and n.instrument_id == phrase.instrument_id
                            and phrase.start < n.start < phrase.end
                            and n.end <= phrase.end):
                        covered[identity] = phrase.id
        for original in local:
            policy = section['voices'][original.voice]
            mode = effective_mode(section, original.voice, original.instrument_id)
            base = dict(id=original.id, voice=original.voice, start=original.start, end=original.end,
                        instrument_id=original.instrument_id, section=index, mode=mode, role=policy['role'],
                        voice_mode=policy['mode'],
                        instrument_mode=_instrument_mode(section, original.voice, original.instrument_id))
            local_settings = settings_at(p, original.start, segment_preferences)
            reason = None
            if mode == 'omit' or local_settings.get(original.voice+'_weight', 1.) <= 0:
                reason = 'voice_not_selected'
            elif original.strength < .35:
                reason = 'weak_detection'
            elif original.id in covered:
                reason = 'represented_by_phrase'
                base['represented_by'] = covered[original.id]
            if reason:
                trace[original.id] = dict(base, decision='omitted', reason=reason)
                continue
            n = replacements.get(original.id, original)
            hold = n.source == 'pitch' and mode != 'onsets' and n.end-n.start >= local_settings['min_ln']
            reason = 'supported_sustain' if hold else ('section_onsets' if mode == 'onsets' else 'short_duration')
            gesture=articulation.get(original.id) if mode=='articulation' else None
            if gesture is not None:
                base['articulation']=gesture
                if gesture['tap']:
                    hold=False
                    reason='articulated_attack_run'
            if hold and n.release_reason in UNOBSERVED_RELEASES:
                hold=False
                reason='unobserved_release'
            n = replace(n, end=n.end if hold else n.start, phrase=n.phrase and hold)
            output.append(n)
            metadata[n.id] = dict(section=index, role=policy['role'], voice=n.voice, instrument_id=n.instrument_id)
            trace[n.id] = dict(base, decision='candidate', reason=reason, expression_end=n.end)
    return output, metadata, trace


resolve_phrase_decisions = _selection_resolve_phrase_decisions
chord_layer_options = _selection_chord_layer_options
_attack_value = _selection_attack_value
_deduplicate_pitches = _selection_deduplicate_pitches
drum_group_anchor = _selection_drum_group_anchor
_group_fits = _selection_group_fits


def _select_drum_anchors(selected, anchors, metadata, settings, segment_preferences,
                         check_times, row_preferences=None):
    return _selection_drum_anchors(
        selected, anchors, metadata, settings, segment_preferences,
        check_times, row_preferences, _solve_binary,
    )


def select_expression(
    candidates, metadata, trace, settings, beat_ms, segment_preferences=(),
    *, coherence=0., sustain_coverage=0., rhythm_policy=None,
):
    return _select_expression(
        candidates, metadata, trace, settings, beat_ms, segment_preferences,
        coherence=coherence,
        sustain_coverage=sustain_coverage,
        rhythm_policy=rhythm_policy,
        _solver=_solve_binary,
    )
