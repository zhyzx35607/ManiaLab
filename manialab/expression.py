"""Choose source phrases before assigning lanes, preserving their expression.

This layer does not claim to transcribe undetected notes or learn from chart
geometry. Every decision refers to an existing audio candidate ID.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csc_matrix

from .arrange import _solve_binary, parameters, settings_at
from .domain import UNOBSERVED_RELEASES,note_from
from .expression_articulation import articulation_policy
from .expression_activity import section_activity
from .structure import detect_structure

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


def signature(analysis):
    payload = [analysis.get('structure'), analysis.get('candidates'), analysis.get('phrase_candidates'),
               analysis.get('duration')]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def propose_expression(analysis, settings):
    p = parameters(settings)
    structure = analysis.get('structure') or detect_structure(analysis)
    notes = analysis.get('candidates', [])
    sections = []
    voices = sorted({n.get('voice', 'mix') for n in notes})
    sources = {voice: [note_from(n) for n in notes if n.get('voice', 'mix') == voice and n.get('strength', 0) >= .35]
               for voice in voices}
    for segment in structure['segments']:
        a, b = segment['start_ms'], segment['end_ms']
        evidence = {voice: section_activity(sources[voice], a, b) for voice in voices}
        active = [voice for voice in voices if evidence[voice]['present'] and p.get(voice+'_weight', 1.) > 0]
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
            elif voice == 'drums' or p['ln_ratio'] == 0:
                mode = 'onsets'
            elif voice in ('piano', 'guitar') and p['ln_ratio'] < .75:
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


def expression_evidence(analysis, plan, settings, *, stream_evidence=None):
    """Apply an explicit source policy, without LN quotas or lane decisions."""
    validate_expression(plan, analysis)
    p = parameters(settings)
    raw = [note_from(n) for n in analysis['candidates']]
    if (any(not isinstance(n.id,str) or not n.id for n in raw)
            or len({n.id for n in raw})!=len(raw)):
        raise ValueError('采音结果的音符标识缺失或重复，请重新分析音频')
    phrases = sorted([note_from(n) for n in analysis.get('phrase_candidates', [])], key=lambda n: n.start)
    output, trace, metadata = [], {}, {}
    # Analyse the source's attack context once. Cutting chart sections through
    # a chord or run must not turn its last local object into a phrase ending.
    articulation = {}
    articulation_streams = {(n.voice, n.instrument_id) for section in plan['sections'] for n in raw
        if section['start_ms'] <= n.start < section['end_ms']
        and _expression_mode(section, n.voice, n.instrument_id) == 'articulation'}
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
        for phrase in phrases:
            anchor = by_id.get(phrase.id)
            if (anchor is None or phrase.voice != anchor.voice or phrase.instrument_id != anchor.instrument_id):
                continue
            if (_expression_mode(section, anchor.voice, anchor.instrument_id) != 'phrase' or
                    phrase.support_coverage < .95 or phrase.end > section['end_ms'] or
                    phrase.release_reason in UNOBSERVED_RELEASES or phrase.source != 'pitch' or
                    phrase.end-phrase.start < p['min_ln']):
                continue
            if (phrase.end <= anchor.end or phrase.id in covered or anchor.strength < .35):
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
            mode = _expression_mode(section, original.voice, original.instrument_id)
            base = dict(id=original.id, voice=original.voice, start=original.start, end=original.end,
                        instrument_id=original.instrument_id, section=index, mode=mode, role=policy['role'],
                        voice_mode=policy['mode'],
                        instrument_mode=_instrument_mode(section, original.voice, original.instrument_id))
            reason = None
            if mode == 'omit' or p.get(original.voice+'_weight', 1.) <= 0:
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
            hold = n.source == 'pitch' and mode != 'onsets' and n.end-n.start >= p['min_ln']
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


def resolve_phrase_decisions(trace):
    # A gesture can lose the later capacity competition. Its interior notes
    # must not claim to be represented by a gesture absent from the chart.
    for entry in trace.values():
        if entry.get('reason') == 'represented_by_phrase':
            representative = trace.get(entry['represented_by'], {})
            if representative.get('decision') != 'selected':
                entry['reason'] = 'phrase_not_selected'


def chord_layer_options(notes, settings, segment_preferences):
    """Offer fewer pitch layers without deleting a source's rhythmic attacks.

    These express one source's pitches at the same quantized attack, not claims
    that omitted pitches are false positives. Retained notes keep their ends.
    Equal source weights use the existing attack-alignment score before pitch;
    a unit-weight detector must not always lose its clearer higher chord tones.
    """
    attacks = defaultdict(list)
    for note in notes:
        attacks[note.start].append(note)
    rows = []
    depth = 1
    for time, row in sorted(attacks.items()):
        pref = settings_at(settings, time, segment_preferences)
        cap = min(pref['chord_size'], math.ceil(pref['density']))
        pitched = all(n.source == 'pitch' for n in row)
        aligned = pitched and all(n.evidence_start is not None for n in row)
        row.sort(key=lambda n: (-n.strength, -_attack_value(n, pref) if aligned else 0,
                                n.pitch, n.id))
        if pitched:
            depth = max(depth, min(len(row), cap))
        rows.append((row, cap, pitched))
    return [[n for row, cap, pitched in rows for n in
             (row[:min(layers, cap)] if pitched else row)] for layers in range(1, depth+1)]


def _attack_value(note, preferences):
    shift = abs(note.start-note.evidence_start) if note.evidence_start is not None else 0
    alignment = .4+.6*math.exp(-.5*(shift/20)**2)
    return (.25+note.strength)*preferences.get(note.voice+'_weight', 1.)*alignment


def _deduplicate_pitches(notes, trace, settings, segment_preferences):
    """Prefer better aligned evidence only for interchangeable expressions.

    Keep the former choice for competing lengths, chord ranks and gestures.
    Raw evidence endpoints may differ; retain the winning Note whole, never
    combine one detection's attack with another detection's release.
    """
    duplicates = defaultdict(list)
    for note in notes:
        duplicates[(note.start, note.pitch, note.source)].append(note)
    phrase_anchors = {row['represented_by'] for row in trace.values()
                      if row.get('reason') == 'represented_by_phrase'}
    kept = []
    for options in duplicates.values():
        first = options[0]
        winner = first
        if len(options) > 1 and not first.phrase and not first.support and first.id not in phrase_anchors:
            fields = ('end', 'strength', 'lane', 'raw', 'release_reason',
                      'phrase', 'support', 'support_coverage')
            compatible = [n for n in options if n.id not in phrase_anchors
                and all(getattr(n, key) == getattr(first, key) for key in fields)
                and all(trace[n.id].get(key) == trace[first.id].get(key)
                        for key in ('mode', 'reason', 'articulation'))]
            pref = settings_at(settings, first.start, segment_preferences)
            winner = min(compatible, key=lambda n: (-_attack_value(n, pref), n.id))
        kept.append(winner)
        for note in options:
            if note.id != winner.id:
                trace[note.id].update(decision='omitted', reason='duplicate_pitch')
    return kept


def drum_group_anchor(notes, settings, segment_preferences):
    """Offer one existing transient when a complete drum figure cannot fit.

    This is an arrangement reduction, not a claim about drum identity or
    detecting missing attacks. Other attacks remain explicitly omitted.
    """
    if len(notes) <= 1 or any(n.voice != 'drums' or n.source != 'transient' or n.hold for n in notes):
        return None
    return min(notes, key=lambda n: (-_attack_value(n, settings_at(settings, n.start, segment_preferences)),
                                    n.start, n.id))


def _select_drum_anchors(selected, anchors, metadata, settings, segment_preferences, check_times):
    """Use residual capacity only; every already selected note stays fixed."""
    if not anchors:
        return [], None
    rows, cols, values, limits = [], [], [], []
    def constraint(indices, remaining):
        if remaining < 0:
            raise ValueError('既有声部方案超过容量，不能补充鼓点')
        if indices:
            row = len(limits)
            limits.append(remaining)
            for index in indices:
                rows.append(row); cols.append(index); values.append(1)
    holds = [(n, settings_at(settings, n.start, segment_preferences)['release_gap'])
             for n in selected if n.hold]
    # Original candidate times can carry a tighter section limit even when
    # the candidate itself was omitted. Keep the original solver's checks.
    for time in sorted(set(check_times) | {n.start for n in selected+anchors}):
        pref = settings_at(settings, time, segment_preferences)
        existing = sum(n.start == time for n in selected)
        onsets = [i for i, n in enumerate(anchors) if n.start == time]
        constraint(onsets, pref['chord_size']-existing)
        occupied = existing+sum(n.start < time < n.end+gap for n, gap in holds)
        constraint(onsets, settings['keys']-occupied)
        recent = sum(time-1000 < n.start <= time for n in selected)
        constraint([i for i, n in enumerate(anchors) if time-1000 < n.start <= time],
                   math.ceil(pref['density'])-recent)
    utility = np.asarray([_attack_value(n, settings_at(settings, n.start, segment_preferences))
                          * (2 if metadata[n.id]['role'] == 'primary' else 1) for n in anchors])
    matrix = csc_matrix((values, (rows, cols)), shape=(len(limits), len(anchors)))
    solved = _solve_binary(-utility, integrality=np.ones(len(anchors)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, np.asarray(limits)),
        options={'time_limit': 5., 'mip_rel_gap': .005})
    if solved.x is None:
        raise ValueError('剩余容量的鼓点选择没有找到可行方案')
    chosen = solved.x > .5
    if np.any(matrix @ chosen.astype(float) > np.asarray(limits)+1e-6):
        raise ValueError('补充鼓点没有通过占轨和密度约束')
    return [n for n, keep in zip(anchors, chosen) if keep], dict(
        candidates=len(anchors), selected=int(chosen.sum()), status=int(solved.status),
        relative_gap=float(solved.mip_gap), objective=float(-solved.fun),
        existing_notes_fixed=True)


def select_expression(
    candidates, metadata, trace, settings, beat_ms, segment_preferences=(),
    *, coherence=0., sustain_coverage=0.,
):
    """Select rhythmic groups and chord depth; retained LNs keep their ends.

    A source group can use fewer simultaneous pitch layers while retaining its
    attack sequence. Choose at most one depth per group. If even one layer per
    attack cannot fit alone, split the group into attacks before selection.
    After selection, wholly omitted transient drum groups may retain one
    aligned attack in the remaining capacity without changing existing notes.
    Research callers can request bounded adjacent-group transition costs;
    sustain coverage is also bounded and rewards only supported long notes.
    Both research weights default to zero pending musical evaluation.
    """
    from .expression_coherence import validate_weight, group_links, add_costs, selection_report
    from . import expression_sustain
    validate_weight(coherence, beat_ms)
    expression_sustain.validate_weight(sustain_coverage, beat_ms)
    p = parameters(settings)
    by_voice = defaultdict(list)
    for n in candidates:
        meta = metadata[n.id]
        if settings_at(p, n.start, segment_preferences).get(n.voice+'_weight', 1.) <= 0:
            trace[n.id].update(decision='omitted', reason='voice_not_selected')
            continue
        if trace[n.id].get('expression_end', n.end) > trace[n.id]['start'] and not n.hold:
            trace[n.id]['reason'] = 'duration_collapsed_on_grid'
        by_voice[(meta['section'], n.voice, n.instrument_id, meta['role'])].append(n)
    units, roles, families = [], [], []
    def family(group, role):
        indices = []
        for option in chord_layer_options(group, p, segment_preferences):
            indices.append(len(units))
            units.append(option)
            roles.append(role)
        families.append((group, indices))
    for (section, voice, instrument_id, role), notes in by_voice.items():
        notes.sort(key=lambda n: (n.start, -n.strength, n.pitch))
        groups, current = [], []
        for n in _deduplicate_pitches(notes, trace, p, segment_preferences):
            # 2-beat blocks preserve short musical figures. Pauses are split
            # earlier; blocks are source-local, so quiet main notes do not
            # compete individually with loud drums on intervening rows.
            if current and (n.start-current[-1].start > beat_ms*1.25 or n.start-current[0].start >= beat_ms*2):
                groups.append(current)
                current = []
            current.append(n)
        if current:
            groups.append(current)
        for group in groups:
            # Test the lightest expression, not the number of MIDI pitches.
            # A five-pitch chord must not force an entire phrase to disappear
            # when the configured chart chord limit is only two notes.
            base = chord_layer_options(group, p, segment_preferences)[0]
            impossible = any(
                sum(t.start-1000 < n.start <= t.start for n in base) > math.ceil(pref['density']) or
                sum(n.hold and n.start <= t.start < n.end for n in base) > pref['max_held'] or
                sum(n.start == t.start for n in base) > pref['chord_size'] or
                sum(n.start == t.start for n in base) + sum(n.hold and n.start < t.start < n.end+pref['release_gap'] for n in base) > p['keys']
                for t in base for pref in (settings_at(p, t.start, segment_preferences),))
            if impossible:
                attacks = defaultdict(list)
                for n in group:
                    attacks[n.start].append(n)
                for attack in attacks.values():
                    family(attack, role)
            else:
                family(group, role)
    if not units:
        resolve_phrase_decisions(trace)
        report = dict(message='当前段落方案没有可用音符。', decisions=list(trace.values()), selected=0)
        if sustain_coverage:
            report['sustain_coverage'] = expression_sustain.coverage_report([], [], sustain_coverage)
        return [], report
    by_start, holds, total = defaultdict(Counter), [], []
    for unit, group in enumerate(units):
        for n in group:
            by_start[n.start][unit] += 1
            total.append((n.start, unit))
            if n.hold:
                holds.append((n, unit))
    rows, cols, vals, upper = [], [], [], []
    def constraint(weights, cap):
        row = len(upper); upper.append(cap)
        for col, value in weights.items():
            if value:
                rows.append(row); cols.append(col); vals.append(value)
    for _, indices in families:
        if len(indices) > 1:
            constraint({index: 1 for index in indices}, 1)
    # Sweep events in half-open intervals. Release and gap-end events happen
    # before attacks at the same millisecond, matching layout feasibility.
    events = defaultdict(list)
    for n, unit in holds:
        pref = settings_at(p, n.start, segment_preferences)
        events[n.start].append((unit, 1, 1))
        events[n.end].append((unit, -1, 0))
        events[n.end+pref['release_gap']].append((unit, 0, -1))
    live, reserved = Counter(), Counter()
    boundaries = {s[k] for s in segment_preferences for k in ('start_ms', 'end_ms')}
    times = sorted(set(events) | set(by_start) | boundaries)
    recent, pos = Counter(), 0
    total.sort()
    added = 0
    for t in times:
        # Check held capacity just after this time, including new holds.
        new_holds = Counter()
        for unit, delta, gap_delta in events[t]:
            live[unit] += delta; reserved[unit] += gap_delta
            if delta > 0:
                new_holds[unit] += delta
        pref = settings_at(p, t, segment_preferences)
        constraint(live, pref['max_held'])
        if t not in by_start:
            continue
        chord = by_start[t]
        constraint(chord, pref['chord_size'])
        occupied = reserved.copy()
        occupied.subtract(new_holds)
        occupied.update(chord)
        constraint(occupied, p['keys'])
        while added < len(total) and total[added][0] <= t:
            recent[total[added][1]] += 1; added += 1
        while pos < added and total[pos][0] <= t-1000:
            recent[total[pos][1]] -= 1; pos += 1
        constraint(recent, math.ceil(pref['density']))
    utility = [0.] * len(units)
    attack_units, attack_instruments = defaultdict(dict), defaultdict(set)
    for unit, (role, group) in enumerate(zip(roles, units)):
        attack_scores = defaultdict(list)
        for n in group:
            pref = settings_at(p, n.start, segment_preferences)
            key = (metadata[n.id]['section'], n.voice, n.start)
            value = _attack_value(n, pref)
            attack_scores[key].append(value * (2 if role == 'primary' else 1))
            attack_instruments[key].add(n.instrument_id)
        for key, values in attack_scores.items():
            attack_units[key][unit] = values
    for key, options in attack_units.items():
        if len(attack_instruments[key]) == 1:
            # Alternatives within one instrument family are mutually exclusive.
            for unit, values in options.items():
                utility[unit] += max(values)+.15*(sum(values)-max(values))
            continue
        # Distinct instruments retain separate phrases, but the source's attack
        # reward is paid once. Otherwise splitting a label makes every chord
        # pitch a full-value attack and can displace an unrelated melody.
        maxima = {unit: max(values) for unit, values in options.items()}
        for unit, values in options.items():
            utility[unit] += .15*sum(values)
        previous = 0.
        for level in sorted(set(maxima.values())):
            coverage = len(utility)
            utility.append(.85*(level-previous))
            # Each positive threshold reward requires a selected option that
            # actually contains a note of that quality. An omitted strong note
            # cannot lend its evidence score to a selected weaker note.
            constraint({coverage: 1, **{unit: -1 for unit, value in maxima.items() if value >= level}}, 0)
            previous = level
    links = group_links(families, metadata, beat_ms, coherence,
        lambda n: _attack_value(n, settings_at(p, n.start, segment_preferences)), boundaries)
    add_costs(links, families, utility, constraint)
    sustain_intervals = expression_sustain.coverage_intervals(units, metadata, trace, beat_ms,
        sustain_coverage, lambda n: _attack_value(n, settings_at(p, n.start, segment_preferences)))
    expression_sustain.add_rewards(sustain_intervals, utility, constraint)
    matrix = csc_matrix((vals, (rows, cols)), shape=(len(upper), len(utility)))
    solved = _solve_binary(-np.asarray(utility), integrality=np.ones(len(utility)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, np.asarray(upper)), options={'time_limit': 15., 'mip_rel_gap': .005})
    if solved.x is None:
        raise ValueError('段落取舍未找到可行方案，请降低主声部层数或提高密度上限')
    chosen = solved.x > .5
    if np.any(matrix @ chosen.astype(float) > np.asarray(upper)+1e-6):
        raise ValueError('段落取舍没有通过占轨约束')
    selected, anchors, anchor_groups = [], [], {}
    for family_index, (group, indices) in enumerate(families):
        active = [index for index in indices if chosen[index]]
        if len(active) > 1:
            raise ValueError('同一音组选择了多个表达层数')
        retained = units[active[0]] if active else []
        if not retained:
            anchor = drum_group_anchor(group, p, segment_preferences)
            if anchor is not None:
                anchors.append(anchor)
                anchor_groups[anchor.id] = group
        retained_ids = {n.id for n in retained}
        represented = defaultdict(list)
        for n in retained:
            represented[n.start].append(n.id)
        for n in group:
            keep = n.id in retained_ids
            trace[n.id].update(decision='selected' if keep else 'omitted', group=family_index)
            if not keep:
                trace[n.id]['reason'] = 'represented_by_chord' if represented[n.start] else 'group_capacity_or_density'
                if represented[n.start]:
                    trace[n.id]['represented_by'] = represented[n.start]
                    trace[n.id]['represented_at'] = n.start
            else:
                selected.append(n)
                trace[n.id].update(chart_start=n.start, chart_end=n.end)
    added_drums, fallback_solver = _select_drum_anchors(selected, anchors, metadata, p, segment_preferences, by_start)
    for n in added_drums:
        for member in anchor_groups[n.id]:
            trace[member.id]['group_simplification'] = 'strongest_drum_attack'
        trace[n.id].update(decision='selected', reason='drum_group_fallback', chart_start=n.start, chart_end=n.end)
    selected.extend(added_drums)
    selected.sort(key=lambda n: (n.start, n.pitch, n.id))
    resolve_phrase_decisions(trace)
    count = len(selected); ln = sum(n.hold for n in selected)
    reasons = dict(Counter(r['reason'] for r in trace.values() if r['decision'] == 'omitted'))
    report = dict(selected=count, candidates=len(trace), omitted=len(trace)-count,
        attack_selection='source_groups_then_drum_anchors',
        attack_optimal=solved.status == 0 and (fallback_solver is None or fallback_solver['status'] == 0),
        drum_fallback_solver=fallback_solver,
        selection_solver=dict(status=int(solved.status), message=str(solved.message),
            relative_gap=float(solved.mip_gap), objective=float(-solved.fun),
            objective_bound=float(-solved.mip_dual_bound)),
        expression_revision=3, actual_ln_ratio=round(ln/max(1, count), 4),
        selected_voices=dict(Counter(n.voice for n in selected)), omission_reasons=reasons,
        selected_large_attack_shifts=sum(abs(n.start-n.evidence_start)>30 for n in selected if n.evidence_start is not None),
        decisions=list(trace.values()),
        message=f'按段落声部策略保留 {count} 个物件、{ln} 个 LN；不以全曲比例截断音尾。可查看各音的保留和省略原因。')
    if coherence:
        report['group_coherence'] = selection_report(links, families, chosen, coherence)
    if sustain_coverage:
        report['sustain_coverage'] = expression_sustain.coverage_report(sustain_intervals, chosen, sustain_coverage)
    return selected, report
