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
from .expression_articulation import articulation_policy,source_activity
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
    sections, previous = [], None
    voices = sorted({n.get('voice', 'mix') for n in notes})
    for segment in structure['segments']:
        a, b = segment['start_ms'], segment['end_ms']
        local = [n for n in notes if a <= n['start'] < b and n.get('strength', 0) >= .35]
        scores,evidence = {},{}
        for voice in voices:
            if p.get(voice+'_weight', 1.) <= 0:
                continue
            source = [n for n in local if n.get('voice', 'mix') == voice]
            evidence[voice]=source_activity([note_from(n) for n in source],a,b)
            # Keep the established recommendation until the separate activity
            # and articulation proposals have musical validation. The exact
            # observed/uncertain diagnostic is not a salience probability.
            occupied=set()
            for n in source:
                occupied.update(range(n['start']//250,(min(b,max(n['end'],n['start']+250))+249)//250))
            scores[voice]=len(occupied)*250/max(1,b-a)
        melodic = {v:s for v,s in scores.items() if v != 'drums' and s > .02}
        if melodic.get('vocals', 0) >= .18:
            main = 'vocals'
        elif previous in melodic and melodic[previous] >= max(melodic.values(), default=0)*.85:
            main = previous
        else:
            main = max(melodic, key=melodic.get) if melodic else (max(scores, key=scores.get) if scores else None)
        previous = main
        policies = {}
        for voice in voices:
            present = scores.get(voice, 0) > .02
            if not present:
                mode = 'omit'
            elif voice == 'drums' or p['ln_ratio'] == 0:
                mode = 'onsets'
            elif voice == main or p['ln_ratio'] >= .75:
                mode = 'sustain'
            else:
                mode = 'onsets'
            policies[voice] = dict(mode=mode, role='primary' if voice == main else 'support',
                activity=evidence.get(voice,{}))
        instruments = _section_instruments(notes, a, b)
        sections.append(dict(start_ms=a, end_ms=b, label=segment.get('label', ''),
            similarity_group=segment.get('similarity_group'), main_voice=main,
            voices=policies,
            instrument_modes={voice: {identity: 'inherit' for identity in sorted(identities)}
                              for voice, identities in sorted(instruments.items())},
            reason='主声部按检测活动建议，仍需试听确认；长短表达可单独修改。连续起音表达为可选编排方式，不代表手指松键检测。'))
    return dict(revision=2, source='automatic', evidence_signature=signature(analysis),
        sections=sections, notice='LN 参数用于选择声部表达方向，不强凑全曲比例。段内策略保持一致；短音、停顿和占轨限制仍会影响实际结果。')


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


def expression_evidence(analysis, plan, settings):
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
    for index, section in enumerate(plan['sections']):
        local = [n for n in raw if section['start_ms'] <= n.start < section['end_ms']]
        by_id = {n.id: n for n in local}
        replacements, covered = {}, {}
        for phrase in phrases:
            anchor = by_id.get(phrase.id)
            if (anchor is None or phrase.voice != anchor.voice or phrase.instrument_id != anchor.instrument_id):
                continue
            if (_expression_mode(section, anchor.voice, anchor.instrument_id) != 'phrase' or
                    phrase.support_coverage < .95 or phrase.end > section['end_ms']):
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
    """
    attacks = defaultdict(list)
    for note in notes:
        attacks[note.start].append(note)
    rows = []
    depth = 1
    for time, row in sorted(attacks.items()):
        row.sort(key=lambda n: (-n.strength, n.pitch, n.id))
        pref = settings_at(settings, time, segment_preferences)
        cap = min(pref['chord_size'], math.ceil(pref['density']))
        pitched = all(n.source == 'pitch' for n in row)
        if pitched:
            depth = max(depth, min(len(row), cap))
        rows.append((row, cap, pitched))
    return [[n for row, cap, pitched in rows for n in
             (row[:min(layers, cap)] if pitched else row)] for layers in range(1, depth+1)]


def select_expression(candidates, metadata, trace, settings, beat_ms, segment_preferences=()):
    """Select rhythmic groups and chord depth; retained LNs keep their ends.

    A source group can use fewer simultaneous pitch layers while retaining its
    attack sequence. Choose at most one depth per group. If even one layer per
    attack cannot fit alone, split the group into attacks before selection.
    """
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
        seen, groups, current = set(), [], []
        for n in notes:
            key = (n.start, n.pitch, n.source)
            if key in seen:
                trace[n.id].update(decision='omitted', reason='duplicate_pitch')
                continue
            seen.add(key)
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
        return [], dict(message='当前段落方案没有可用音符。', decisions=list(trace.values()), selected=0)
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
            shift = abs(n.start-n.evidence_start) if n.evidence_start is not None else 0
            alignment = .4+.6*math.exp(-.5*(shift/20)**2)
            key = (metadata[n.id]['section'], n.voice, n.start)
            value = (.25+n.strength)*pref.get(n.voice+'_weight', 1.)*alignment
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
    matrix = csc_matrix((vals, (rows, cols)), shape=(len(upper), len(utility)))
    solved = _solve_binary(-np.asarray(utility), integrality=np.ones(len(utility)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, np.asarray(upper)), options={'time_limit': 15., 'mip_rel_gap': .005})
    if solved.x is None:
        raise ValueError('段落取舍未找到可行方案，请降低主声部层数或提高密度上限')
    chosen = solved.x > .5
    if np.any(matrix @ chosen.astype(float) > np.asarray(upper)+1e-6):
        raise ValueError('段落取舍没有通过占轨约束')
    selected = []
    for family_index, (group, indices) in enumerate(families):
        active = [index for index in indices if chosen[index]]
        if len(active) > 1:
            raise ValueError('同一音组选择了多个表达层数')
        retained = units[active[0]] if active else []
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
    selected.sort(key=lambda n: (n.start, n.pitch, n.id))
    resolve_phrase_decisions(trace)
    count = len(selected); ln = sum(n.hold for n in selected)
    reasons = dict(Counter(r['reason'] for r in trace.values() if r['decision'] == 'omitted'))
    return selected, dict(selected=count, candidates=len(trace), omitted=len(trace)-count,
        attack_selection='source_groups', attack_optimal=solved.status == 0,
        expression_revision=2, actual_ln_ratio=round(ln/max(1, count), 4),
        selected_voices=dict(Counter(n.voice for n in selected)), omission_reasons=reasons,
        selected_large_attack_shifts=sum(abs(n.start-n.evidence_start)>30 for n in selected if n.evidence_start is not None),
        decisions=list(trace.values()),
        message=f'按段落声部策略保留 {count} 个物件、{ln} 个 LN；不以全曲比例截断音尾。可查看各音的保留和省略原因。')
