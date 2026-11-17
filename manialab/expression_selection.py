"""Select source expression candidates under chart capacity constraints."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csc_matrix

from .arrange import parameters, settings_at
from .preferences import row_settings
from .solver import solve_binary


def resolve_phrase_decisions(trace):
    # A gesture can lose the later capacity competition. Its interior notes
    # must not claim to be represented by a gesture absent from the chart.
    for entry in trace.values():
        if entry.get('reason') == 'represented_by_phrase':
            representative = trace.get(entry['represented_by'], {})
            if (representative.get('decision') != 'selected' or
                    representative.get('chart_end', 0) <= representative.get('chart_start', 0)):
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
        pref = row_settings(settings, row, segment_preferences)
        cap = min(pref['chord_size'], math.ceil(pref['density']))
        pitched = all(n.source == 'pitch' for n in row)
        aligned = pitched and all(n.evidence_start is not None for n in row)
        row.sort(key=lambda n: (-n.strength, -_attack_value(n, settings_at(settings, n, segment_preferences)) if aligned else 0,
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
            pref = settings_at(settings, first, segment_preferences)
            winner = min(compatible, key=lambda n: (-_attack_value(n, settings_at(settings, n, segment_preferences)), n.id))
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
    return min(notes, key=lambda n: (-_attack_value(n, settings_at(settings, n, segment_preferences)),
                                    n.start, n.id))


def _group_fits(notes, settings, segments, times, attack_times, row_preferences):
    """Check a lightest group against the same checkpoints as the final solve.

    Other offered voices can impose tighter row limits even if later omitted.
    A boundary without an attack changes held capacity, not rolling density.
    """
    starts = sorted(n.start for n in notes)
    holds = [(n, settings_at(settings, n, segments)['release_gap']) for n in notes if n.hold]
    stop = max(starts[-1]+1000, max((n.end+gap for n, gap in holds), default=0))
    for t in times[bisect_left(times, starts[0]):bisect_left(times, stop)]:
        chart_pref = settings_at(settings, t, segments)
        pref = row_preferences.get(t, chart_pref)
        if sum(n.start <= t < n.end for n, _ in holds) > min(pref['max_held'], chart_pref['max_held']):
            return False
        if t not in attack_times:
            continue
        chord = sum(n.start == t for n in notes)
        if (sum(t-1000 < n.start <= t for n in notes) > math.ceil(pref['density'])
                or chord > pref['chord_size']
                or chord+sum(n.start < t < n.end+gap for n, gap in holds) > settings['keys']):
            return False
    return True


def _select_drum_anchors(selected, anchors, metadata, settings, segment_preferences,
                         check_times, row_preferences, solver):
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

    holds = [(n, settings_at(settings, n, segment_preferences)['release_gap'])
             for n in selected if n.hold]
    # Original candidate times can carry a tighter section limit even when
    # the candidate itself was omitted. Keep the original solver's checks.
    for time in sorted(set(check_times) | {n.start for n in selected+anchors}):
        pref = (row_preferences or {}).get(time) or settings_at(settings, time, segment_preferences)
        existing = sum(n.start == time for n in selected)
        onsets = [i for i, n in enumerate(anchors) if n.start == time]
        constraint(onsets, pref['chord_size']-existing)
        occupied = existing+sum(n.start < time < n.end+gap for n, gap in holds)
        constraint(onsets, settings['keys']-occupied)
        recent = sum(time-1000 < n.start <= time for n in selected)
        constraint([i for i, n in enumerate(anchors) if time-1000 < n.start <= time],
                   math.ceil(pref['density'])-recent)
    utility = np.asarray([_attack_value(n, settings_at(settings, n, segment_preferences))
                          * (2 if metadata[n.id]['role'] == 'primary' else 1) for n in anchors])
    matrix = csc_matrix((values, (rows, cols)), shape=(len(limits), len(anchors)))
    solved = solver(-utility, integrality=np.ones(len(anchors)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, np.asarray(limits)),
        options={'time_limit': 5., 'mip_rel_gap': 0.})
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
    *, coherence=0., sustain_coverage=0., rhythm_policy=None, _solver=None,
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
    A research rhythm policy can explicitly omit whole attack rows on a
    confirmed chart grid. It keeps the original grouping boundaries and never
    changes retained note durations; no automatic accompaniment is inferred.
    """
    from .expression_coherence import validate_weight, group_links, add_costs, selection_report
    from . import expression_sustain
    validate_weight(coherence, beat_ms)
    expression_sustain.validate_weight(sustain_coverage, beat_ms)
    from .expression_rhythm import plan_rhythm
    if rhythm_policy is not None and not isinstance(rhythm_policy, dict):
        raise ValueError('节奏策略必须包含显式拍格与段内规则')
    rhythm, rhythm_report = plan_rhythm(candidates, metadata, **(rhythm_policy or {}))
    if rhythm_report is not None:
        if coherence:
            raise ValueError('节奏简化尚不能与相邻组连贯惩罚组合')
        if not math.isclose(beat_ms, 60000 / rhythm_policy['clock']['bpm'], rel_tol=1e-10):
            raise ValueError('节奏策略拍格与表达分组拍长不一致')
        for identity, decision in rhythm.items():
            trace[identity]['rhythm_policy'] = decision
    solver = solve_binary if _solver is None else _solver
    p = parameters(settings)
    by_voice = defaultdict(list)
    for n in candidates:
        meta = metadata[n.id]
        if settings_at(p, n, segment_preferences).get(n.voice+'_weight', 1.) <= 0:
            trace[n.id].update(decision='omitted', reason='voice_not_selected')
            continue
        if trace[n.id].get('expression_end', n.end) > trace[n.id]['start'] and not n.hold:
            trace[n.id]['reason'] = 'duration_collapsed_on_grid'
        by_voice[(meta['section'], n.voice, n.instrument_id, meta['role'])].append(n)
    units, roles, families = [], [], []

    def family(group, role, options=None):
        indices = []
        for option in options if options is not None else chord_layer_options(group, p, segment_preferences):
            indices.append(len(units))
            units.append(option)
            roles.append(role)
        families.append((group, indices))

    source_group_index, rhythm_check_times = 0, set()
    source_groups, row_notes = [], defaultdict(dict)
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
            # Form groups from the complete source first: a mask must not
            # bridge an original boundary or reset phase at a new group.
            if rhythm_report is not None:
                rhythm_check_times.update(n.start for n in group)
                retained = []
                for n in group:
                    trace[n.id]['rhythm_source_group'] = source_group_index
                    if rhythm.get(n.id, {}).get('keep', True):
                        retained.append(n)
                    else:
                        trace[n.id].update(decision='omitted', reason='rhythm_policy_omitted')
                group = retained
                source_group_index += 1
                if not group:
                    continue
            options = chord_layer_options(group, p, segment_preferences)
            source_groups.append((group, role, options))
            # Only offered notes own row preferences. Muted, duplicate,
            # rhythm-masked and permanently trimmed pitches cannot tighten them.
            for option in options:
                for n in option:
                    row_notes[n.start][n.id] = n
    boundaries = {s[k] for s in segment_preferences for k in ('start_ms', 'end_ms')}
    attack_times = set(row_notes) | rhythm_check_times
    check_times = sorted(attack_times | boundaries)
    row_preferences = {t: row_settings(p, notes.values(), segment_preferences) for t, notes in row_notes.items()}
    for group, role, options in source_groups:
        # Test the lightest expression across all offered rows and future
        # capacity boundaries before deciding whether to split its attacks.
        if _group_fits(options[0], p, segment_preferences, check_times, attack_times, row_preferences):
            family(group, role, options)
        else:
            attacks = defaultdict(list)
            for n in group:
                attacks[n.start].append(n)
            for attack in attacks.values():
                family(attack, role)
    if not units:
        resolve_phrase_decisions(trace)
        report = dict(message='当前段落方案没有可用音符。', decisions=list(trace.values()), selected=0)
        if sustain_coverage:
            report['sustain_coverage'] = expression_sustain.coverage_report([], [], sustain_coverage)
        if rhythm_report is not None:
            report['rhythm_policy'] = rhythm_report
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
        pref = settings_at(p, n, segment_preferences)
        events[n.start].append((unit, 1, 1))
        events[n.end].append((unit, -1, 0))
        events[n.end+pref['release_gap']].append((unit, 0, -1))
    live, reserved = Counter(), Counter()
    times = sorted(set(events) | set(by_start) | boundaries | rhythm_check_times)
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
        pref = row_preferences.get(t) or settings_at(p, t, segment_preferences)
        # Occupancy belongs to chart time even if quantization moved a note
        # from a less constrained audio section onto this exact boundary.
        constraint(live, min(pref['max_held'], settings_at(p, t, segment_preferences)['max_held']))
        if t not in by_start and t not in rhythm_check_times:
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
            pref = settings_at(p, n, segment_preferences)
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
        lambda n: _attack_value(n, settings_at(p, n, segment_preferences)), boundaries)
    add_costs(links, families, utility, constraint)
    sustain_intervals = expression_sustain.coverage_intervals(units, metadata, trace, beat_ms,
        sustain_coverage, lambda n: _attack_value(n, settings_at(p, n, segment_preferences)))
    expression_sustain.add_rewards(sustain_intervals, utility, constraint)
    matrix = csc_matrix((vals, (rows, cols)), shape=(len(upper), len(utility)))
    # A song-wide relative gap can hide the value of several short figures.
    # Keep the time budget, but do not stop early on that relative allowance.
    solved = solver(-np.asarray(utility), integrality=np.ones(len(utility)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, np.asarray(upper)), options={'time_limit': 15., 'mip_rel_gap': 0.})
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
    added_drums, fallback_solver = _select_drum_anchors(
        selected, anchors, metadata, p, segment_preferences, set(by_start) | rhythm_check_times,
        row_preferences, solver)
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
        # Numerical optimality of both stages, not musical or joint optimality.
        attack_optimal=solved.status == 0 and solved.mip_gap == 0.
            and (fallback_solver is None or
                 (fallback_solver['status'] == 0 and fallback_solver['relative_gap'] == 0.)),
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
    if rhythm_report is not None:
        report['rhythm_policy'] = rhythm_report
    return selected, report
