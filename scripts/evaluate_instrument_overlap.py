"""Evaluate fixed half-window tail refinement without fitting on the results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_instrument_benchmark import match
from scripts.evaluate_synthetic_mix import pitched_match
from scripts.probe_mapper_adapter import sha
from manialab.domain import Note, note_from


def metrics(reference, predicted, end):
    complete_ref = [n for n in reference if n['offset'] < end]
    complete_pred = [n for n in predicted if n['offset'] < end]
    return {('strict' if strict else 'standard'): dict(
        pitched_onsets=match(reference, predicted, strict=strict),
        complete_intervals=match(complete_ref, complete_pred, offset=True, strict=strict),
        complete_instrument_intervals=match(complete_ref, complete_pred, offset=True,
            strict=strict, instrument=True)) for strict in (False, True)}


def inspect_edits(reference, base, decisions, end):
    """Diagnostics only; no predictions are chosen from annotation errors."""
    result = []
    for edit in decisions:
        if edit['status'] != 'replaced_from_explicit_prediction':
            continue
        n = base[edit['note_index']]
        possible = [r for r in reference if r['pitch'] == n['pitch']
            and abs(r['onset']-n['onset']) <= .05 and not n['is_drum']]
        row = dict(edit, reference_onset_candidates=len(possible))
        if n['onset'] >= end:
            row['comparison'] = 'context_only'
        elif len(possible) != 1:
            row['comparison'] = 'no_unique_reference_attack'
        elif possible[0]['offset'] >= end:
            row['comparison'] = 'reference_crosses_end'
        else:
            ref = possible[0]
            before, after = [abs(edit[k]-ref['offset'])*1000
                for k in ('original_offset', 'predicted_offset')]
            row.update(comparison='improved' if after < before else 'worsened' if after > before else 'equal',
                before_error_ms=before, after_error_ms=after)
        result.append(row)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/yourmt3/overlap-v1'))
    args = cli.parse_args()
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    plan = read(args.root/'plan.json')
    manifest = read('test-data/urmp/benchmark/manifest.json')
    cases = {c['name']: c for c in manifest['cases']}
    rows, private = [], {}
    from collections import Counter
    for item in plan['cases']:
        name = item['name']; original = Path(item['base']); folder = args.root/name
        base_report = read(original/'report.json'); refined_report = read(folder/'refined/report.json')
        base = read(original/'notes.json'); refined = read(folder/'refined/notes.json')
        if (refined_report['baseline_notes_sha256'] != sha(original/'notes.json') or
            refined_report['alternate_notes_sha256'] != sha(folder/'shifted/notes.json')):
            raise ValueError('Frozen model evidence changed')
        if len(base) != len(refined) or any({k:v for k,v in a.items() if k != 'offset'} !=
            {k:v for k,v in b.items() if k != 'offset'} for a,b in zip(base, refined)):
            raise ValueError('Refinement changed an attack or its instrument')
        end = base_report['identity']['start_ms']/1000 + base_report['identity']['seconds']
        row = dict(name=name, refinement=refined_report, metrics={})
        decisions = read(folder/'refined/decisions.json')
        if name.startswith('urmp-'):
            case = cases[name[5:]]
            if sha(case['labels_path']) != case['labels_sha256']:
                raise ValueError('Reference annotations changed')
            if base_report['identity']['sources'] != {case['audio']:case['audio_sha256']}:
                raise ValueError('Reference and prediction use different audio')
            reference = [n for n in read(case['labels_path']) if 0 <= n['onset'] < end]
            for kind, prediction in (('baseline', base), ('refined', refined)):
                local = [n for n in prediction if 0 <= n['onset'] < end]
                row['metrics'][kind] = metrics(reference, local, end)
                row.setdefault('interval_scope', {})[kind] = dict(
                    reference_crosses_end=sum(n['offset'] >= end for n in reference),
                    predicted_crosses_end=sum(n['offset'] >= end for n in local))
            private[name] = inspect_edits(reference, base, decisions, end)
            row['edited_reference_diagnostics'] = dict(Counter(r['comparison'] for r in private[name]))
        else:
            if name == 'control-controlled-mix':
                reference = [note_from(n) for n in read('test-data/synthetic-mix/ground-truth.json')]
            elif name == 'control-repeated-chords':
                reference = [Note(round((.115+group+i*.5)*1000), round((.585+group+i*.5)*1000), pitch=pitch)
                    for group in (0, 5) for i in range(8) for pitch in (60, 64, 67)]
            else:
                reference = None
                row['notice'] = 'Natural piano has tap-only authored labels. All original attacks retained; no acoustic offset truth.'
            if reference is not None:
                for kind, prediction in (('baseline', base), ('refined', refined)):
                    notes = [Note(round(n['onset']*1000), round(n['offset']*1000), pitch=n['pitch'], source='pitch')
                        for n in prediction if not n['is_drum'] and 0 <= n['onset'] < end]
                    row['metrics'][kind] = dict(pitched_onsets=pitched_match(notes, reference),
                        pitched_intervals=pitched_match(notes, reference, True),
                        false_long_on_repeated_pitches=sum(n.pitch in (60, 64, 67) and n.end-n.start >= 800 for n in notes)
                            if name == 'control-repeated-chords' else None)
        rows.append(row)
    result = dict(plan=plan, rows=rows, enabled_in_application=False,
        protocol=['All frozen cases retained. No search over shifts, match tolerance or boundary margin.',
            'Same first-pass attacks and exact programs/pitches. Second pass only supplies selected explicit predicted tails.',
            'Physical audio clock; no mapping anticipation or grid snapping.',
            'Complete URMP intervals exclude evaluation-end-crossing notes and report those counts.',
            'Synthetic controls use their existing pitch metric and exclude drum predictions.',
            'Edited-note error diagnostics do not select or filter predictions.'])
    (args.root/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (args.root/'edit-diagnostics.json').write_text(json.dumps(private, indent=2), encoding='utf-8')
    print(json.dumps([dict(name=r['name'], changed=r['refinement']['changed'], metrics=r['metrics'],
        edited_reference_diagnostics=r.get('edited_reference_diagnostics')) for r in rows]), flush=True)


if __name__ == '__main__':
    main()
