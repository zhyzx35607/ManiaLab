"""Replay every saved Transkun control/music window with onset preservation."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.transkun_onset_evidence import collect_onset_evidence
from scripts.piano_source_controls import score_attacks
from scripts.evaluate_piano_acoustics import counts, pairs
from scripts.check_transkun_music import nova_comparison


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def compare(reference, official, preserved, strict):
    old, new = pairs(reference, official, strict=strict), pairs(reference, preserved, strict=strict)
    old_ids, new_ids = {i for i, _ in old}, {i for i, _ in new}
    return dict(official=counts(len(old), len(reference), len(official)),
        preserved=counts(len(new), len(reference), len(preserved)),
        recovered_ref_indices=sorted(new_ids-old_ids), new_missing_ref_indices=sorted(old_ids-new_ids),
        official_matches=old, preserved_matches=new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local research directory')
    paths = dict(runner=Path(__file__), decoder=ROOT/'scripts/transkun_onset_evidence.py',
        control_matcher=ROOT/'scripts/piano_source_controls.py',
        music_matcher=ROOT/'scripts/evaluate_piano_acoustics.py',
        nova_matcher=ROOT/'scripts/check_transkun_music.py',
        author_matcher=ROOT/'scripts/probe_mapper_adapter.py',
        domain=ROOT/'manialab/domain.py', protocol=ROOT/'docs/piano-onset-preservation-protocol.md')
    fixed = [('controls', '46ec799e1e8e170d6e2cfa1fe0e36009249132d62db9acc886aa17a1328e88c0'),
             ('music', 'fe56f7006cba5bbe2e51cdcc1672f941f51428af4d3074c746f33adeb5780b9c')]
    sources = {}
    for scope, digest in fixed:
        folder = ROOT/f'test-data/piano-transkun-{scope}-v1'
        if sha256(folder/'report.json') != digest:
            raise ValueError('Original run changed')
        report = read(folder/'report.json')
        for name, value in report['artifacts_sha256'].items():
            if sha256(folder/name) != value:
                raise ValueError('Original evidence changed: '+name)
        sources[scope] = (folder, report)
    identity = dict(code={k:sha256(p) for k,p in paths.items()}, reports=dict(fixed),
                    training_performed=False, new_inference_performed=False)
    output.mkdir(parents=True)
    save(output/'identity.json', identity)
    for key, path in paths.items():
        shutil.copyfile(path, output/(key+'-snapshot'+path.suffix))
    rows = []
    for scope in ('controls', 'music'):
        folder, report = sources[scope]
        plans = ({p['name']: p for p in report['identity']['plans']} if scope == 'music' else {})
        for item in report['cases']:
            name = item['name']
            source_path = folder/(name+'-result.json')
            source = read(source_path)
            evidence = collect_onset_evidence(source['trace'])
            official = source['trace']['piano_events']
            preserved = [c['note'] for c in evidence['candidates']]
            recovered_official = sorted((c['note'] for c in evidence['candidates'] if c['origin']=='official'),
                                       key=lambda n:(n['onset'], n['offset'], n['pitch']))
            if recovered_official != sorted(official, key=lambda n:(n['onset'], n['offset'], n['pitch'])):
                raise ValueError('Official events changed')
            row = dict(scope=scope, name=name, added=len(preserved)-len(official),
                source_sha256=sha256(source_path),
                onset_outside_input=sum(c['onset_outside_input'] for c in evidence['candidates']))
            if scope == 'controls':
                row['scores'] = dict(official=source['score'],
                    preserved=score_attacks(source['expected'], preserved, tolerance_seconds=.1))
            elif plans[name]['kind'] == 'smd':
                plan = plans[name]
                if sha256(ROOT/plan['labels']) != plan['labels_sha256']:
                    raise ValueError('Reference changed')
                in_scope = lambda ns:[n for n in ns if 0 <= n['onset'] < plan['score_seconds']]
                reference = in_scope(read(ROOT/plan['labels']))
                row['comparisons'] = {label: compare(reference, in_scope(official), in_scope(preserved), strict)
                                      for label, strict in [('50ms',False), ('40ms',True)]}
            else:
                reference = ROOT/'test-data/sakura-benchmark/reference.osu'
                if sha256(reference) != report['identity']['files']['reference']:
                    raise ValueError('Author reference changed')
                # Rename helper's original engine labels to avoid misreporting
                # this same-model decoder experiment as another Kong inference.
                value = nova_comparison(official, preserved, reference.read_text(encoding='utf-8-sig'))
                row['nova'] = dict(candidates=dict(official=len(official), preserved=len(preserved)),
                    author_onset_diagnostic={'official':value['author_onset_diagnostic']['kong'],
                                             'preserved':value['author_onset_diagnostic']['transkun']},
                    acoustic_pitch_truth_available=False, acoustic_offset_truth_available=False)
            save(output/(scope+'-'+name+'-result.json'), dict(**row, evidence=evidence))
            rows.append(row)
    controls = [r for r in rows if r['scope']=='controls']
    music = [r for r in rows if 'comparisons' in r]
    summary = dict(controls={engine:{key:sum(r['scores'][engine][key] for r in controls)
                    for key in ('expected','predicted','matched','missing','extra')}
                    for engine in ('official','preserved')}, music={},
                   additions={scope:sum(r['added'] for r in rows if r['scope']==scope) for scope in sources})
    for tolerance in ('50ms','40ms'):
        comparisons = [r['comparisons'][tolerance] for r in music]
        summary['music'][tolerance] = dict(
            **{engine:counts(*(sum(c[engine][k] for c in comparisons) for k in ('matched','reference','predicted')))
               for engine in ('official','preserved')},
            recovered=sum(len(c['recovered_ref_indices']) for c in comparisons),
            new_missing=sum(len(c['new_missing_ref_indices']) for c in comparisons))
    target = {r['name']:r for r in music}
    checks = dict(known_two_recovered=604 in target['ravel']['comparisons']['50ms']['recovered_ref_indices']
                  and 41 in target['skryabin']['comparisons']['50ms']['recovered_ref_indices'],
        controls_extra_not_increased=summary['controls']['preserved']['extra']<=summary['controls']['official']['extra'],
        music_extra_not_increased=all(c['preserved']['predicted']-c['preserved']['matched']
            <= c['official']['predicted']-c['official']['matched'] for c in summary['music'].values()))
    if len(rows)!=103 or len(controls)!=91 or len(music)!=11:
        raise ValueError('Incomplete replay')
    if any(sha256(p)!=identity['code'][k] for k,p in paths.items()):
        raise ValueError('Code changed during replay')
    result = dict(identity=identity, cases=rows, summary=summary, checks=checks,
                  all_checks_passed=all(checks.values()), musical_acceptance=False, application_enabled=False,
                  artifacts_sha256={p.name:sha256(p) for p in output.iterdir()})
    save(output/'report.json', result)
    print(json.dumps(dict(summary=summary, checks=checks)), flush=True)


if __name__ == '__main__':
    main()
