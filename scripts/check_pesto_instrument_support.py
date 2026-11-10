"""Measure fixed PESTO pitch evidence on all frozen URMP inputs, without tuning."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.io import wavfile
from scripts.research_pesto_pitch import PestoPredictor, note_support, sha
from scripts.evaluate_instrument_benchmark import window_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))


def counts(reference, frames, end=30.):
    times = frames['times']
    active = np.zeros(len(times), dtype=int)
    target = np.zeros(len(times))
    for note in reference:
        mask = (times >= note['onset']) & (times < note['offset'])
        active += mask
        target[mask] = note['pitch']
    eligible = frames['valid'] & (times < end)
    mono = eligible & (active == 1)
    rest = eligible & (active == 0)
    voiced = frames['voicing'] >= .5
    match = np.abs(target - frames['pitch_midi']) <= .5
    return dict(monophonic_frames=int(mono.sum()), correct_voiced_frames=int((mono&voiced&match).sum()),
        rest_frames=int(rest.sum()), rest_false_voiced=int((rest&voiced).sum()),
        polyphonic_frames=int((eligible&(active>1)).sum()),
        reference_semantics='note_pitch_intervals_not_frame_F0_annotations')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--source', type=Path, default=Path('test-data/pesto-source'))
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/rosvot-runtime'))
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    root = Path('test-data/tsumugi-urmp-comparison-v1')
    manifest_path = Path('test-data/urmp/comparison-extension/manifest.json')
    comparison_path = root/'comparison.json'
    expected = 'ff98a2e3c53671c1bcf4ec16a6c3cdff3215e17e239101dfe6883f884fbe2e75'
    if sha(comparison_path) != expected:
        raise ValueError('Frozen Tsumugi baseline differs')
    comparison, manifest = read(comparison_path), read(manifest_path)
    if sha(manifest_path) != comparison['identity']['manifest_sha256']:
        raise ValueError('Frozen URMP inputs changed')
    assert len(manifest['cases']) == len(comparison['rows']) == 35
    baseline = {row['name']:row for row in comparison['rows']}
    predictor = PestoPredictor(args.source, args.dependencies)
    args.output.mkdir(parents=True)
    plan = dict(predictor=predictor.identity, runner_sha256=sha(__file__),
        manifest_sha256=sha(manifest_path), baseline_sha256=sha(comparison_path),
        cases=[c['name'] for c in manifest['cases']], voicing_threshold=.5, pitch_tolerance_semitones=.5,
        diagnostic_retention_fraction=.5, all_thresholds_fixed_before_inference=True,
        annotation_access='scoring_only', automatic_application_filter=False)
    save(args.output/'plan.json', plan)
    # Fixed controls: silence, isolated fundamentals and simultaneous partials.
    rate = 16000
    times = np.arange(4*rate)/rate
    controls = {'silence':np.zeros(len(times),dtype=np.float32)}
    for pitch in (40, 57, 69):
        controls[f'tone-{pitch}'] = (.1*np.sin(2*np.pi*(440*2**((pitch-69)/12))*times)).astype(np.float32)
    controls['two_tones'] = (controls['tone-57']+controls['tone-69']).astype(np.float32)
    control_rows = []
    for name, pcm in controls.items():
        frames, report = predictor.predict(pcm, rate, args.output/name, source_identity=dict(control=name))
        valid = frames['valid']
        control_rows.append(dict(name=name, voiced_frames=int(((frames['voicing']>=.5)&valid).sum()),
            observed_frames=int(valid.sum()), median_pitch=float(np.median(frames['pitch_midi'][valid])),
            median_voicing=float(np.median(frames['voicing'][valid])), seconds=report['seconds']))
    save(args.output/'controls.json', control_rows)
    print(json.dumps(dict(controls=control_rows)), flush=True)
    rows = []
    for index, case in enumerate(manifest['cases']):
        name = case['name']; prior = baseline[name]
        clip, predictions_path = root/(name+'.wav'), root/name/'notes.json'
        if (sha(clip) != prior['clip_sha256'] or sha(predictions_path) != prior['notes_sha256']
                or sha(case['labels_path']) != case['labels_sha256']):
            raise ValueError('Frozen audio/predictions/labels changed')
        rate, integer = wavfile.read(clip)
        assert integer.dtype == np.int16 and integer.ndim == 1 and rate == 16000
        pcm = integer.astype(np.float32)/32768.
        frames, report = predictor.predict(pcm, rate, args.output/name,
            source_identity=dict(audio=str(clip.resolve()), sha256=sha(clip), start_sample=0, normalization=False))
        predicted, reference = read(predictions_path), read(case['labels_path'])
        support = note_support(predicted, frames)
        # This deliberately tests whether filtering is useful; it is not an
        # application rule. Unknown/no-frame events are retained, not invented.
        kept = [n for n,s in zip(predicted,support) if s['support_fraction'] is None or s['support_fraction']>=.5]
        metrics, scope = window_metrics(reference, kept, begin=0,end=30,context_start=0,context_end=len(pcm)/rate)
        metrics = {key:{m:v for m,v in values.items() if 'instrument' not in m} for key,values in metrics.items()}
        save(args.output/name/'note-support.json',support)
        save(args.output/name/'diagnostic-kept-notes.json',kept)
        row = dict(name=name,kind=case['kind'],instrument=case.get('instrument'),
            frames=counts(reference,frames), seconds=report['seconds'],
            report_sha256=sha(args.output/name/'report.json'), support_sha256=sha(args.output/name/'note-support.json'),
            baseline=prior['metrics'], diagnostic_filter=metrics, interval_scope=scope)
        rows.append(row)
        save(args.output/name/'metrics.json',row)
        print(json.dumps(dict(index=index+1,name=name,seconds=report['seconds'],
            baseline=prior['metrics']['standard']['pitched_onsets']['f1'],
            diagnostic_filter=metrics['standard']['pitched_onsets']['f1'])),flush=True)
    groups = defaultdict(list)
    for row in rows:
        for key in [row['kind']]+(['instrument/'+row['instrument']] if row['instrument'] else []):
            groups[key].append(row)
    summaries = {}
    for key, group in groups.items():
        summary = dict(cases=len(group),seconds=sum(row['seconds'] for row in group))
        for variant in ('baseline','diagnostic_filter'):
            summary[variant]={}
            for metric in ('pitched_onsets','complete_intervals'):
                sums = {k:sum(row[variant]['standard'][metric][k] for row in group) for k in ('reference','predicted','matched')}
                sums['f1']=2*sums['matched']/(sums['reference']+sums['predicted'])
                summary[variant][metric]=sums
        summary['frames']={k:sum(row['frames'][k] for row in group) for k in
            ('monophonic_frames','correct_voiced_frames','rest_frames','rest_false_voiced','polyphonic_frames')}
        summaries[key]=summary
    assert sha(__file__)==plan['runner_sha256'] and sha(Path(__file__).with_name('research_pesto_pitch.py'))==predictor.identity['runner_sha256']
    save(args.output/'report.json',dict(plan=plan,controls=control_rows,rows=rows,groups=summaries,
        enabled_in_application=False,musical_acceptance=False))
    print(json.dumps(dict(stage='complete',groups=summaries)),flush=True)


if __name__ == '__main__':main()
