"""Apply the unchanged app onset detector independently to five separated drums."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.audio import rhythmic_evidence
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import sha, event_similarity
from scripts.research_source_attack_classes import drum_event_counts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier onset evaluations')
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    report = read(args.run / 'report.json')
    manifest = read(args.root / 'manifest.json')
    original = next(row for row in manifest['audio'] if row['id'] == 'drums')
    if sha(args.root / original['path']) != report['identity']['original_audio_sha256']:
        raise ValueError('Original drum source changed')
    reference_path = args.root / 'reference.osu'
    if sha(reference_path) != manifest['charts']['reference']['chart_sha256']:
        raise ValueError('Author reference changed')
    reference = list(drum_event_counts(parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes']))
    if set(report['stems']) != {'kick', 'snare', 'toms', 'hihat', 'cymbals'}:
        raise ValueError('Incomplete drum separation')
    events = []
    for role, info in report['stems'].items():
        path = args.run / (role+'.wav')
        if sha(path) != info['sha256']:
            raise ValueError('Separated drum source changed')
        rate, audio = wavfile.read(path)
        if (rate != 44100 or audio.dtype != np.float32 or audio.shape != (report['identity']['samples'], 2)
                or not np.isfinite(audio).all()):
            raise ValueError('Unexpected drum PCM')
        mono = resample_poly(audio.mean(axis=1), 1, 2)
        notes, _ = rhythmic_evidence(mono, estimate=False)
        events.extend(dict(start_ms=n.start, kind=role, strength=n.strength) for n in notes)
    groups = {'kick':({'kick'}, 'kick'), 'snare':({'snare'}, 'snare'), 'hat_strict':({'hihat'}, 'hi-hat'),
        'hat_tom_diagnostic':({'hihat','toms'}, 'hi-hat'),
        'hat_tom_cymbals_diagnostic':({'hihat','toms','cymbals'}, 'hi-hat')}
    scores = {}
    duration = report['identity']['samples'] / 44100 * 1000
    for name, start, end in [('whole',0,duration), ('fit_region',0,162340), ('calibration_region',162340,188286), ('later_sections',188286,duration)]:
        scores[name] = {}
        for label, shift in [('raw',0), ('chart_lead_22_5ms',22.5)]:
            scores[name][label] = {}
            for kind, (sources, author) in groups.items():
                human = [(t,) for t,c in reference if c == author and start <= t < end]
                proposed = [(t,) for t in sorted({e['start_ms']-shift for e in events if e['kind'] in sources
                    and start <= e['start_ms']-shift < end})]
                scores[name][label][kind] = event_similarity(human, proposed, [40])
    args.output.mkdir(parents=True)
    (args.output/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    result = dict(separation_report_sha256=sha(args.run/'report.json'), source_identity=report['identity'],
        detector_sha256=sha('manialab/audio.py'), runner_sha256=sha(__file__), reference_sha256=sha(reference_path),
        events_sha256=sha(args.output/'events.json'), counts={role:sum(e['kind']==role for e in events) for role in report['stems']},
        scores=scores, enabled_in_application=False,
        limits=['Reference is authored selection, not exhaustive physical transcription.',
            'No beat snapping, threshold fitting, or chart conditioning; original detector on each stem.',
            'Diagnostic unions are not pure hi-hat classification accuracy. Separation does not establish note duration.'])
    (args.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(counts=result['counts'], whole=scores['whole'], later=scores['later_sections'])), flush=True)


if __name__ == '__main__':
    main()
