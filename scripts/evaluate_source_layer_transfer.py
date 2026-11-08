"""Apply a frozen Sakura layer model to the previously frozen piano control.

No fitting or calibration on the other song. Chart counts label another
author's arrangement choices, not an exhaustive piano MIDI transcription.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_source_layers import source_features, predict_layers
from scripts.research_instrument_overlap import load_run
from scripts.probe_mapper_adapter import sha, event_similarity
from manialab.domain import parse_osu


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,default=Path('test-data/sakura-benchmark/piano-layer-transfer-v1'))
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve prior experiment output')
    import numpy as np
    from scipy.io import wavfile
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    model_path=Path('test-data/sakura-benchmark/piano-yourmt3-layers/development-model.npz')
    with np.load(model_path,allow_pickle=False) as packed:model={k:packed[k] for k in packed.files}
    # Check refactoring against the exact previously generated development
    # events before using the same feature function on another recording.
    _,raw,flags=load_run(Path('test-data/sakura-benchmark/piano-yourmt3-full'))
    rate,signal=wavfile.read('test-data/sakura-benchmark/bundled/piano/preview.wav')
    groups,times,x=source_features(raw,flags,signal.astype(np.float32)/32768,rate)
    chosen=predict_layers(x,model)
    archived=read('test-data/sakura-benchmark/piano-yourmt3-layers/events.json')
    if (len(archived)!=len(times) or any(r['source_note_indices']!=g or r['start_ms']!=float(t) or r['layers']!=int(c)
        for r,g,t,c in zip(archived,groups,times,chosen))):
        raise ValueError('Shared feature extraction does not reproduce frozen development predictions')
    folder=Path('test-data/yourmt3/controls/river-flows-piano')
    report,raw,flags=load_run(folder);identity=report['identity']
    case=next(c for c in read('test-data/piano-release-probe/cases.json') if c['name']=='river-flows-piano')
    audio=Path(case['audio']);reference_path=Path('test-data/audio-extension2-10k/5317787/reference.osu')
    reference_info=read(reference_path.with_suffix('.json'))
    if (identity['sources']!={str(audio):sha(audio)} or identity['start_ms']!=0 or
        identity['seconds']!=40 or reference_info['chart_sha256']!=sha(reference_path)):
        raise ValueError('Frozen transfer case changed')
    rate,signal=wavfile.read(audio)
    if rate!=22050 or signal.dtype!=np.int16 or signal.ndim!=1:raise ValueError('Expected original mono PCM clock')
    groups,times,x=source_features(raw,flags,signal.astype(np.float32)/32768,rate)
    chosen=predict_layers(x,model)
    reference=[n for n in parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes'] if 0<=n.start<40000]
    counts=Counter(n.start for n in reference)
    modes=dict(single=np.ones(len(times),int),always_double=np.full(len(times),2),
        pitch_count=np.asarray([min(2,len(g)) for g in groups]),learned=chosen)
    metrics={mode:event_similarity([(n.start,) for n in reference],
        [(float(t),) for t,c in zip(times,layers) if 0<=t<40000 for _ in range(int(c))],[40])
        for mode,layers in modes.items()}
    result=dict(model_sha256=sha(model_path),reference_sha256=sha(reference_path),identity=identity,
        runner_sha256=sha(__file__),features_sha256=sha('scripts/research_source_layers.py'),
        reproduced_development_predictions=True,reference_layer_counts=dict(Counter(counts.values())),
        predicted_layer_counts=dict(Counter(int(c) for t,c in zip(times,chosen) if 0<=t<40000)),metrics=metrics,
        enabled_in_application=False,
        protocol=['Frozen model, class bias and 22.5ms anticipation from Sakura; no new fitting, tuning or grid snapping.',
            'First 40 seconds of the previously frozen piano control; different song and mapper.',
            'The target model has at most two layers. Higher authored layers remain scored and reported.',
            'These counts compare authored choices, not physical MIDI completeness or universally correct difficulty.',
            'This chart appears in separate layout-prior training; it was not used to fit/calibrate this layer classifier.'])
    args.output.mkdir(parents=True)
    (args.output/'events.json').write_text(json.dumps([dict(start_ms=float(t),layers=int(c),source_note_indices=g)
        for t,c,g in zip(times,chosen,groups)],indent=2),encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
