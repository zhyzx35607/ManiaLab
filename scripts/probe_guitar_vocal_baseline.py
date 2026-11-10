"""Run the unchanged generic detector on every fixed guitar/vocal recording."""
import argparse
from collections import Counter
import json
from math import gcd
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,required=True)
    args=cli.parse_args()
    import numpy as np
    import onnxruntime as ort
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from manialab import audio
    manifest=json.loads((args.root/'manifest.json').read_text(encoding='utf-8'))
    model=Path('assets/basic-pitch.onnx')
    identity=dict(manifest_sha256=sha(args.root/'manifest.json'),model_sha256=sha(model),
                  audio_code_sha256=sha('manialab/audio.py'),runner_sha256=sha(__file__),
                  intra_threads=2,inter_threads=1,normalization=False,inference_scope='full_original_audio')
    output=args.root/'application';output.mkdir(exist_ok=True)
    plan=output/'plan.json'
    if plan.exists():
        if json.loads(plan.read_text())!=identity:
            raise ValueError('Frozen baseline plan changed')
    else:
        plan.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    audio._session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    for index,case in enumerate(manifest['cases']):
        folder=output/case['name'];folder.mkdir(exist_ok=True)
        expected=dict(identity,audio_sha256=case['audio_sha256'],audio=case['audio'])
        if sha(case['audio'])!=case['audio_sha256']:
            raise ValueError('Frozen audio changed: '+case['name'])
        if (folder/'report.json').exists():
            report=json.loads((folder/'report.json').read_text())
            if (report['identity']!=expected or sha(folder/'notes.json')!=report['notes_sha256']
                    or sha(folder/'activations.npz')!=report['activations_sha256']):
                raise ValueError('Existing output changed: '+case['name'])
            continue
        if any(folder.iterdir()):
            raise ValueError('Partial output exists; inspect before replacing: '+case['name'])
        rate,wave=wavfile.read(case['audio'])
        if wave.ndim!=1 or wave.dtype!=np.int16 or len(wave)!=case['samples'] or rate!=case['sample_rate']:
            raise ValueError('Original source format changed')
        signal=wave.astype(np.float32)/32768
        common=gcd(rate,audio.SR)
        signal=resample_poly(signal,audio.SR//common,rate//common).astype(np.float32)
        began=time.monotonic()
        frames,onsets,times=audio.infer(signal,model)
        np.savez_compressed(folder/'activations.npz',frames=frames,onsets=onsets,times=times)
        attacks,_=audio.rhythmic_evidence(signal,estimate=False)
        attacks,_=audio.source_evidence(attacks,signal,signal)
        notes=audio.pitched_notes(frames,onsets,times,len(signal)/audio.SR,attacks=attacks)
        notes,_=audio.source_evidence(notes,signal,signal)
        notes,releases=audio.refine_releases(notes,signal)
        notes,_,_=audio.fuse_attacks(notes,attacks)
        events=[dict(onset=n.start/1000,offset=n.end/1000,pitch=n.pitch,program=-1,is_drum=False,
                     strength=n.strength,release_reason=n.release_reason) for n in notes]
        (folder/'notes.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        report=dict(identity=expected,elapsed_seconds=time.monotonic()-began,
            context_start_seconds=0.,context_end_seconds=len(wave)/rate,
            resampled_duration_seconds=len(signal)/audio.SR,
            notes_sha256=sha(folder/'notes.json'),activations_sha256=sha(folder/'activations.npz'),
            notes=len(notes),release_reasons=dict(Counter(n.release_reason for n in notes)),
            release_refinement=releases)
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(index=index+1,total=len(manifest['cases']),case=case['name'],
                             notes=len(notes),elapsed_seconds=report['elapsed_seconds'])),flush=True)
    case_reports={case['name']:dict(report_sha256=sha(output/case['name']/'report.json'),
        notes_sha256=sha(output/case['name']/'notes.json'),
        activations_sha256=sha(output/case['name']/'activations.npz')) for case in manifest['cases']}
    (output/'completed.json').write_text(json.dumps(dict(identity=identity,cases=len(manifest['cases']),
        plan_sha256=sha(plan),case_reports=case_reports),indent=2),encoding='utf-8')


if __name__=='__main__':main()
