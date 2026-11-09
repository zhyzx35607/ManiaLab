"""Run the existing melodic source detector on frozen acoustic benchmark cases."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.audio import infer,pitched_notes,rhythmic_evidence,source_evidence,refine_releases,fuse_attacks
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/urmp/benchmark'))
    args=cli.parse_args()
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    manifest=json.loads((args.root/'manifest.json').read_text());model=Path('assets/basic-pitch.onnx')
    for case in manifest['cases']:
        folder=args.root/'application'/case['name'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'report.json').exists():raise ValueError('Preserve frozen detector output')
        if sha(case['audio'])!=case['audio_sha256']:raise ValueError('Audio identity mismatch')
        rate,y=wavfile.read(case['audio'])
        if rate!=16000 or y.ndim!=1 or y.dtype!=np.int16:raise ValueError('Unexpected PCM')
        begin=max(0,round((case['start_ms']/1000-2.1)*rate))
        finish=min(len(y),round((case['start_ms']/1000+case['seconds']+2.1)*rate))
        if finish<=begin:raise ValueError('Requested interval is outside audio')
        y=y[begin:finish].astype(np.float32)/32768
        signal=resample_poly(y,441,320).astype(np.float32);began=time.monotonic()
        frames,onsets,times=infer(signal,model)
        attacks,_=rhythmic_evidence(signal,estimate=False)
        attacks,_=source_evidence(attacks,signal,signal)
        notes=pitched_notes(frames,onsets,times,len(signal)/22050,attacks=attacks)
        notes,_=source_evidence(notes,signal,signal)
        notes,releases=refine_releases(notes,signal)
        notes,_,_=fuse_attacks(notes,attacks)
        (folder/'notes.json').write_text(json.dumps([dict(onset=begin/rate+n.start/1000,offset=begin/rate+n.end/1000,
            pitch=n.pitch,program=-1,is_drum=False) for n in notes],indent=2),encoding='utf-8')
        report=dict(identity=dict(sources={case['audio']:case['audio_sha256']},model_sha256=sha(model),
            audio_code_sha256=sha('manialab/audio.py'),runner_sha256=sha(__file__),
            start_ms=case['start_ms'],seconds=case['seconds'],context_seconds=2.1),
            elapsed_seconds=time.monotonic()-began,release_refinement=releases,
            context_start_seconds=begin/rate,context_end_seconds=finish/rate,
            note_count=len(notes),notice='Existing melodic detector; no instrument classifier or transcription labels.')
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(name=case['name'],**report)),flush=True)


if __name__=='__main__':main()
