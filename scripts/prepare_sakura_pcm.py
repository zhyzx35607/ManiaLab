"""Preserve original float PCM and quantify legacy preview clipping.

Do not overwrite previous research inputs or their reports. The --verify-existing
mode re-decodes originals and proves the saved float artifact is unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--verify-existing',action='store_true')
    args=cli.parse_args();root=args.root;output=args.output
    if output.exists() and not args.verify_existing:raise ValueError('Preserve existing raw PCM')
    if args.verify_existing and not output.exists():raise ValueError('Missing saved PCM')
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for item in json.loads((root/'manifest.json').read_text(encoding='utf-8'))['audio']:
        path=root/item['path']
        if sha(path)!=item['sha256']:raise ValueError('Original source changed')
        process=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-i',str(path),
            '-vn','-ac','1','-ar','22050','-f','f32le','pipe:1'],capture_output=True,check=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        raw=np.frombuffer(process.stdout,dtype='<f4').copy()
        sr,preview=wavfile.read(root/item['preview'])
        if sr!=22050 or len(raw)!=len(preview) or not np.isfinite(raw).all():raise ValueError('PCM clock/values differ')
        if not np.array_equal((np.clip(raw,-1,1)*32767).astype(np.int16),preview):
            raise ValueError('Legacy preview is not the documented clipping transform')
        saved=output/(item['id']+'.wav')
        if args.verify_existing:
            rate,actual=wavfile.read(saved)
            if rate!=sr or actual.dtype!=np.float32 or not np.array_equal(raw,actual):raise ValueError('Saved float PCM changed')
        else:wavfile.write(saved,sr,raw)
        rows.append(dict(role=item['id'],original_sha256=item['sha256'],raw_float_sha256=sha(saved),samples=len(raw),
            peak=float(np.max(abs(raw))),exceed_one=int(np.sum(abs(raw)>1)),clipped_fraction=float(np.mean(abs(raw)>1)),
            rms=float(np.sqrt(np.mean(raw**2))),preview_exact_clip_reproduction=True))
    result=dict(rows=rows,runner_sha256=sha(__file__),ffmpeg_sha256=sha(imageio_ffmpeg.get_ffmpeg_exe()),
        source_manifest_sha256=sha(root/'manifest.json'),verified_existing=args.verify_existing,
        notice='Original float amplitudes retained, no normalization or int16 clipping. Earlier clipped-preview studies remain separate.')
    (output/('verification.json' if args.verify_existing else 'report.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
