"""Freeze a small URMP acoustic benchmark from the author's 16kHz archive.

Uses aligned Notes text, never the unaligned score MIDI. The two lowest IDs
in MT3's published test partition are selected before running any model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha

TEST_IDS=(1,2,12,13,24,25,31,38,39)
SELECTED_IDS=(1,2)
PROGRAMS=dict(vn=40,va=41,vc=42,db=43,fl=73,ob=68,cl=71,sax=65,bn=70,tpt=56,hn=60,tbn=57,tba=58)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/urmp'))
    cli.add_argument('--seconds',type=int,default=30)
    args=cli.parse_args();root=args.root.resolve()
    if not 1<=args.seconds<=120:raise ValueError('Use a bounded acoustic interval')
    from scipy.io import wavfile
    archive=root/'urmp_yourmt3_16k.tar.gz';source=json.loads((root/'source.json').read_text())
    with archive.open('rb') as stream:md5=hashlib.file_digest(stream,'md5').hexdigest()
    if source['file']['checksum']!='md5:'+md5 or archive.stat().st_size!=source['file']['size']:
        raise ValueError('Archive identity mismatch')
    output=root/'benchmark'
    if (output/'manifest.json').exists():raise ValueError('Preserve the frozen benchmark')
    extracted=root/'audio-and-labels';extracted.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as stream:
        for item in stream:
            parts=PurePosixPath(item.name).parts
            if len(parts)!=3 or parts[0]!='urmp_yourmt3_16k' or not item.isfile():continue
            match=re.match(r'^(\d+)_',parts[1])
            if not match or int(match[1]) not in SELECTED_IDS:continue
            if not ((parts[2].startswith(('AuSep_','AuMix_')) and parts[2].endswith('.wav')) or
                    (parts[2].startswith('Notes_') and parts[2].endswith('.txt'))):continue
            target=(extracted/parts[1]/parts[2]).resolve()
            if not target.is_relative_to(extracted.resolve()):raise ValueError('Unsafe archive member')
            target.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(item) as inp,target.open('wb') as out:
                import shutil
                shutil.copyfileobj(inp,out)
    output.mkdir(exist_ok=True);cases=[]
    for piece_id in SELECTED_IDS:
        matches=[p for p in extracted.iterdir() if int(p.name.split('_')[0])==piece_id]
        if len(matches)!=1:raise ValueError('Expected one complete piece for each selected ID')
        folder=matches[0];parts=folder.name.split('_');identifier,title=parts[:2];instruments=parts[2:]
        all_truth=[];annotations={}
        for number,instrument in enumerate(instruments,1):
            label=folder/f'Notes_{number}_{instrument}_{identifier}_{title}.txt'
            truth=[]
            for line in label.read_text().splitlines():
                if not line.strip():continue
                onset,frequency,duration=map(float,line.split())
                if not (math.isfinite(onset) and math.isfinite(frequency) and math.isfinite(duration)
                        and onset>=0 and frequency>0 and duration>0):raise ValueError('Invalid acoustic label')
                truth.append(dict(onset=onset,offset=onset+duration,pitch=round(69+12*math.log2(frequency/440)),
                    program=PROGRAMS[instrument],instrument=instrument,part=number))
            audio=folder/f'AuSep_{number}_{instrument}_{identifier}_{title}.wav'
            case=dict(name=f'{piece_id:02}-{number}-{instrument}',piece_id=piece_id,kind='isolated',
                instrument=instrument,program=PROGRAMS[instrument],audio=audio,labels=[label])
            all_truth.extend(truth);annotations[str(label)]=sha(label)
            cases.append((case,truth))
        mixture=list(folder.glob('AuMix_*.wav'))
        if len(mixture)!=1:raise ValueError('Missing original mixture')
        cases.append((dict(name=f'{piece_id:02}-mix',piece_id=piece_id,kind='mixture',audio=mixture[0],
            labels=[Path(p) for p in annotations]),all_truth))
    inventory=[]
    for case,truth in cases:
        audio=case.pop('audio');labels=case.pop('labels');rate,samples=wavfile.read(audio,mmap=True)
        if rate!=16000 or samples.ndim!=1 or str(samples.dtype)!='int16':raise ValueError('Unexpected prepared PCM')
        if len(samples)/rate<args.seconds:raise ValueError('Selected piece is shorter than the fixed interval')
        truth_path=output/(case['name']+'-labels.json')
        truth_path.write_text(json.dumps(truth,indent=2),encoding='utf-8')
        inventory.append(dict(**case,audio=str(audio),audio_sha256=sha(audio),start_ms=0,seconds=args.seconds,
            labels_path=str(truth_path),labels_sha256=sha(truth_path),annotations={str(p):sha(p) for p in labels}))
    manifest=dict(revision=1,archive=source,selected_ids=SELECTED_IDS,published_mt3_test_ids=TEST_IDS,cases=inventory,
        protocol=['Select the two lowest IDs in the fixed MT3 test partition, all isolated parts and original mixtures.',
            'Predict from original audio only, no transcription labels, MIDI, chart Timing or source program hints.',
            'Evaluate onset and release on the same unshifted physical audio clock; no chart lead or grid snapping.',
            'This partition is also used for upstream validation; it is not a newly held-out model test set.',
            'Aligned Notes annotations can have boundary uncertainty; report strict and duration-relative tolerances.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(cases=len(inventory),names=[c['name'] for c in inventory])),flush=True)


if __name__=='__main__':main()
