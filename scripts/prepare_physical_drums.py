"""Prepare fixed source-group/song splits with physical drum annotations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha
from scripts.physical_drum_features import load_audio,features,target_frames,RATE,HOP,CLASSES

MDB_TRAIN={'80sRock','BebopJazz','Britpop','CoolJazz','FunkJazz','FusionJazz','Reggae','Rock','Shadows'}
MDB_VALIDATION={'Disco','Rockabilly','Zeppelin'}
MDB_TEST={'Beatles','Country1','FreeJazz','Gospel','Grunge','Hendrix','LatinJazz','ModalJazz','Punk','SpeedMetal','SwingJazz'}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier datasets')
    items=[];excluded=[];mapping=dict(KD='kick',SD='snare',HH='hi-hat',TT='tom',CY='cymbal',OT='other')
    root=Path('test-data/idmt-drums/data')
    for path in sorted((root/'annotation_xml').glob('*.xml')):
        tree=ET.parse(path);name=path.with_suffix('.wav').name;family=name.split('_')[0]
        metadata_name=tree.findtext('.//audioFileName')
        if metadata_name!=name and path.name!='RealDrum01_13#MIX.xml':
            raise ValueError('Unexpected XML source name discrepancy')
        if path.name=='RealDrum01_13#MIX.xml':
            excluded.append(dict(path=str(path),sha256=sha(path),reason='Embedded source name differs; KD XML has 8 events but same-name SVL has 7. Quarantined before training.'))
            continue
        split={'RealDrum01':'train','TechnoDrum01':'train','WaveDrum01':'train','TechnoDrum02':'validation','WaveDrum02':'test'}[family]
        events=[dict(time_ms=float(e.findtext('onsetSec'))*1000,kind=mapping[e.findtext('instrument')]) for e in tree.findall('.//transcription/event')]
        discrepancies=[]
        if metadata_name!=name:
            for code in ('KD','SD','HH'):
                sidecar=root/'annotation_svl'/(path.stem.replace('#MIX','#'+code)+'.svl')
                svl=ET.parse(sidecar);rate=float(svl.find('.//model').get('sampleRate'))
                svl_times=sorted(float(point.get('frame'))/rate*1000 for point in svl.findall('.//point'))
                xml_times=sorted(event['time_ms'] for event in events if event['kind']==mapping[code])
                if len(svl_times)!=len(xml_times) or not np.allclose(svl_times,xml_times,atol=.1,rtol=0):
                    raise ValueError('Ambiguous IDMT XML/SVL identity')
                discrepancies.append(dict(metadata_audio=metadata_name,matched_audio=name,svl_sha256=sha(sidecar),instrument=code))
        items.append(dict(id='idmt-'+path.stem,dataset='IDMT',group=family,split=split,audio=root/'audio'/name,annotation=path,events=events,metadata_name_audit=discrepancies))
    root=Path('test-data/mdb-drums/MDB Drums')
    for path in sorted((root/'annotations/class').glob('*.txt')):
        song=path.stem.removeprefix('MusicDelta_').removesuffix('_class')
        split='train' if song in MDB_TRAIN else 'validation' if song in MDB_VALIDATION else 'test' if song in MDB_TEST else None
        if split is None:raise ValueError('Unknown official song split')
        events=[dict(time_ms=float(row.split()[0])*1000,kind=mapping[row.split()[1]]) for row in path.read_text().splitlines() if row.strip()]
        items.append(dict(id='mdb-'+song,dataset='MDB',group=song,split=split,audio=root/'audio/drum_only'/('MusicDelta_'+song+'_Drum.wav'),annotation=path,events=events))
    if len(items)!=117:raise ValueError('Unexpected physical corpus inventory')
    args.output.mkdir(parents=True);records=[];training=[];seen={}
    for item in items:
        audio_hash=sha(item['audio'])
        if audio_hash in seen:raise ValueError('Duplicate physical audio across records')
        seen[audio_hash]=item['split']
        audio=load_audio(item['audio']);x,times=features(audio);target=target_frames(times,item['events'])
        duration=len(audio)/RATE*1000
        # Published IDMT annotations contain ten attacks 0.068–0.204ms before
        # sample zero. Keep their original event times for scoring; nearest-frame
        # training naturally represents them at frame zero. Never drop targets.
        if any(not -1<=event['time_ms']<duration for event in item['events']):raise ValueError('Annotation outside audio: '+item['id'])
        data=args.output/(item['id']+'.npz');np.savez_compressed(data,features=x,times=times,targets=target)
        if item['split']=='train':training.append(x)
        records.append(dict(**{key:str(value) if isinstance(value,Path) else value for key,value in item.items()},
            audio_sha256=audio_hash,annotation_sha256=sha(item['annotation']),data_sha256=sha(data),duration_ms=duration,frames=len(times),
            near_zero_annotations=[event for event in item['events'] if event['time_ms']<0]))
    train=np.concatenate(training);mean=train.mean(axis=0);scale=np.maximum(.05,train.std(axis=0))
    np.savez_compressed(args.output/'normalization.npz',mean=mean,scale=scale)
    result=dict(records=records,excluded=excluded,classes=CLASSES,sample_rate=RATE,hop_samples=HOP,
        counts=dict(Counter(r['split'] for r in records)),train_frames=len(train),
        normalization_sha256=sha(args.output/'normalization.npz'),feature_code_sha256=sha('scripts/physical_drum_features.py'),
        protocol_sha256=sha('docs/physical-drum-training-protocol.md'),runner_sha256=sha(__file__),
        source_records=dict(idmt=sha('test-data/idmt-drums/source.json'),mdb_commit='b29e2d63c3a023506f4bf353c5b2e8a558eed135'),
        enabled_in_application=False)
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts=result['counts'],seconds={s:sum(r['duration_ms'] for r in records if r['split']==s)/1000 for s in result['counts']},train_frames=len(train))),flush=True)


if __name__=='__main__':main()
