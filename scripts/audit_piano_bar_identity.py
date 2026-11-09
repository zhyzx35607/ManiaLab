"""Audit written ties and attack identity using self-authored MusicXML cases."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256

REVISION = '1462e7c28d7adaae033150883a7cb00238ee364a'


def note(identifier, duration=480, ties=(), staff=1, voice=1):
    tags = ''.join(f'<tie type="{kind}"/>' for kind in ties)
    notation = ''.join(f'<tied type="{kind}"/>' for kind in ties)
    return (f'<note id="{identifier}"><pitch><step>C</step><octave>4</octave></pitch>'
        f'<duration>{duration}</duration>{tags}<voice>{voice}</voice><type>quarter</type>'
        f'<staff>{staff}</staff><notations>{notation}</notations></note>')


def rest(duration):
    return f'<note><rest/><duration>{duration}</duration><voice>1</voice><staff>1</staff></note>'


def xml(measures):
    attributes = ('<attributes><divisions>480</divisions><time><beats>4</beats><beat-type>4</beat-type></time>'
        '<staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef>'
        '<clef number="2"><sign>F</sign><line>4</line></clef></attributes>')
    body = ''.join(f'<measure number="{i+1}">{attributes if i == 0 else ""}{content}</measure>'
                   for i, content in enumerate(measures))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<score-partwise version="4.0">'
        '<work><work-title>ManiaLab self-authored tie audit</work-title></work>'
        '<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
        f'<part id="P1">{body}</part></score-partwise>')


def controls():
    return [
        dict(name='within_measure',measures=[note('a',ties=('start',))+note('b',ties=('stop',))+rest(960)],
             expected_ties=[['a','b']],expected_raw_events=1,expected_preprocessed_events=1,scope='same_measure'),
        dict(name='within_measure_chain',measures=[note('a',ties=('start',))+note('b',ties=('stop','start'))+
             note('c',ties=('stop',))+rest(480)],expected_ties=[['a','b'],['b','c']],
             expected_raw_events=1,expected_preprocessed_events=1,scope='same_measure'),
        dict(name='across_measure',measures=[rest(1440)+note('a',ties=('start',)),
             note('b',ties=('stop',))+rest(1440)],expected_ties=[['a','b']],
             expected_raw_events=1,expected_preprocessed_events=2,scope='cross_measure'),
        dict(name='repeated_pitch',measures=[note('a')+note('b')+rest(960)],expected_ties=[],
             expected_raw_events=2,expected_preprocessed_events=2,scope='no_tie'),
        dict(name='cross_staff',measures=[note('a',ties=('start',),staff=1)+
             note('b',ties=('stop',),staff=2)+rest(960)],expected_ties=[['a','b']],
             expected_raw_events=1,expected_preprocessed_events=1,scope='same_measure_cross_staff'),
        dict(name='single_note',measures=[note('a')+rest(1440)],expected_ties=[],
             expected_raw_events=1,expected_preprocessed_events=1,scope='no_tie'),
        dict(name='parallel_unison_ties',measures=[
             note('a1',ties=('start',),voice=1)+'<backup><duration>480</duration></backup>'+
             note('b1',ties=('start',),staff=2,voice=2)+note('b2',ties=('stop',),staff=2,voice=2)+
             '<backup><duration>480</duration></backup>'+note('a2',ties=('stop',),voice=1)+rest(960)],
             expected_ties=[['a1','a2'],['b1','b2']],expected_raw_events=2,
             expected_preprocessed_events=2,scope='parallel_unison_import_control'),
    ]


def scalar(value):
    return value.item() if isinstance(value,np.generic) else value


def array_rows(array):
    return [{name:scalar(row[name]) for name in array.dtype.names} for row in array]


def written(score,partitura):
    result=[]
    for part in score.parts:
        for n in part.iter_all(partitura.score.Note):
            result.append(dict(part=str(part.id),id=n.id,onset=int(n.start.t),end=int(n.end.t),
                pitch=int(n.midi_pitch),staff=n.staff,voice=n.voice,
                tie_prev=None if n.tie_prev is None else n.tie_prev.id,
                tie_next=None if n.tie_next is None else n.tie_next.id))
    return sorted(result,key=lambda n:(n['part'],n['onset'],n['id']))


def tie_pairs(rows):
    return sorted([n['id'],n['tie_next']] for n in rows if n['tie_next'] is not None)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous audits')
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    source_changes=lambda:subprocess.check_output(['git','-C',str(args.source),'diff','HEAD','--name-only'],text=True).strip()
    if revision!=REVISION or source_changes():raise ValueError('Upstream source identity differs')
    sys.path.insert(0,str(args.source.resolve()))
    import partitura
    from piano_svsep.utils import remove_ties_acros_barlines
    import partitura.io.importmusicxml as importer
    import partitura.utils.music as music_utils
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    identity=dict(source_revision=revision,runner_sha256=sha256(__file__),
        protocol_sha256=sha256(ROOT/'docs/piano-bar-identity-protocol.md'),
        preprocessing_sha256=sha256(args.source/'piano_svsep/utils/vocsep.py'),
        partitura_import_sha256=sha256(importer.__file__),partitura_note_array_sha256=sha256(music_utils.__file__),
        partitura_version=importlib.metadata.version('partitura'),numpy_version=np.__version__,
        input_origin='Self-authored MusicXML only',training_cache_read=False,model_loaded=False)
    save('identity.json',identity)
    results=[]
    for case in controls():
        name=case['name'];path=args.output/f'{name}.musicxml'
        path.write_text(xml(case['measures']),encoding='utf-8');input_sha=sha256(path)
        record={key:value for key,value in case.items() if key!='measures'}
        record.update(input_sha256=input_sha,status='error')
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                original=partitura.load_score(str(path))
                processed=partitura.load_score(str(path))
                before=written(original,partitura)
                assert before==written(processed,partitura),'Independent loads differ'
                raw_array=original.note_array(include_time_signature=True,include_grace_notes=True,include_staff=True)
                assert written(original,partitura)==before,'note_array mutated source ties'
                returned=remove_ties_acros_barlines(processed)
                after=written(processed,partitura)
                processed_array=processed.note_array(include_time_signature=True,include_grace_notes=True,include_staff=True)
                assert written(original,partitura)==before,'Reference branch mutated'
                assert written(processed,partitura)==after,'note_array mutated processed ties'
                assert sha256(path)==input_sha,'Original XML changed'
            geometry=lambda rows:[{k:v for k,v in n.items() if k not in ('tie_prev','tie_next')} for n in rows]
            assert geometry(before)==geometry(after),'Written note time/identity changed'
            parsed_ties=tie_pairs(before);after_ties=tie_pairs(after)
            raw_keys={(int(n['onset_div']),int(n['pitch']),str(n['id'])) for n in raw_array}
            processed_keys={(int(n['onset_div']),int(n['pitch']),str(n['id'])) for n in processed_array}
            checks=dict(imported_ties_match_authored=parsed_ties==sorted(case['expected_ties']),
                raw_event_count_expected=len(raw_array)==case['expected_raw_events'],
                preprocessed_event_count_expected=len(processed_array)==case['expected_preprocessed_events'],
                written_geometry_unchanged=True,reference_branch_unchanged=True,xml_unchanged=True)
            if case['scope']=='no_tie':
                checks['untied_input_unchanged']=np.array_equal(raw_array,processed_array)
            detail=dict(written_before=before,written_after=after,parsed_ties=parsed_ties,processed_ties=after_ties,
                preprocessing_returned_ties=returned.tolist(),raw_note_array=array_rows(raw_array),
                preprocessed_note_array=array_rows(processed_array),
                new_event_keys=[list(k) for k in sorted(processed_keys-raw_keys)],
                removed_event_keys=[list(k) for k in sorted(raw_keys-processed_keys)])
            save(name+'-detail.json',detail)
            np.savez_compressed(args.output/(name+'-arrays.npz'),raw=raw_array,preprocessed=processed_array)
            record.update(status='complete',checks=checks,parsed_tie_count=len(parsed_ties),
                processed_tie_count=len(after_ties),raw_events=len(raw_array),processed_events=len(processed_array),
                added_event_count=len(processed_keys-raw_keys),removed_event_count=len(raw_keys-processed_keys),
                returned_tie_count=len(returned[0]),warnings=[str(w.message) for w in caught])
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'
        results.append(record)
        save('progress.json',results)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    if source_changes():raise ValueError('Upstream source changed during audit')
    report=dict(identity=identity,cases=results,completed=sum(r['status']=='complete' for r in results),
        errors=sum(r['status']=='error' for r in results),
        observed_failed_expectations=sum(sum(not value for value in r.get('checks',{}).values()) for r in results),
        upstream_unchanged=True,training_cache_read=False,training_cache_changed=False,
        weights_changed=False,enabled_in_application=False,musical_acceptance=False)
    report['artifacts_sha256']={p.name:sha256(p) for p in args.output.iterdir()
        if p.name not in ('progress.json','report.json')}
    save('report.json',report)
    print(json.dumps(dict(completed=report['completed'],errors=report['errors'],
        failed_expectations=report['observed_failed_expectations'],report_sha256=sha256(args.output/'report.json'))),flush=True)


if __name__=='__main__':main()
