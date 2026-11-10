"""Compare a ranked model's saved control predictions with the fixed BCE baseline."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.build_provenance import sha256
from scripts.check_piano_successor_decode import diagnose


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked(folder):
    report=read(folder/'report.json')
    for name,digest in report['artifacts_sha256'].items():
        if sha256(folder/name)!=digest:raise ValueError('Changed control input: '+name)
    return report


def paired_truth(actual,mask,old,new):
    return dict(tp_to_fn=int((mask & actual & old & ~new).sum()),
        fn_to_tp=int((mask & actual & ~old & new).sum()),
        fp_to_tn=int((mask & ~actual & old & ~new).sum()),
        tn_to_fp=int((mask & ~actual & ~old & new).sum()))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--controls',type=Path,required=True)
    cli.add_argument('--training',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve prior comparisons')
    base=ROOT/'test-data/piano-svsep'
    old_folder=base/'source-event-overlap-controls-v1'
    decode_folder=base/'successor-decode-v1'
    new_report=checked(args.controls);old_report=checked(old_folder);decode_report=checked(decode_folder)
    training=checked(args.training)
    if (new_report['selected_sha256']!=training['selected_sha256'] or
            sha256(args.training/'selected.pt')!=training['selected_sha256'] or
            new_report['training_report_sha256']!=sha256(args.training/'report.json')):
        raise ValueError('Control weights differ from ranked training')
    if old_report['code_sha256']!=new_report['code_sha256']:
        raise ValueError('Different control construction')
    if decode_report['identity']['weights_sha256']['overlap_adapted']!=old_report['selected_sha256']:
        raise ValueError('Different decoder baseline model')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    cases=[]
    for old_row in decode_report['controls']:
        if old_row['model']!='overlap_adapted':continue
        name=old_row['control'];filename='source_event_adapted-'+name+'.npz'
        with np.load(args.controls/('original-'+name+'.npz'),allow_pickle=False) as current,\
             np.load(old_folder/('original-'+name+'.npz'),allow_pickle=False) as previous:
            assert current.files==previous.files
            assert all(np.array_equal(current[k],previous[k]) for k in current.files)
        with np.load(args.controls/filename,allow_pickle=False) as current,\
             np.load(old_folder/filename,allow_pickle=False) as previous:
            assert np.array_equal(current['potential_edges'],previous['potential_edges'])
            new_logits=current['voice_logits'].copy();old_logits=previous['voice_logits'].copy()
        with np.load(decode_folder/old_row['file'],allow_pickle=False) as saved:
            data={k:saved[k].copy() for k in saved.files if k!='selected'}
            before=saved['selected'].copy()
        assert np.array_equal(data['logits'],old_logits)
        original,_=diagnose(data)
        assert all(original[key]==old_row[key] for key in original)
        data['logits']=new_logits
        result,after=diagnose(data)
        np.savez_compressed(args.output/(name+'.npz'),**data,baseline_logits=old_logits,
                            baseline_selected=before,selected=after)
        cases.append(dict(control=name,baseline=original,ranked=result,
            raw_paired=paired_truth(data['actual'],data['score_mask'],old_logits>=0,new_logits>=0),
            decoded_paired=paired_truth(data['actual'],data['score_mask'],before,after)))
    if len(cases)!=6:raise ValueError('Unexpected control suite')
    report=dict(identity=dict(training_report_sha256=sha256(args.training/'report.json'),
        controls_report_sha256=sha256(args.controls/'report.json'),baseline_report_sha256=sha256(old_folder/'report.json'),
        decoder_report_sha256=sha256(decode_folder/'report.json'),runner_sha256=sha256(__file__),
        baseline_weights_sha256=old_report['selected_sha256'],ranked_weights_sha256=new_report['selected_sha256']),
        cases=cases,new_model_inference=False,development_controls=True,musical_acceptance=False,
        artifacts_sha256={p.name:sha256(p) for p in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in cases:
        print(json.dumps(dict(control=row['control'],raw=row['ranked']['raw'],decoded=row['ranked']['decoded'],
                             raw_paired=row['raw_paired'],decoded_paired=row['decoded_paired'])),flush=True)


if __name__=='__main__':main()
