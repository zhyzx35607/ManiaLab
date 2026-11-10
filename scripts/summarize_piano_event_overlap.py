"""Check fixed overlap-fit results and independent controls without model inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.build_provenance import sha256

KINDS=('clean','plus1','plus2')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked_controls(path):
    result=read(path/'report.json')
    for name,digest in result['artifacts_sha256'].items():
        if sha256(path/name)!=digest:
            raise ValueError('Control artifact changed')
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--clean-dataset',type=Path,required=True)
    cli.add_argument('--overlap-dataset',type=Path,required=True)
    cli.add_argument('--controls',type=Path,required=True)
    cli.add_argument('--clean-controls',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing summaries')
    run=read(args.run/'report.json')
    for key,digest in run['identity']['code_sha256'].items():
        suffix='.md' if key=='protocol' else '.py'
        if sha256(args.run/(key+'-snapshot'+suffix))!=digest:
            raise ValueError('Training snapshot changed')
    folders=dict(clean=args.clean_dataset,plus1=args.overlap_dataset/'plus1',plus2=args.overlap_dataset/'plus2')
    totals={}
    for kind,folder in folders.items():
        if sha256(folder/'manifest.json')!=run['identity']['dataset_sha256'][kind]:
            raise ValueError('Training data manifest changed')
        manifest=read(folder/'manifest.json')
        total={split:Counter() for split in ('train','validation')}
        for item in manifest['items']:
            if sha256(folder/item['cache'])!=item['cache_sha256']:
                raise ValueError('Training cache changed')
            total[item['split']].update(item['counts'])
            total[item['split']]['windows']+=1
        totals[kind]=total
    if sha256(args.overlap_dataset/'manifest.json')!=run['identity']['dataset_sha256']['augmentation']:
        raise ValueError('Augmentation manifest changed')
    epochs=[]
    for epoch in range(run['identity']['epochs']+1):
        p=args.run/f'validation-epoch-{epoch}.json'
        value=read(p)
        for kind,result in value['variants'].items():
            pooled=Counter();eligible=[]
            for case in result['cases']:
                c={k:case['voice'][k] for k in ('tp','fp','fn','tn')}
                assert sum(c.values())==case['supervised_edges']
                f1=2*c['tp']/max(1,2*c['tp']+c['fp']+c['fn'])
                assert f1==case['voice']['f1']
                pooled.update(c)
                if c['tp']+c['fn']:
                    eligible.append(f1)
            assert all(pooled[k]==result['voice'][k] for k in pooled)
            assert sum(pooled.values())==totals[kind]['validation']['voice_supervised']
            assert pooled['tp']+pooled['fn']==totals[kind]['validation']['voice_positives']
            assert abs(float(np.mean(eligible))-result['macro_voice_f1'])<1e-12
            assert sum(c['staff_correct'] for c in result['cases'])==result['staff_correct']
            assert sum(c['staff_notes'] for c in result['cases'])==result['staff_notes']
        score=float(np.mean([value['variants'][k]['macro_voice_f1'] for k in KINDS]))
        assert abs(score-value['selection_score'])<1e-12
        epochs.append(dict(epoch=epoch,evaluation_sha256=sha256(p),**value))
    selected=max(range(len(epochs)),key=lambda i:epochs[i]['selection_score'])
    assert selected==run['selected_epoch']
    assert sha256(args.run/'selected.pt')==sha256(args.run/f'epoch-{selected}.pt')==run['selected_sha256']
    controls=checked_controls(args.controls)
    clean_controls=checked_controls(args.clean_controls)
    assert controls['training_report_sha256']==sha256(args.run/'report.json')
    assert controls['selected_sha256']==run['selected_sha256']
    old_original={c['control']:c for c in clean_controls['cases'] if c['model']=='original'}
    for row in controls['cases']:
        if row['model']=='original':
            assert row==old_original[row['control']]
            file='original-'+row['control']+'.npz'
            with np.load(args.controls/file,allow_pickle=False) as new, np.load(args.clean_controls/file,allow_pickle=False) as old:
                assert new.files==old.files
                assert all(np.array_equal(new[k],old[k]) for k in new.files)
    changes={}
    for kind in KINDS:
        baseline={r['name']:r for r in epochs[0]['variants'][kind]['cases']}
        changes[kind]=[dict(name=r['name'],
            voice_f1_delta=r['voice']['f1']-baseline[r['name']]['voice']['f1'],
            staff_correct_delta=r['staff_correct']-baseline[r['name']]['staff_correct'])
            for r in epochs[selected]['variants'][kind]['cases']]
    result=dict(identity=run['identity'],training_report_sha256=sha256(args.run/'report.json'),
        dataset_totals=totals,epochs=epochs,selected_epoch=selected,selected_sha256=run['selected_sha256'],
        selected_piece_changes=changes,controls=controls['cases'],
        clean_adaptation_controls=clean_controls['cases'],
        controls_sha256=sha256(args.controls/'report.json'),
        clean_controls_sha256=sha256(args.clean_controls/'report.json'),
        elapsed_seconds=run['elapsed_seconds'],summarizer_sha256=sha256(__file__),
        model_rerun=False,test_loaded=False,application_enabled=False,musical_acceptance=False,
        limitations=['Controls were already seen and informed augmentation; they are development checks.',
          'Validation uses three observations of the same nine symbolic pieces, not 27 independent pieces.',
          'Local masked successors do not establish global melody identity, acoustic precision, or LN quality.'])
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected_epoch=selected,
        baseline=epochs[0]['selection_score'],selected=epochs[selected]['selection_score'],changes=changes)),flush=True)


if __name__=='__main__':
    main()
