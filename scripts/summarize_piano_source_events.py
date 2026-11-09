"""Verify source-event fit identities and publish aggregate validation results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.build_provenance import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve prior summaries')
    report=read(args.run/'report.json');manifest=read(args.dataset/'manifest.json')
    if report['identity']['dataset_sha256']!=sha256(args.dataset/'manifest.json'):
        raise ValueError('Changed training dataset')
    for key,digest in report['identity']['code_sha256'].items():
        extension='.md' if key=='protocol' else '.py'
        if sha256(args.run/(key+'-snapshot'+extension))!=digest:
            raise ValueError('Changed execution snapshot')
    if sha256(args.run/'selected.pt')!=report['selected_sha256']:
        raise ValueError('Changed selected weights')
    totals={split:Counter() for split in ('train','validation')}
    seen_pairs=set();seen_nodes=set()
    for row in manifest['items']:
        path=args.dataset/row['cache']
        if sha256(path)!=row['cache_sha256']:raise ValueError('Changed event graph')
        with np.load(path,allow_pickle=False) as data:
            ids=data['event_ids'];voice=data['voice'];mask=data['voice_mask'];sm=data['staff_mask']
            actual=dict(voice_candidates=len(voice),voice_supervised=int(mask.sum()),
                voice_positives=int(voice[mask].sum()),chord_supervised=int(data['chord_mask'].sum()),
                staff_supervised=int(sm.sum()),carry_in=int(data['carry_in'].sum()),
                carry_out=int(data['carry_out'].sum()))
            for field,count in actual.items():assert count==row['counts'][field],(field,row['name'])
            for a,b in data['potential_edges'][:,mask].T:
                key=(str(ids[a]),str(ids[b]));assert key not in seen_pairs;seen_pairs.add(key)
            for identity in ids[sm]:
                assert str(identity) not in seen_nodes;seen_nodes.add(str(identity))
            totals[row['split']].update(actual)
    epochs=[]
    for epoch in range(report['identity']['epochs']+1):
        result=read(args.run/f'validation-epoch-{epoch}.json')
        pooled=Counter();macros=[]
        for case in result['cases']:
            counts={key:case['voice'][key] for key in ('tp','fp','fn','tn')}
            assert sum(counts.values())==case['supervised_edges']
            f1=2*counts['tp']/max(1,2*counts['tp']+counts['fp']+counts['fn'])
            assert f1==case['voice']['f1']
            pooled.update(counts)
            if counts['tp']+counts['fn']:macros.append(f1)
        assert all(pooled[k]==result['voice'][k] for k in pooled)
        assert sum(pooled.values())==totals['validation']['voice_supervised']
        assert pooled['tp']+pooled['fn']==totals['validation']['voice_positives']
        assert abs(np.mean(macros)-result['macro_voice_f1'])<1e-12
        epochs.append(dict(epoch=epoch,voice=result['voice'],macro_voice_f1=result['macro_voice_f1'],
            staff_correct=result['staff_correct'],staff_notes=result['staff_notes'],
            evaluation_sha256=sha256(args.run/f'validation-epoch-{epoch}.json'),cases=result['cases']))
    selected=max(range(len(epochs)),key=lambda i:epochs[i]['macro_voice_f1'])
    assert selected==report['selected_epoch']
    assert sha256(args.run/f'epoch-{selected}.pt')==report['selected_sha256']
    first={r['name']:r for r in epochs[0]['cases']}
    changes=[dict(name=r['name'],voice_f1_delta=r['voice']['f1']-first[r['name']]['voice']['f1'],
        staff_correct_delta=r['staff_correct']-first[r['name']]['staff_correct']) for r in epochs[selected]['cases']]
    result=dict(identity=report['identity'],training_report_sha256=sha256(args.run/'report.json'),
        summarizer_sha256=sha256(__file__),dataset_totals=totals,source_files=manifest['source_files'],
        ready_files=manifest['ready_files'],graph_files_verified=len(manifest['items']),
        unique_supervised_pairs=len(seen_pairs),unique_staff_nodes=len(seen_nodes),
        selected_epoch=selected,selected_sha256=report['selected_sha256'],epochs=epochs,
        selected_piece_changes=changes,elapsed_seconds=report['elapsed_seconds'],
        test_sources_read=False,model_rerun=False,musical_acceptance=False,
        limitations=['Only local notation successor classification on nine validation files, first sixteen sounding bars.',
          'Unsupervised cross-bar scores are not melody identities.',
          'Clean symbolic inputs do not prove acoustic-tail, chord, phrase or LN quality.'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(graphs=result['graph_files_verified'],selected_epoch=selected,
        initial_macro=epochs[0]['macro_voice_f1'],selected_macro=epochs[selected]['macro_voice_f1'],
        changes=changes)),flush=True)


if __name__=='__main__':main()
