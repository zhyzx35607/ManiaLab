"""Re-score all saved HF cases after the legacy URMP aggregation failure; no NN."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import check_hf_instrument_specialists as comparison

read,require,verify,sha=comparison.read,comparison.require,comparison.verify,comparison.sha


def recover(source,output):
    source,output=Path(source),Path(output)
    require(not output.exists(),'Preserve existing summaries')
    plan=read(source/'plan.json')
    require(plan['order']==comparison.CASE_NAMES and
            [c['name'] for c in plan['cases']]==comparison.CASE_NAMES,'Incomplete frozen plan')
    runner=Path(comparison.__file__).resolve()
    original_runner=plan['code_sha256'][str(runner)]
    require(original_runner=='886582bc8ca3de6cc5db7028bc253c854849d49bdad50c3e5cb2cc4177b902c3',
            'This recovery is for the recorded legacy availability aggregation failure')
    for path,digest in plan['code_sha256'].items():
        if Path(path).resolve()!=runner:verify(path,digest)
    for path,digest in plan['cohort_files'].items():verify(path,digest)
    current=comparison.prepare()  # Validates source, weights, baselines and physical clocks; no network.
    rows=[];evidence=[];identities={}
    for original,case in zip(plan['cases'],current['cases']):
        require({k:v for k,v in original.items() if k!='baseline'}==
                {k:v for k,v in case.items() if k!='baseline'},'Frozen case changed')
        path=source/case['name']/'metrics.json'
        saved=read(path)
        require(saved['name']==case['name'] and saved['model']==case['model'], 'Case identity changed')
        folder=Path(saved['inference_folder'])
        verify(folder/'report.json',saved['report_sha256'])
        identity=read(folder/'report.json')['identity']
        expected=identities.setdefault(case['model'],identity)
        require(identity==expected and identity['model']==case['model']
                and identity['revision']==comparison.adapter.REVISION
                and identity['source']==plan['source']
                and identity['checkpoint_sha256']==plan['weights'][case['model']]['sha256']
                and identity['adapter_sha256']==sha(comparison.adapter.__file__)
                and identity['decoder_sha256']==sha(comparison.ROOT/'manialab/piano.py')
                and identity['strict_weights'] is True
                and identity['thresholds']==plan['thresholds'], 'Saved inference identity changed')
        notes,report=comparison.verified_reuse(folder,expected,case)
        verify(folder/'notes.json',saved['notes_sha256'])
        require(report['seconds']==saved['seconds'] and report['release_reasons']==saved['release_reasons']
                and report['context_end_seconds']==case['context_end'], 'Saved inference metadata changed')
        reference=read(case['labels'])
        if case['model']=='bass':
            result=comparison.score_case(reference,notes,end=case['scoring_end'],context_end=case['context_end'])
        else:
            metrics,scope=comparison.window_metrics(reference,notes,begin=0,end=case['scoring_end'],
                context_start=0,context_end=case['context_end'])
            result=dict(metrics=comparison.normalized_metrics(metrics,scope),interval_scope=scope)
        prior=saved['engines']['hf_specialist']
        prior=dict(prior,metrics=comparison.normalized_metrics(prior['metrics'],result['interval_scope']))
        require(result==prior,'Re-scored physical predictions differ from saved metrics')
        require(result['metrics']['standard']['pitched_onsets']['reference']==case['scoring_reference'],
                'Scoring denominator changed')
        require({k:v for k,v in saved['engines'].items() if k!='hf_specialist'}==original['baseline'],
                'Saved baseline differs from frozen plan')
        rows.append(dict(saved,engines=dict(case['baseline'],hf_specialist=result)))
        evidence.append(dict(metrics_path=str(path.resolve()),metrics_sha256=sha(path),
                             report_sha256=saved['report_sha256']))
    result=comparison.summarize(plan,rows)
    result['summary_recovery']=dict(original_plan_sha256=sha(source/'plan.json'),
        original_runner_sha256=original_runner,runner_sha256=sha(runner),
        recovery_script_sha256=sha(__file__),evidence=evidence,network_loaded=False,
        reason='Legacy URMP scores lack available; all reference tails verified in physical context.',
        scoring_recomputed=True,original_files_modified=False)
    comparison.adapter.write_new(output/'report.json',result)
    return result


def main():
    import json
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    result=recover(args.source,args.output)
    require('torch' not in sys.modules,'Summary must never load Torch')
    print(json.dumps(dict(cases=len(result['rows']),totals=result['totals'],network_loaded=False)),flush=True)


if __name__=='__main__':main()
