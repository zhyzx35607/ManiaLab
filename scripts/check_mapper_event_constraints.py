"""Check the optional grammar against all saved real cold-start targets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,default=Path('test-data/sakura-benchmark/expression-lora-prefix-v1'))
    args=cli.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import torch
    from assets.mania_event_constraints import ManiaEventState
    tokenizer=json.loads((root/'test-data/mapperatorinator/model/gamemode=3/tokenizer.json').read_text())
    inventory=json.loads((args.run/'samples.json').read_text())
    key=hashlib.sha256(json.dumps(inventory['identity'],sort_keys=True).encode()).hexdigest()[:16]
    starts,ends=tokenizer['event_start'],tokenizer['event_end'];rows=[]
    for chart in inventory['charts']:
        entries=torch.load(args.run.parent/'encoded'/f'{key}-{chart["id"]}.pt',weights_only=True)['samples']
        for entry in entries:
            if entry['window']['kind']!='cold_start':continue
            labels=entry['batch']['labels'][0].tolist()
            begin=next(i for i,t in enumerate(labels) if starts['t']<=t<ends['t'])
            state=ManiaEventState(starts,ends,chart['keys'],[2,tokenizer['context_eos']['map']])
            consumed=0
            for token in labels[begin:]:
                if token<0:continue
                try:state.consume(token)
                except ValueError as exc:
                    raise ValueError(f'{chart["split"]} {entry["window"]} event {consumed}: {exc}') from exc
                consumed+=1
            rows.append(dict(split=chart['split'],window=entry['window'],tokens=consumed))
    report=dict(passed=True,windows=len(rows),tokens=sum(r['tokens'] for r in rows),
        notice='Authored event grammar compatibility only, not constrained free-generation quality.')
    (args.run/'event-grammar-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
