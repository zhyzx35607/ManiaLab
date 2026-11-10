"""Audit LN context and measure real audio dependence on fixed validation windows.

Teacher-forced diagnostics only. Both models see identical text/history. A
different validation song's audio embedding is the counterfactual; no weights
are updated. Low loss alone cannot establish correct musical interpretation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

from mapper_training_context import typed_time_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    opts = parser.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(opts.run/'resolved-config.yaml')
    source = Path(cfg.pretrained_path).parents[1]/'source'
    sys.path[:0] = [str(source/'osuT5'), str(source)]
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers.modeling_outputs import BaseModelOutput
    from osuT5.utils.model_utils import load_model
    from osuT5.event import EventType
    torch.set_num_threads(opts.threads)
    args = OmegaConf.to_object(cfg)
    manifest = json.loads((opts.run/'samples.json').read_text())
    identity = manifest['identity']
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    samples = []
    audit = []
    model, tokenizer = load_model(cfg.pretrained_path, args, device='cpu', precision='fp32', attn_implementation='sdpa',
                                  eval_mode=True, gamemode=3)
    starts = {t.value:n for t,n in tokenizer.event_start.items()}
    ends = {t.value:n for t,n in tokenizer.event_end.items()}
    for chart in manifest['charts']:
        cached = torch.load(opts.run.parent/'encoded'/f"{fingerprint}-{chart['id']}.pt", weights_only=True)
        entries = cached['samples'] if isinstance(cached, dict) else cached
        for sample in entries:
            batch = sample['batch'];tokens = batch['decoder_input_ids'][0].tolist();labels = batch['labels'][0].tolist()
            active = set();column = None;orphan_tails = 0;label_tails = 0
            for i, token in enumerate(tokens):
                if starts['column'] <= token < ends['column']:
                    column = token-starts['column']
                elif token in (starts['hold_note'], starts['hold_note_sustain']):
                    active.add(column)
                elif token == starts['hold_note_end']:
                    if i > 0 and labels[i-1] == token:
                        label_tails += 1
                        orphan_tails += column not in active
                    active.discard(column)
            audit.append(dict(id=chart['id'], ordinal=sample['ordinal'], split=chart['split'],
                              label_tails=label_tails, unconditioned_label_tails=orphan_tails))
            if chart['split'] == 'validation':
                samples.append(dict(sample, chart=chart))
    peers = [next(j for j in range((i+2)%len(samples), (i+2)%len(samples)+len(samples))
                  if samples[j%len(samples)]['chart']['id'] != sample['chart']['id'])%len(samples)
             for i, sample in enumerate(samples)]
    report = dict(identity=identity, protocol='Fixed validation text; correct vs another song audio. Teacher-forced diagnostics, not free generation.',
                  audit=audit, summary=dict(windows=len(audit), unconditioned_label_tails=sum(r['unconditioned_label_tails'] for r in audit),
                                           label_tails=sum(r['label_tails'] for r in audit)), results={})
    path = opts.run/'audio-evaluation.json'
    for stage in ('baseline', 'adapter'):
        if stage == 'adapter':
            model = PeftModel.from_pretrained(model, opts.run/'adapter', is_trainable=False)
        model.eval()
        report['results'][stage] = {}
        for condition in ('correct_audio', 'other_song_audio'):
            rows = []
            with torch.no_grad():
                for i, sample in enumerate(samples):
                    hidden = sample['hidden'] if condition == 'correct_audio' else samples[peers[i]]['hidden']
                    out = model(**sample['batch'], encoder_outputs=BaseModelOutput(last_hidden_state=hidden), use_cache=False)
                    y = sample['batch']['labels'];pred = out.logits.argmax(-1)
                    loss = torch.nn.functional.cross_entropy(out.logits.transpose(1,2), y, ignore_index=-100, reduction='none')
                    groups = typed_time_positions(y[0].tolist(), starts, ends)
                    groups['all_time'] = torch.nonzero((y[0]>=starts['t'])&(y[0]<ends['t']), as_tuple=False).flatten().tolist()
                    stats = {}
                    for name, positions in groups.items():
                        if not positions:continue
                        expected = y[0,positions];actual = pred[0,positions]
                        is_time = (actual>=starts['t'])&(actual<ends['t'])
                        error = (expected-actual).abs()*10
                        stats[name] = dict(count=len(positions), nll=float(loss[0,positions].mean()),
                                           within_30ms=float((is_time&(error<=30)).float().mean()),
                                           exact_accuracy=float((actual==expected).float().mean()))
                    rows.append(dict(id=sample['chart']['id'], keys=sample['chart']['keys'], ordinal=sample['ordinal'], stats=stats))
            macro = {}
            for name in ('ln_head_time','ln_tail_time','tap_time','all_time'):
                songs = defaultdict(list)
                for row in rows:
                    if name in row['stats']:songs[row['id']].append(row['stats'][name])
                macro[name] = {metric:float(np.mean([np.mean([s[metric] for s in values]) for values in songs.values()])) if songs else None
                               for metric in ('nll','within_30ms','exact_accuracy')}
            report['results'][stage][condition] = dict(macro=macro, windows=rows)
            path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(dict(stage=stage, condition=condition, macro=macro)), flush=True)


if __name__ == '__main__':
    main()
