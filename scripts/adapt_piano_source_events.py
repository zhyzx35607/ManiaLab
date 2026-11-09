"""Fit masked local notation successors on verified sustain-event graphs.

Explicit research entry: no test split, acoustic model, or application weights.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from scripts.adapt_piano_streams import forward, load_model, verify_source
from scripts.build_provenance import sha256
from scripts.piano_event_training import load_event_items, make_event_graph
from scripts.research_piano_streams import MODEL_SHA


def loss_terms(outputs, targets):
    voice, staff, _, chord = outputs
    if targets['chord_mask'].any():
        raise ValueError('Chord supervision is not defined by this protocol')
    terms = []
    vm, sm = targets['voice_mask'], targets['staff_mask']
    if vm.any():
        terms.append(F.binary_cross_entropy_with_logits(voice[vm], targets['voice'][vm],
            pos_weight=voice.new_tensor(5.)))
    if sm.any():
        terms.append(F.cross_entropy(staff[sm], targets['staff'][sm]))
    return sum(terms) if terms else None


def edge_counts(logits, targets, mask):
    actual = targets[mask].bool()
    predicted = logits[mask] >= 0  # fixed sigmoid threshold 0.5
    return Counter(tp=int((actual & predicted).sum()), fp=int((~actual & predicted).sum()),
                   fn=int((actual & ~predicted).sum()), tn=int((~actual & ~predicted).sum()))


def scores(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    return dict(**counts, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn))


def evaluate(model, items):
    model.eval()
    pieces = defaultdict(lambda: dict(voice=Counter(), staff_correct=0, staff_notes=0,
                                     candidate_edges=0, supervised_edges=0))
    with torch.inference_mode():
        for item in items:
            graph, target = make_event_graph(item)
            voice, staff, _, _ = forward(model, graph)
            row = pieces[item['name']]
            row['voice'].update(edge_counts(voice, target['voice'], target['voice_mask']))
            sm = target['staff_mask']
            row['staff_correct'] += int((staff[sm].argmax(dim=1)==target['staff'][sm]).sum())
            row['staff_notes'] += int(sm.sum())
            row['candidate_edges'] += len(voice)
            row['supervised_edges'] += int(target['voice_mask'].sum())
    pooled = Counter()
    rows = []
    for name, row in sorted(pieces.items()):
        pooled.update(row['voice'])
        rows.append(dict(name=name, **dict(row, voice=scores(row['voice'])),
            staff_accuracy=row['staff_correct']/row['staff_notes'] if row['staff_notes'] else None))
    eligible = [r['voice']['f1'] for r in rows if r['voice']['tp']+r['voice']['fn'] > 0]
    if not eligible:
        raise ValueError('No positive validation successor labels')
    return dict(voice=scores(pooled), macro_voice_f1=float(np.mean(eligible)), cases=rows,
        staff_correct=sum(r['staff_correct'] for r in rows), staff_notes=sum(r['staff_notes'] for r in rows),
        candidate_edges=sum(r['candidate_edges'] for r in rows),
        supervised_edges=sum(r['supervised_edges'] for r in rows),
        scope='masked local notation successors at fixed 0.5; not decoded phrase or LN accuracy')


def check_graphs(items, model):
    counts = Counter()
    step = None
    for item in items:
        graph, target = make_event_graph(item)
        assert 'staff' not in graph['note'] and 'voice' not in graph['note']
        assert not any('truth' in kind[1] for kind in graph.edge_types)
        assert all(torch.isfinite(value).all() for value in graph['note'].values()
                   if isinstance(value, torch.Tensor))
        for kind in ('voice', 'chord', 'staff'):
            mask = target[kind+'_mask']
            assert mask.dtype == torch.bool and mask.shape == target[kind].shape
            counts[kind+'_labels'] += int(mask.sum())
        assert not target['chord_mask'].any()
        counts['windows'] += 1
        counts['nodes_with_context'] += len(graph['note'].x)
        if step is None and target['voice_mask'].any():
            step = graph, target
    if step is None:
        raise ValueError('No supervised train graph')
    # Disposable update catches mask/shape/autograd errors without altering the
    # initial model used in baseline or fitting.
    disposable = copy.deepcopy(model).train()
    loss = loss_terms(forward(disposable, step[0]), step[1])
    if loss is None or not torch.isfinite(loss):
        raise ValueError('Nonfinite control loss')
    loss.backward()
    gradients = [p.grad for p in disposable.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Nonfinite control gradient')
    optimizer = torch.optim.AdamW(disposable.parameters(), lr=1e-4, weight_decay=5e-4)
    torch.nn.utils.clip_grad_norm_(disposable.parameters(), 1.)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in disposable.parameters()):
        raise ValueError('Nonfinite control weights')
    return dict(counts, finite_discarded_backward=True, control_loss=float(loss.detach()))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--dataset', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--epochs', type=int, default=4)
    cli.add_argument('--check-only', action='store_true')
    args = cli.parse_args()
    if args.output.exists() or not 1 <= args.epochs <= 8:
        raise ValueError('Use a new output directory and 1..8 fixed epochs')
    revision = verify_source(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    train = load_event_items(args.dataset, 'train')
    validation = load_event_items(args.dataset, 'validation')
    if not train or not validation:
        raise ValueError('Empty fit/validation partition')
    model = load_model(args.source/'pretrained_models/model.ckpt')
    args.output.mkdir(parents=True)
    save = lambda name,obj: (args.output/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    sources = {'runner': Path(__file__), 'event_graph': ROOT/'scripts/piano_event_training.py',
        'base_runner': ROOT/'scripts/adapt_piano_streams.py',
        'protocol': ROOT/'docs/piano-source-event-training-protocol.md'}
    identity = dict(source_revision=revision, pretrained_sha256=MODEL_SHA,
        dataset_sha256=sha256(args.dataset/'manifest.json'),
        code_sha256={key:sha256(path) for key,path in sources.items()}, seed=42,
        epochs=args.epochs, train_windows=len(train), validation_windows=len(validation),
        lr=1e-4, weight_decay=5e-4, voice_positive_weight=5., threshold=.5,
        input_kind='clean verified sustain events', chord_supervision=False,
        test_loaded=False, application_enabled=False)
    for key,path in sources.items():
        shutil.copyfile(path, args.output/(key+'-snapshot'+path.suffix))
    save('identity.json', identity)
    checks = check_graphs(train+validation, model)
    save('checks.json', checks)
    print(json.dumps(dict(stage='checks', **checks)), flush=True)
    if args.check_only:
        return
    started = time.monotonic()
    baseline = evaluate(model, validation)
    save('validation-epoch-0.json', baseline)
    torch.save(dict(state_dict=model.state_dict(), epoch=0), args.output/'epoch-0.pt')
    best_score, best_epoch = baseline['macro_voice_f1'], 0
    print(json.dumps(dict(epoch=0, voice=baseline['voice'], macro_voice_f1=best_score)), flush=True)
    torch.manual_seed(42)
    rng = random.Random(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        skipped = 0
        for position,index in enumerate(order):
            graph,target = make_event_graph(train[index])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_terms(forward(model, graph), target)
            if loss is None:
                skipped += 1
                continue
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fit loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (position+1)%50 == 0:
                print(json.dumps(dict(epoch=epoch, window=position+1, total=len(order),
                    mean_loss=float(np.mean(losses)), elapsed=time.monotonic()-started)), flush=True)
        if not losses:
            raise ValueError('No supervised training windows')
        value = evaluate(model, validation)
        save(f'validation-epoch-{epoch}.json', value)
        torch.save(dict(state_dict=model.state_dict(), epoch=epoch), args.output/f'epoch-{epoch}.pt')
        if value['macro_voice_f1'] > best_score:
            best_score,best_epoch = value['macro_voice_f1'],epoch
        row = dict(epoch=epoch, mean_train_loss=float(np.mean(losses)), skipped=skipped,
            voice=value['voice'], macro_voice_f1=value['macro_voice_f1'],
            staff_correct=value['staff_correct'],staff_notes=value['staff_notes'],best_epoch=best_epoch)
        history.append(row)
        save('progress.json', history)
        print(json.dumps(row), flush=True)
    if any(sha256(path) != identity['code_sha256'][key] for key,path in sources.items()):
        raise ValueError('Training source changed during fit')
    if sha256(args.dataset/'manifest.json') != identity['dataset_sha256']:
        raise ValueError('Dataset manifest changed during fit')
    shutil.copyfile(args.output/f'epoch-{best_epoch}.pt', args.output/'selected.pt')
    report = dict(identity=identity,checks=checks,baseline=baseline,history=history,
        selected_epoch=best_epoch,selected_sha256=sha256(args.output/'selected.pt'),
        elapsed_seconds=time.monotonic()-started, musical_acceptance=False)
    save('report.json', report)
    print(json.dumps(dict(selected_epoch=best_epoch, validation_macro_voice_f1=best_score,
                         elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
