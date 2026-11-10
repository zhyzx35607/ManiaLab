"""Fit local successors with source labels and independently extended input tails."""
import argparse
from collections import Counter
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
from scripts.adapt_piano_source_events import check_graphs, evaluate, loss_terms
from scripts.adapt_piano_streams import forward, load_model, verify_source
from scripts.build_provenance import sha256
from scripts.piano_event_training import load_event_items, make_event_graph
from scripts.research_piano_streams import MODEL_SHA

KINDS = ('clean', 'plus1', 'plus2')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired_variants(folders, split):
    groups = {kind: load_event_items(folder, split) for kind, folder in folders.items()}
    identity = lambda item: (item['name'], item['chunk'])
    expected = [identity(item) for item in groups['clean']]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Empty or duplicate clean windows')
    for kind in KINDS[1:]:
        indexed = {identity(item): item for item in groups[kind]}
        if len(indexed) != len(groups[kind]) or set(indexed) != set(expected):
            raise ValueError('Augmentation changed supervised window identities')
        groups[kind] = [indexed[key] for key in expected]
    return groups


def evaluate_variants(model, groups):
    results = {kind: evaluate(model, groups[kind]) for kind in KINDS}
    return dict(variants=results,
                selection_score=float(np.mean([r['macro_voice_f1'] for r in results.values()])))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--clean-dataset', type=Path, required=True)
    cli.add_argument('--overlap-dataset', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--check-only', action='store_true')
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous runs; use a new output directory')
    revision = verify_source(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    folders = dict(clean=args.clean_dataset, plus1=args.overlap_dataset/'plus1',
                   plus2=args.overlap_dataset/'plus2')
    augmentation = read(args.overlap_dataset/'manifest.json')
    if augmentation['identity']['clean_manifest_sha256'] != sha256(args.clean_dataset/'manifest.json'):
        raise ValueError('Augmentation belongs to a different clean dataset')
    for kind in KINDS[1:]:
        if augmentation['variants'][kind]['sha256'] != sha256(folders[kind]/'manifest.json'):
            raise ValueError('Augmentation manifest changed')
    train = paired_variants(folders, 'train')
    validation = paired_variants(folders, 'validation')
    model = load_model(args.source/'pretrained_models/model.ckpt')
    args.output.mkdir(parents=True)
    save = lambda name, obj: (args.output/name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    sources = dict(runner=Path(__file__),
        clean_runner=ROOT/'scripts/adapt_piano_source_events.py',
        base_runner=ROOT/'scripts/adapt_piano_streams.py',
        event_graph=ROOT/'scripts/piano_event_training.py',
        overlap=ROOT/'scripts/piano_event_overlap.py',
        prepare=ROOT/'scripts/prepare_piano_event_overlap.py',
        protocol=ROOT/'docs/piano-source-event-overlap-protocol.md')
    manifests = {kind: folder/'manifest.json' for kind, folder in folders.items()}
    manifests['augmentation'] = args.overlap_dataset/'manifest.json'
    identity = dict(source_revision=revision, pretrained_sha256=MODEL_SHA,
        dataset_sha256={key: sha256(path) for key, path in manifests.items()},
        code_sha256={key: sha256(path) for key, path in sources.items()},
        seed=42, epochs=4, lr=1e-4, weight_decay=5e-4, voice_positive_weight=5., threshold=.5,
        train_windows_per_epoch=len(train['clean']), validation_windows_per_kind=len(validation['clean']),
        input_schedule='kind[(original_window_index + epoch - 1) % 3]; seeded shuffled traversal',
        selection='equal mean of clean/plus1/plus2 per-piece macro voice F1, earliest maximum',
        chord_supervision=False, test_loaded=False, application_enabled=False)
    for key, path in sources.items():
        shutil.copyfile(path, args.output/(key+'-snapshot'+path.suffix))
    save('identity.json', identity)
    checks = check_graphs([item for group in (train, validation)
                          for kind in KINDS for item in group[kind]], model)
    save('checks.json', checks)
    print(json.dumps(dict(stage='checks', **checks)), flush=True)
    if args.check_only:
        return
    started = time.monotonic()
    baseline = evaluate_variants(model, validation)
    save('validation-epoch-0.json', baseline)
    torch.save(dict(state_dict=model.state_dict(), epoch=0), args.output/'epoch-0.pt')
    best_score, best_epoch = baseline['selection_score'], 0
    print(json.dumps(dict(epoch=0, selection_score=best_score,
        macro_by_kind={k: r['macro_voice_f1'] for k, r in baseline['variants'].items()})), flush=True)
    torch.manual_seed(42)
    rng = random.Random(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    history = []
    for epoch in range(1, 5):
        model.train()
        order = list(range(len(train['clean'])))
        rng.shuffle(order)
        losses, skipped, used = [], Counter(), Counter()
        for position, index in enumerate(order):
            kind = KINDS[(index+epoch-1) % len(KINDS)]
            graph, target = make_event_graph(train[kind][index])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_terms(forward(model, graph), target)
            if loss is None:
                skipped[kind] += 1
                continue
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fit loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            used[kind] += 1
            if (position+1) % 50 == 0:
                print(json.dumps(dict(epoch=epoch, window=position+1, total=len(order),
                    mean_loss=float(np.mean(losses)), elapsed=time.monotonic()-started)), flush=True)
        if not losses:
            raise ValueError('No supervised windows')
        result = evaluate_variants(model, validation)
        save(f'validation-epoch-{epoch}.json', result)
        torch.save(dict(state_dict=model.state_dict(), epoch=epoch), args.output/f'epoch-{epoch}.pt')
        if result['selection_score'] > best_score:
            best_score, best_epoch = result['selection_score'], epoch
        row = dict(epoch=epoch, mean_train_loss=float(np.mean(losses)), used=used, skipped=skipped,
            selection_score=result['selection_score'], best_epoch=best_epoch,
            variants={k: {field: r[field] for field in ('voice', 'macro_voice_f1', 'staff_correct', 'staff_notes')}
                      for k, r in result['variants'].items()})
        history.append(row)
        save('progress.json', history)
        print(json.dumps(row), flush=True)
    if any(sha256(path) != identity['code_sha256'][key] for key, path in sources.items()):
        raise ValueError('Training source changed during fit')
    if any(sha256(path) != identity['dataset_sha256'][key] for key, path in manifests.items()):
        raise ValueError('Dataset manifest changed during fit')
    shutil.copyfile(args.output/f'epoch-{best_epoch}.pt', args.output/'selected.pt')
    report = dict(identity=identity, checks=checks, baseline=baseline, history=history,
        selected_epoch=best_epoch, selected_sha256=sha256(args.output/'selected.pt'),
        elapsed_seconds=time.monotonic()-started, musical_acceptance=False)
    save('report.json', report)
    print(json.dumps(dict(selected_epoch=best_epoch, selection_score=best_score,
                         elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
