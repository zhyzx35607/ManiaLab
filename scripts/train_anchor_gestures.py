"""Train a research endpoint distribution from frozen onset-only evidence.

Prepare uses the application environment; fit uses the optional torch worker.
No prepared dataset or weights are distributed with the application.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.anchor_gestures import AnchorContext, FEATURES, REVISION, labels
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import sha
from scripts.train_layout_network import records_and_groups
from scripts.train_pattern_prior import split_for


def source_chart(path):
    chart = parse_osu(path.read_text(encoding='utf-8-sig'))
    def voice(n):
        return ('kick' if n.lane == 0 else 'snare' if n.lane < 3 else 'cymbal'
                if n.lane == 3 else 'piano' if n.lane < 6 else 'other')
    chart['notes'] = [replace(n, end=n.start, voice=voice(n), id=f'anchor:{i}') for i, n in enumerate(chart['notes'])]
    return chart


def target_groups(context, authored):
    """Assign each authored head to at most one supplied onset, nearest first."""
    grouped = defaultdict(list)
    for n in authored:
        j = int(np.searchsorted(context.times, n.start))
        near = [i for i in (j-1, j) if 0 <= i < len(context.times)]
        if near:
            i = min(near, key=lambda i: (abs(int(context.times[i])-n.start), i))
            if abs(int(context.times[i])-n.start) <= 30:
                grouped[i].append(n)
    return grouped


def add_examples(context, authored, indices, store, weight):
    grouped = target_groups(context, authored)
    counts = Counter()
    for i in indices:
        y, diagnostics = labels(context, i, grouped[i])
        counts.update(diagnostics)
        x = context.features(i)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite supervised example; refuse dataset preparation')
        store['x'].append(x); store['y'].append(y)
        store['lengths'].append(len(y)); store['weights'].append(weight)
    return dict(counts)


def prepare(args):
    if args.output.exists():
        raise ValueError('Preserve existing datasets; choose a new output')
    data = defaultdict(lambda: dict(x=[], y=[], lengths=[], weights=[]))
    from manialab.release_candidates import ReleaseContext, AcousticReleaseContext
    plain_context = ReleaseContext if args.independent_releases else AnchorContext
    records = []; exclusions = []; feature_names = plain_context.feature_names; acoustic_identity = None
    if args.source == 'sakura':
        root = Path('test-data/sakura-benchmark')
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        for role in ('reference', 'finished'):
            if sha(root/(role+'.osu')) != manifest['charts'][role]['chart_sha256']:
                raise ValueError('Frozen author chart changed')
        sketch = root/'detected-onset-sketch-v1/sketch.osu'
        evidence = source_chart(sketch)
        # These targets are limited to the declared fit/calibration portions.
        human = parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
        ratio = sum(n.hold for n in human)/len(human)
        if args.acoustic:
            from manialab.acoustic_anchors import AcousticAnchorContext
            from scripts.anchor_acoustic_inputs import load_evidence
            acoustic, acoustic_identity = load_evidence(root)
            acoustic_context = AcousticReleaseContext if args.independent_releases else AcousticAnchorContext
            context = acoustic_context(evidence['notes'], 10, evidence['timing'], ratio, acoustic)
            feature_names = context.feature_names
        else:
            context = plain_context(evidence['notes'], 10, evidence['timing'], ratio)
        for split, start, end in [('train', 0, 149367), ('validation', 162340, 188286)]:
            indices = [i for i in range(len(context.times)) if context.contained(i, start, end)]
            truth = [n for n in human if start <= n.start and n.end < end]
            counts = add_examples(context, truth, indices, data[split], 1/max(1, len(indices)))
            records.append(dict(split=split, keys=10, source_sha256=sha(sketch),
                target_sha256=sha(root/'finished.osu'), rows=len(indices), bounds=[start, end], labels=counts))
        scope = 'Same-song later sections; detected sources, author Timing and whole-chart LN condition.'
    else:
        if args.acoustic: raise ValueError('Community chart-only snapshot has no paired acoustic input')
        selected, excluded = records_and_groups(args.corpus)
        selected = [r for r in selected if r.get('reference_role') == 'expression_and_layout']
        group_charts = Counter((r['keys'], r['group']) for r in selected)
        key_groups = Counter((key, split_for(group)) for key, group in group_charts)
        for at, record in enumerate(selected):
            path = args.corpus/'maps'/f"{record['beatmap_id']}.osu"
            chart = parse_osu(path.read_text(encoding='utf-8-sig'))
            notes = [replace(n, end=n.start, lane=0, voice='mix') for n in chart['notes']]
            ratio = sum(n.hold for n in chart['notes'])/len(notes)
            try:
                context = plain_context(notes, chart['keys'], chart['timing'], ratio)
            except ValueError as error:
                # A layout-valid chart can lack usable red lines. Never invent
                # a 120 BPM clock to make expression training accept it.
                exclusions.append(dict(id=record['beatmap_id'], reason=str(error)))
                continue
            budget = max(1, min(32, 48//group_charts[(record['keys'], record['group'])]))
            indices = sorted(np.random.default_rng(int(record['beatmap_id'])).choice(
                len(context.times), min(budget, len(context.times)), replace=False).tolist())
            split = split_for(record['group'])
            counts = add_examples(context, chart['notes'], indices, data[split],
                1/(group_charts[(record['keys'], record['group'])]*len(indices)*key_groups[(record['keys'], split)]))
            records.append(dict(id=record['beatmap_id'], keys=chart['keys'], split=split,
                group=record['group'], sha256=sha(path), rows=len(indices), labels=counts))
            if at % 250 == 0:
                print(json.dumps(dict(prepared=at, charts=len(selected))), flush=True)
        scope = 'Known authored onsets/counts/Timing/LN condition; song-group split; not audio transcription.'
    args.output.mkdir(parents=True)
    files = {}
    for split, group in data.items():
        path = args.output/(split+'.npz')
        np.savez_compressed(path, x=np.concatenate(group['x']), y=np.concatenate(group['y']),
            lengths=np.asarray(group['lengths']), weights=np.asarray(group['weights']))
        files[split] = dict(sha256=sha(path), rows=len(group['lengths']), options=sum(group['lengths']))
    manifest = dict(revision=plain_context.revision, features=feature_names, source=args.source, scope=scope,
        records=records, exclusions=exclusions, files=files, script_sha256=sha(__file__),
        acoustic_identity=acoustic_identity, acoustic_feature_sha256=sha('manialab/acoustic_anchors.py') if args.acoustic else None,
        release_feature_sha256=sha('manialab/release_candidates.py') if args.independent_releases else None,
        feature_sha256=sha('manialab/anchor_gestures.py'),
        protocol_sha256=sha('docs/release-candidate-protocol.md' if args.independent_releases else 'docs/anchor-gesture-protocol.md'))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(files), flush=True)


def fit(args):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(20261109)
    rng = np.random.default_rng(20261109)
    if args.output.exists():
        raise ValueError('Preserve completed experiments')
    manifest = json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    if manifest['feature_sha256'] != sha('manialab/anchor_gestures.py'):
        raise ValueError('Rebuild data after feature changes')
    feature_names = manifest['features']
    if manifest.get('acoustic_feature_sha256') and manifest['acoustic_feature_sha256'] != sha('manialab/acoustic_anchors.py'):
        raise ValueError('Acoustic feature implementation changed')
    if manifest.get('release_feature_sha256') and manifest['release_feature_sha256'] != sha('manialab/release_candidates.py'):
        raise ValueError('Release feature implementation changed')
    if manifest['source'] == 'sakura' and args.initial is None:
        print('Training explicit same-song-only diagnostic without a community initialization', flush=True)
    data = {}
    # Test arrays are not loaded during fitting or epoch selection.
    for split in ('train', 'validation'):
        path = args.dataset/(split+'.npz')
        if sha(path) != manifest['files'][split]['sha256']:
            raise ValueError('Training dataset changed')
        with np.load(path, allow_pickle=False) as saved:
            data[split] = {k: saved[k] for k in saved.files}
    model = nn.Sequential(nn.Linear(len(feature_names), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    if args.initial:
        initial = json.loads(args.initial.read_text(encoding='utf-8'))
        if initial['revision'] != manifest['revision'] or initial['features'] != feature_names[:len(initial['features'])]:
            raise ValueError('Incompatible initialization')
        previous_width = len(initial['features'])
        mean = data['train']['x'].mean(axis=0); scale = np.maximum(.1, data['train']['x'].std(axis=0))
        mean[:previous_width] = initial['mean']; scale[:previous_width] = initial['scale']
        for i, (layer, w, b) in enumerate(zip((model[0], model[2], model[4]), initial['weights'], initial['biases'])):
            if i == 0:
                layer.weight.data.zero_(); layer.weight.data[:, :previous_width].copy_(torch.tensor(w))
            else: layer.weight.data.copy_(torch.tensor(w))
            layer.bias.data.copy_(torch.tensor(b))
    else:
        mean = data['train']['x'].mean(axis=0)
        scale = np.maximum(.1, data['train']['x'].std(axis=0))
    for group in data.values():
        group['offsets'] = np.concatenate(([0], np.cumsum(group['lengths'])))
        group['x'] = torch.from_numpy(np.clip((group['x']-mean)/scale, -10, 10).astype(np.float32))
        group['y'] = torch.from_numpy(group['y'].astype(np.float32))
        group['weights'] *= len(group['weights'])/group['weights'].sum()
    def batch(group, indices):
        length = int(max(group['lengths'][indices]))
        x = torch.zeros((len(indices), length, len(feature_names)))
        y = torch.zeros((len(indices), length)); mask = torch.zeros((len(indices), length), dtype=torch.bool)
        for j, i in enumerate(indices):
            lo, hi = group['offsets'][i:i+2]; n = hi-lo
            x[j, :n] = group['x'][lo:hi]; y[j, :n] = group['y'][lo:hi]; mask[j, :n] = True
        return x, y, mask, torch.tensor(group['weights'][indices], dtype=torch.float32)
    def objective(group, indices):
        x, y, mask, weights = batch(group, indices)
        score = model(x).squeeze(-1).masked_fill(~mask, -1e9)
        losses = -(torch.log_softmax(score, dim=1)*y).sum(dim=1)
        return (losses*weights).mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best, best_state, history = float('inf'), None, []
    args.output.mkdir(parents=True)
    epochs = args.epochs or (80 if manifest['source'] == 'sakura' else 24)
    for epoch in range(epochs):
        model.train(); train_losses = []
        order = rng.permutation(len(data['train']['lengths']))
        for offset in range(0, len(order), 256):
            optimizer.zero_grad(); loss = objective(data['train'], order[offset:offset+256])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval(); validation = []
        with torch.inference_mode():
            for offset in range(0, len(data['validation']['lengths']), 256):
                ids = np.arange(offset, min(offset+256, len(data['validation']['lengths'])))
                validation.append((float(objective(data['validation'], ids)), len(ids)))
        val = sum(v*n for v, n in validation)/sum(n for _, n in validation)
        row = dict(epoch=epoch+1, train_loss=float(np.mean(train_losses)), validation_loss=val)
        history.append(row)
        if val < best:
            best = val; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_epoch = epoch+1
        if epoch % 5 == 0 or epoch+1 == epochs:
            print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    value = dict(revision=manifest['revision'], features=feature_names, mean=mean.tolist(), scale=scale.tolist(),
        weights=[model[i].weight.detach().numpy().tolist() for i in (0, 2, 4)],
        biases=[model[i].bias.detach().numpy().tolist() for i in (0, 2, 4)],
        dataset_sha256=sha(args.dataset/'manifest.json'), feature_sha256=manifest['feature_sha256'],
        acoustic_feature_sha256=manifest.get('acoustic_feature_sha256'),
        release_feature_sha256=manifest.get('release_feature_sha256'),
        selected_epoch=best_epoch, enabled_in_application=False)
    (args.output/'model.json').write_text(json.dumps(value), encoding='utf-8')
    # Compare the NumPy scorer's primitive layers with torch before export.
    x = data['validation']['x'][:1024].numpy(); expected = model(torch.from_numpy(x)).detach().numpy()
    actual = x.copy()
    for i, (w, b) in enumerate(zip(value['weights'], value['biases'])):
        actual = actual@np.asarray(w, np.float32).T+np.asarray(b, np.float32)
        if i < 2: actual = np.maximum(actual, 0.)
    error = float(np.max(np.abs(expected-actual)))
    if error > 1e-4: raise ValueError('Export failed scorer agreement')
    report = dict(source=manifest['source'], scope=manifest['scope'], epochs=epochs, selected_epoch=best_epoch,
        model_sha256=sha(args.output/'model.json'), dataset_sha256=sha(args.dataset/'manifest.json'),
        initialization_sha256=sha(args.initial) if args.initial else None,
        script_sha256=sha(__file__), export_max_error=error, history=history,
        files=manifest['files'], enabled_in_application=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare'); prep.add_argument('--source', choices=('sakura', 'corpus'), required=True)
    prep.add_argument('--corpus', type=Path, default=Path('test-data/corpus-expanded/snapshot-verified'))
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--acoustic', action='store_true', help='Add audited model observations and Nova energy, without author tails')
    prep.add_argument('--independent-releases', action='store_true', help='Allow beat-grid releases without new input attacks')
    training = commands.add_parser('fit'); training.add_argument('--dataset', type=Path, required=True)
    training.add_argument('--output', type=Path, required=True); training.add_argument('--initial', type=Path)
    training.add_argument('--epochs', type=int)
    args = p.parse_args()
    (prepare if args.command == 'prepare' else fit)(args)


if __name__ == '__main__': main()
