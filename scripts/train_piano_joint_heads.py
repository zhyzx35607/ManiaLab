"""Preflight, then fit two acoustic heads under a fixed development protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.piano_head_runtime import load_frozen_model, decode
from scripts.piano_adaptation_data import stitch
from scripts.piano_joint_data import frame_targets, owned_mask
from scripts.piano_joint_runtime import heads_copy, forward_logits, replay_outputs
from scripts.piano_joint_selection import compare_recording, eligible, select_epoch
from scripts.evaluate_piano_acoustics import metrics

CACHE_SHA = '29a84fe641efbda504cd815dfea777de81ae8452fb6fc3fad1ebb87e36ad0a8f'
ROSTER = {'train': ['449', '1046', '518', '1176', '926', '1222'],
          'validation': ['1219', '628', '1134', '752']}
CELLS = 9498720


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def paths():
    names = ['train_piano_joint_heads', 'piano_joint_runtime', 'piano_joint_data',
             'piano_joint_selection', 'piano_head_runtime', 'piano_adaptation_data',
             'evaluate_piano_acoustics', 'train_piano_onset_head', 'transkun_runtime',
             'probe_piano_acoustics']
    return {**{name: ROOT/'scripts'/(name+'.py') for name in names},
            'protocol': ROOT/'docs/piano-joint-head-adaptation-protocol.md'}


def verify_cache(cache):
    if sha256(cache/'report.json') != CACHE_SHA:
        raise ValueError('Use the fixed, complete development cache')
    report = read(cache/'report.json')
    if len(report['windows']) != 350 or not report['all_epoch0_outputs_identical']:
        raise ValueError('Incomplete or unverified cache')
    for split, expected in ROSTER.items():
        if [r['id'] for r in report['records'] if r['split'] == split] != expected:
            raise ValueError('Development recording roster differs')
        for name in expected:
            windows = [r for r in report['windows'] if r['id'] == name]
            if [r['index'] for r in windows] != list(range(35)) or any(r['split'] != split for r in windows):
                raise ValueError('Window roster differs')
    for name, digest in report['artifacts_sha256'].items():
        path = (cache/name).resolve()
        if path.parent != cache or sha256(path) != digest:
            raise ValueError('Cache artifact changed: '+name)
    return report


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        array = tensor.detach().cpu().numpy()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    if model.training or any(m.training for m in model.modules()):
        raise ValueError('Frozen model left eval mode')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen model acquired gradients')
    return digest.hexdigest()


def target_file(item):
    return item['id']+f"-w{item['index']:02d}-targets.npz"


def batch_tensors(cache, output, items):
    import torch
    arrays = {key: [] for key in ('features', 'frame_base', 'reg_offset_output',
                                  'onset', 'frame', 'mask')}
    for item in items:
        with np.load(cache/item['file'], allow_pickle=False) as data:
            for key in ('features', 'frame_base', 'reg_offset_output'):
                arrays[key].append(data[key][0].copy())
            arrays['onset'].append(data['targets'].copy())
        with np.load(output/target_file(item), allow_pickle=False) as target:
            arrays['frame'].append(target['frame'].copy())
            arrays['mask'].append(target['mask'].copy())
    return {key: torch.from_numpy(np.stack(value)) for key, value in arrays.items()}


def losses(model, heads, batch):
    import torch
    onset, frame = forward_logits(model, heads, batch['features'], batch['frame_base'],
                                  batch['reg_offset_output'])
    denominator = int(batch['mask'].sum())*88
    if denominator <= 0:
        raise ValueError('Empty supervised batch')
    mask = batch['mask'][..., None]
    values = {key: (torch.nn.functional.binary_cross_entropy_with_logits(logits, batch[key],
               reduction='none') * mask).sum()/denominator
              for key, logits in [('onset', onset), ('frame', frame)]}
    if any(not torch.isfinite(value) for value in values.values()):
        raise ValueError('Nonfinite training loss')
    return values, denominator


def gradient_preflight(model, batch):
    import torch
    torch.manual_seed(42)
    heads = heads_copy(model)
    if sum(p.numel() for p in heads.parameters()) != 90288:
        raise ValueError('Expected exactly 90288 trainable parameters')
    original = {key: tensor.clone() for key, tensor in heads.state_dict().items()}
    proof = {}
    for target in ('onset', 'frame'):
        heads.zero_grad(set_to_none=True)
        values, _ = losses(model, heads, batch)
        values[target].backward()
        branches = {}
        for name, branch in heads.items():
            grads = [p.grad for p in branch.parameters()]
            if name == target:
                if any(g is None or not torch.isfinite(g).all() or not torch.any(g != 0) for g in grads):
                    raise ValueError('Missing, zero or invalid gradient in '+name)
                branches[name] = dict(nonzero=True, norm=float(torch.cat([g.flatten() for g in grads]).norm()))
            else:
                if any(g is not None for g in grads):
                    raise ValueError('Gradient crossed the fixed detach boundary')
                branches[name] = dict(gradient_absent=True)
        proof[target] = branches
    heads.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-5, weight_decay=1e-4)
    values, _ = losses(model, heads, batch)
    sum(values.values()).backward()
    norm = torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.)
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite joint gradient')
    optimizer.step()
    changed = [key for key, tensor in heads.state_dict().items() if not torch.equal(tensor, original[key])]
    if set(changed) != set(original):
        raise ValueError('Joint step did not update all four head tensors')
    proof.update(joint_step_changed=changed, parameters=90288, joint_gradient_norm=float(norm))
    return proof


def prepare(cache, output, report):
    import torch
    if output.exists():
        raise ValueError('Preserve earlier evidence; use a new output directory')
    model, model_identity = load_frozen_model()
    if model_identity != report['identity']['model']:
        raise ValueError('Model differs from original cache')
    frozen_before = frozen_digest(model)
    torch.manual_seed(42)
    heads = heads_copy(model)
    output.mkdir(parents=True)
    identity = dict(cache_report_sha256=CACHE_SHA, model=model_identity,
                    code={name: sha256(path) for name, path in paths().items()},
                    seed=42, epochs=2, batch_windows=4, parameters=90288,
                    lr=1e-5, weight_decay=1e-4, clip_norm=1., cells_per_head_per_epoch=CELLS,
                    validation_input_seconds=180, validation_score_seconds=170,
                    frame_label='half-open union of physical MIDI sustain intervals',
                    mask='original ownership intersected with 5 <= global_frame < 17995',
                    loss='sum of separately masked unweighted onset and frame BCEWithLogits means',
                    frozen_state_sha256=frozen_before, test_accessed=False, application_enabled=False)
    save(output/'identity.json', identity)
    for name, path in paths().items():
        shutil.copyfile(path, output/(name+'-snapshot'+path.suffix))
    target_proof, replay_proof = [], []
    for record in report['records']:
        name = record['id']
        notes = read(cache/(name+'-labels.json'))['sustain_notes']
        pieces = {}
        ownership = np.zeros(18001, dtype=np.int32)
        began = time.monotonic()
        for item in [r for r in report['windows'] if r['id'] == name]:
            frame = frame_targets(notes, item['index']*5.)
            mask = owned_mask(item['index'])
            np.savez(output/target_file(item), frame=frame, mask=mask)
            indices = 500*item['index']+np.flatnonzero(mask)
            ownership[indices] += 1
            target_proof.append(dict(id=name, split=item['split'], index=item['index'],
                                     file=target_file(item), sha256=sha256(output/target_file(item)),
                                     frames=int(mask.sum()), positive_frame_cells=int(frame[mask].sum())))
            with np.load(cache/item['file'], allow_pickle=False) as data:
                repeated = replay_outputs(model, heads, data)
                if len(repeated) != 7 or any(not np.array_equal(value, data[key]) for key, value in repeated.items()):
                    raise ValueError('Original two-head replay is not bit identical')
                for key, value in repeated.items():
                    pieces.setdefault(key, []).append(value[0])
        expected = np.zeros(18001, dtype=np.int32)
        expected[5:17995] = 1
        if not np.array_equal(ownership, expected):
            raise ValueError('Global supervised frames are not uniquely owned')
        combined = stitch(pieces)
        from piano_transcription_inference.inference import PianoTranscription
        with np.load(cache/(name+'-epoch0-outputs.npz'), allow_pickle=False) as original:
            for key, value in combined.items():
                upstream = PianoTranscription.deframe(None, np.stack(pieces[key]))
                if not np.array_equal(value, original[key]) or not np.array_equal(value, upstream):
                    raise ValueError('Complete replay or official stitching differs')
        predicted = decode(combined)
        if predicted != read(cache/(name+'-epoch0-events.json')):
            raise ValueError('Original two-head decoded events differ')
        save(output/(name+'-epoch0-events.json'), predicted)
        replay_proof.append(dict(id=name, split=record['split'], windows=35,
                                  seven_outputs_bit_identical=True, official_stitch_identical=True,
                                  events_identical=True, supervised_frames=int(ownership.sum()),
                                  elapsed_seconds=time.monotonic()-began))
        save(output/'prepare-progress.json', replay_proof)
        print(json.dumps(dict(preflight_recording=name, complete=len(replay_proof), total=10,
                              seconds=replay_proof[-1]['elapsed_seconds'])), flush=True)
    cells = sum(r['frames'] for r in target_proof if r['split'] == 'train')*88
    if cells != CELLS:
        raise ValueError('Training target count differs from protocol')
    train = [r for r in report['windows'] if r['split'] == 'train']
    gradient = gradient_preflight(model, batch_tensors(cache, output, train[:4]))
    if frozen_digest(model) != frozen_before:
        raise ValueError('Frozen model changed during preflight')
    if any(sha256(path) != identity['code'][name] for name, path in paths().items()):
        raise ValueError('Code changed during preflight')
    save(output/'preflight.json', dict(identity=identity, targets=target_proof, records=replay_proof,
        gradient=gradient, cells_per_head_per_epoch=cells, frozen_model_unchanged=True,
        training_performed=False, test_accessed=False,
        artifacts_sha256={p.name: sha256(p) for p in output.iterdir()}))
    print(json.dumps(dict(preflight_complete=True, windows=350, cells_per_head_per_epoch=cells,
                          gradient=gradient)), flush=True)


def train(cache, output, report):
    import torch
    if (output/'report.json').exists() or (output/'epoch0-heads.pt').exists():
        raise ValueError('Preserve complete or partial training; do not restart in place')
    preflight = read(output/'preflight.json')
    identity = read(output/'identity.json')
    if preflight['identity'] != identity or identity['cache_report_sha256'] != CACHE_SHA:
        raise ValueError('Preflight identity differs')
    for name, digest in preflight['artifacts_sha256'].items():
        if sha256(output/name) != digest:
            raise ValueError('Preflight artifact changed: '+name)
    if any(sha256(path) != identity['code'][name] for name, path in paths().items()):
        raise ValueError('Code changed since preflight')
    model, model_identity = load_frozen_model()
    if model_identity != identity['model'] or frozen_digest(model) != identity['frozen_state_sha256']:
        raise ValueError('Model changed since preflight')
    torch.manual_seed(42)
    np.random.seed(42)
    heads = heads_copy(model)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-5, weight_decay=1e-4)
    generator = np.random.default_rng(42)
    training = [r for r in report['windows'] if r['split'] == 'train']
    history = []
    for epoch in range(3):
        began = time.monotonic()
        totals, denominator, order = dict(onset=0., frame=0.), 0, []
        if epoch:
            order = generator.permutation(len(training)).tolist()
            for start in range(0, len(order), 4):
                batch = batch_tensors(cache, output, [training[i] for i in order[start:start+4]])
                optimizer.zero_grad(set_to_none=True)
                values, cells = losses(model, heads, batch)
                sum(values.values()).backward()
                norm = torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient')
                optimizer.step()
                denominator += cells
                for key in totals:
                    totals[key] += float(values[key].detach())*cells
                if start % 40 == 0:
                    print(json.dumps(dict(epoch=epoch, trained_windows=min(start+4, len(order)),
                                          mean_losses={k: v/denominator for k, v in totals.items()})), flush=True)
            if denominator != CELLS:
                raise ValueError('Supervised cell count changed')
        checkpoint = output/f'epoch{epoch}-heads.pt'
        torch.save(heads.state_dict(), checkpoint)
        validation, comparisons = [], {}
        for name in ROSTER['validation']:
            if epoch:
                pieces = {}
                for item in [r for r in report['windows'] if r['id'] == name]:
                    with np.load(cache/item['file'], allow_pickle=False) as data:
                        repeated = replay_outputs(model, heads, data)
                    for key, value in repeated.items():
                        pieces.setdefault(key, []).append(value[0])
                combined = stitch(pieces)
                with np.load(cache/(name+'-epoch0-outputs.npz'), allow_pickle=False) as original:
                    for key, value in combined.items():
                        if key not in ('reg_onset_output', 'frame_output') and not np.array_equal(value, original[key]):
                            raise ValueError('Frozen output changed')
                predicted = decode(combined)
                np.savez(output/(name+f'-epoch{epoch}-outputs.npz'), **combined)
                save(output/(name+f'-epoch{epoch}-events.json'), predicted)
            else:
                predicted = read(output/(name+'-epoch0-events.json'))
            scope = lambda notes: [n for n in notes if 0 <= n['onset'] < 170]
            labels = read(cache/(name+'-labels.json'))
            references = dict(key=scope(labels['notes']), sustain=scope(labels['sustain_notes']))
            current = scope(predicted['notes'])
            baseline = scope(read(output/(name+'-epoch0-events.json'))['notes'])
            row = dict(id=name, split='validation', predictions_full=len(predicted['notes']),
                       predicted_tails_outside_input=sum(n['offset'] > 180 for n in current),
                       unobservable_reference_offsets={k: sum(n['offset'] > 180 for n in ns)
                                                       for k, ns in references.items()},
                       metrics={kind: {label: metrics(ref, current, strict=strict)
                                       for label, strict in [('50ms', False), ('40ms', True)]}
                                for kind, ref in references.items()})
            comparison = compare_recording(references['key'], references['sustain'], baseline, current)
            comparisons[name] = comparison
            save(output/(name+f'-epoch{epoch}-comparison.json'), comparison)
            validation.append(row)
            print(json.dumps(dict(epoch=epoch, validation=name,
                                  onsets=row['metrics']['key']['50ms']['onsets'],
                                  lost_sustain=len(comparison['50ms']['baseline_sustain_not_guaranteed']),
                                  tail_regressions=len(comparison['50ms']['tail_regressions']))), flush=True)
        selection = eligible(history[0]['validation'] if history else validation, validation, comparisons)
        history.append(dict(epoch=epoch, heads_sha256=sha256(checkpoint),
                            losses={k: v/denominator if denominator else None for k, v in totals.items()},
                            cells_per_head=denominator, order=order, validation=validation,
                            selection=selection, elapsed_seconds=time.monotonic()-began))
        save(output/'progress.json', history)
        if frozen_digest(model) != identity['frozen_state_sha256']:
            raise ValueError('Frozen model changed during training')
    selected = select_epoch(history)
    if any(sha256(path) != identity['code'][name] for name, path in paths().items()):
        raise ValueError('Frozen training code changed')
    save(output/'report.json', dict(identity=identity, preflight_sha256=sha256(output/'preflight.json'),
        epochs=history, selected_epoch=selected, frozen_model_unchanged=True, training_performed=True,
        test_accessed=False, application_enabled=False, direct_listening_performed=False,
        musical_acceptance=False, artifacts_sha256={p.name: sha256(p) for p in output.iterdir()}))
    print(json.dumps(dict(complete=True, selected_epoch=selected)), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cache', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--stage', choices=['prepare', 'train'], required=True)
    args = cli.parse_args()
    cache, output = args.cache.resolve(), args.output.resolve()
    if not output.is_relative_to((ROOT/'test-data').resolve()) or output == cache:
        raise ValueError('Use a separate local research directory')
    report = verify_cache(cache)
    (prepare if args.stage == 'prepare' else train)(cache, output, report)


if __name__ == '__main__':
    main()
