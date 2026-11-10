"""Fixed eight-family separation controls using the author's Mega 53 weights.

Research only: no application defaults, automatic instrument discovery or LN
decisions. Inputs are byte-pinned to the completed Banquet control experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_banquet_separation import wave_metrics

ASSETS = ROOT / 'test-data/mega53-source-review-v1'
SOURCE = ASSETS / 'source'
SOURCE_COMMIT = '1d4b3c431b37de696b3795ed2018493aacba946f'
CHECKPOINT_SHA = 'c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f'
CONFIG_SHA = '7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103'
RATE = 44100
CASES = [('12-vn', 'violin'), ('12-vc', 'cello'), ('13-va', 'viola'),
         ('25-sax', 'saxophone'), ('31-tpt', 'trumpet'), ('31-hn', 'french-horn'),
         ('31-tbn', 'trombone'), ('38-db', 'double-bass')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def prepare():
    checkpoint = ASSETS / 'mvsep_mega_model_bs_roformer_53_stems_v1.ckpt'
    config = ASSETS / 'full-config.txt'
    assert sha(checkpoint) == CHECKPOINT_SHA and sha(config) == CONFIG_SHA
    git = lambda *args: subprocess.check_output(
        ['git', '-C', str(SOURCE), *args], text=True).strip()
    assert git('rev-parse', 'HEAD') == SOURCE_COMMIT
    assert not git('diff', 'HEAD', '--name-only')
    previous = ROOT / 'test-data/banquet-separation-v1'
    report = json.loads((previous / 'report.json').read_text(encoding='utf-8'))
    assert [row['case']['name'] for row in report['rows']] == [name for name, _ in CASES]
    rows = []
    for (name, target), row in zip(CASES, report['rows']):
        inputs = previous / name / 'inputs.npz'
        assert sha(inputs) == row['inputs_sha256']
        rows.append(dict(name=name, target=target, inputs=str(inputs),
                         inputs_sha256=sha(inputs)))
    source_files = {p: sha(SOURCE / p) for p in (
        'models/bs_roformer/bs_roformer.py', 'models/bs_roformer/attend.py', 'LICENSE')}
    return dict(cases=rows, source_commit=SOURCE_COMMIT, source_files=source_files,
        checkpoint_sha256=CHECKPOINT_SHA, config_sha256=CONFIG_SHA,
        author_release='v1.0.21', runner_sha256=sha(__file__),
        metric_source_sha256=sha(ROOT / 'scripts/check_banquet_separation.py'),
        original_report_sha256=sha(previous / 'report.json'),
        sample_rate=RATE, window_seconds=6, source_seconds=[20, 26],
        variants=['mixture', 'target', 'absent', 'silence'],
        threads={'intra': 2, 'inter': 1}, dtype='float32', normalization=False,
        target_selection='known_source_family_not_automatic_detection',
        model_parallel=False, enabled_in_application=False,
        notes_or_chart_labels_read=False)


def load_model():
    sys.path.insert(0, str(ASSETS / 'deps'))
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)

    # Load only the reviewed BS-RoFormer modules. The upstream package __init__
    # also imports unrelated Conformer models and their optional dependencies.
    for name, folder in [('models', SOURCE / 'models'),
                         ('models.bs_roformer', SOURCE / 'models/bs_roformer')]:
        assert name not in sys.modules
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    from models.bs_roformer.bs_roformer import BSRoformer

    class ConfigLoader(yaml.SafeLoader):
        pass
    ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                lambda loader, node: tuple(loader.construct_sequence(node)))
    config = yaml.load((ASSETS / 'full-config.txt').read_text(encoding='utf-8'), Loader=ConfigLoader)
    instruments = config['training']['instruments']
    assert len(instruments) == 53 and len(set(instruments)) == 53
    assert config['inference']['normalize'] is False
    desired = [name for _, name in CASES]
    original_ids = [instruments.index(name) for name in desired]
    remap = {old: new for new, old in enumerate(original_ids)}
    state = torch.load(ASSETS / 'mvsep_mega_model_bs_roformer_53_stems_v1.ckpt',
                       map_location='cpu', weights_only=True, mmap=True)
    selected = {}
    all_heads = set()
    for key, value in state.items():
        assert isinstance(value, torch.Tensor) and value.is_floating_point()
        if key.startswith('mask_estimators.'):
            parts = key.split('.')
            head = int(parts[1]); all_heads.add(head)
            if head not in remap:
                continue
            parts[1] = str(remap[head]); key = '.'.join(parts)
        selected[key] = value
    assert all_heads == set(range(53))
    # Heads operate independently after the shared trunk. Keep all eight
    # predetermined heads, without allocating the other 45 decoders.
    model = BSRoformer(**(config['model'] | {'num_stems': len(desired)})).eval()
    loaded = model.load_state_dict(selected, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    for key, actual in model.state_dict().items():
        assert torch.equal(actual, selected[key].to(dtype=actual.dtype)), key
    identity = dict(source_commit=SOURCE_COMMIT, checkpoint_sha256=CHECKPOINT_SHA,
        config_sha256=CONFIG_SHA, original_stems=instruments, selected_stems=desired,
        original_head_ids=original_ids, selected_tensors=len(selected),
        all_checkpoint_tensors=len(state), strict_load=True, exact_tensor_copy=True,
        unused_heads_removed=45, upstream_forward_unmodified=True,
        original_checkpoint_dtype=sorted({str(t.dtype) for t in state.values()}),
        parameter_count=sum(p.numel() for p in model.parameters()),
        versions={s: importlib.metadata.version(s) for s in
                  ('torch', 'einops', 'rotary-embedding-torch', 'beartype', 'PyYAML')})
    return model, identity


def run(output):
    import numpy as np
    assert not output.exists(), 'Preserve previous complete or incomplete evidence'
    plan = prepare()
    save(output / 'plan.json', plan)
    model, identity = load_model()
    save(output / 'model-identity.json', identity)
    import torch
    results = []
    for head, row in enumerate(plan['cases']):
        assert sha(row['inputs']) == row['inputs_sha256']
        with np.load(row['inputs'], allow_pickle=False) as data:
            arrays = {k: data[k] for k in plan['variants']}
        assert all(x.shape == (2, RATE * 6) and x.dtype == np.float32 and
                   np.isfinite(x).all() for x in arrays.values())
        folder = output / row['name']; folder.mkdir()
        variants = []
        for variant in plan['variants']:
            wave = arrays[variant]
            started = time.monotonic()
            with torch.inference_mode():
                result = model(torch.from_numpy(wave.copy())[None], active_stem_ids=[head])
            assert result.shape == (1, 1, 2, RATE * 6) and torch.isfinite(result).all()
            estimate = result[0, 0].numpy()
            elapsed = time.monotonic() - started
            np.save(folder / (variant + '.npy'), estimate, allow_pickle=False)
            reference = arrays['target'] if variant in ('target', 'mixture') else arrays['silence']
            scores = {}
            for name, begin, end in [('full', 0, 6), ('interior', 2, 4)]:
                sl = slice(begin * RATE, end * RATE)
                scores[name] = dict(input=wave_metrics(reference[:, sl], wave[:, sl]),
                                    output=wave_metrics(reference[:, sl], estimate[:, sl]))
            item = dict(variant=variant, seconds=elapsed, scores=scores,
                        estimate_sha256=sha(folder / (variant + '.npy')))
            variants.append(item)
            print(json.dumps(dict(case=row['name'], variant=variant, seconds=elapsed,
                                  full=scores['full']['output'])), flush=True)
        result = dict(case=row, variants=variants)
        save(folder / 'report.json', result); results.append(result)
    assert sha(__file__) == plan['runner_sha256']
    assert sha(ROOT / 'scripts/check_banquet_separation.py') == plan['metric_source_sha256']
    assert all(sha(SOURCE / name) == digest for name, digest in plan['source_files'].items())
    save(output / 'report.json', dict(plan=plan, rows=results,
        model_identity_sha256=sha(output / 'model-identity.json'),
        seconds=sum(v['seconds'] for r in results for v in r['variants']),
        musical_acceptance=False, default_changed=False,
        limitations=['Known target family, not automatic instrument detection.',
            'Eight families share five pieces; public training overlap is unknown.',
            '16k mono sources upsampled and duplicated; not original stereo masters.',
            'Single 6-second forward, not the default 20-second overlapped inference.',
            'Controlled mixtures, not Nova stems or mastered pop songs.',
            'No note transcription or LN quality evaluation.']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--prepare-only', action='store_true')
    args = cli.parse_args()
    if args.prepare_only:
        print(json.dumps(prepare(), indent=2))
        assert 'torch' not in sys.modules
    else:
        if args.output is None:
            cli.error('--output is required')
        run(args.output)
