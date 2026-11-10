"""Recover pinned MuScriptor GGUF tensors for later upstream CPU research.

Offline data conversion only. Preserves the published F16/F32 values; does not
restore the original F32 checkpoint, load a neural model or change the app.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = ROOT / 'test-data/muscriptor-cpp-review-v1'
CPP_COMMIT = '8589219cc3b584bce970f95aec2ec4a04e53b280'
HF_REVISION = '13a851294c5b5701966e850c7eba326f9da97e8d'
WEIGHT_NAME = 'muscriptor-small-f16.gguf'
WEIGHT_BYTES = 209425152
WEIGHT_SHA = '925f55af65a20ebc4f8b45ceaf095a12b72493d436cb112623cd0041a1af23d4'
CONFIG = dict(dim=768, num_heads=12, num_layers=14, card=1393)
SOURCE_HASHES = {
    'msl/__init__.py': '511c3477e2583280a5d3f8e77fb0c647d577d7d897c6827df6b063d2ed44b136',
    'msl/checksum.py': 'e88b28df4704bdffcf017069a311b439f239952ae058ecb756fef2d6e19db006',
    'msl/paths.py': '94ae1f9d422af5bfb69e746489f94679f9eba3a2992d261a2a70d8d304b664bd',
    'msl/tensor_io.py': '152e35495c62f4dbe5bc2ffc7a215041ce838ec53f45373c122b6957766e0c80',
    'msl/notice.py': '1a33ce3fa5f1f20325c0587ed8dc1b93296062d77a7c30e987a90b473d99fe97',
    'msl/convert_gguf.py': '05d0fd354ab59f0fe63a557a94f3a87096cf906baf5f415924f777e7dcaf44c7',
    'LICENSE': 'a27ec7144ad7a2427c9bcc29dca0e16698dcb3941980d2c714ac32e7334c04a8',
}
LICENSE_HASHES = {
    'LICENSE': '41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0',
    'NOTICE.md': '70d98970b1cae063002356f369732a0772946eb6271865bfa91cbb2e03dd7ae3',
}
GLOBAL_SHAPES = {
    'token_embd.weight': (1394, 768), 'output.weight': (1393, 768),
    'output_norm.weight': (768,), 'output_norm.bias': (768,),
    'cond.mel_fb.weight': (512, 1025), 'cond.stft_window': (2048,),
    'cond.proj.weight': (768, 512), 'cond.proj.bias': (768,),
    'cond.instrument_group.weight': (1001, 768), 'cond.dataset_name.weight': (5, 768),
}
LAYER_SHAPES = {
    'attn_norm.weight': (768,), 'attn_norm.bias': (768,),
    'ffn_norm.weight': (768,), 'ffn_norm.bias': (768,),
    'attn_qkv.weight': (2304, 768), 'attn_out.weight': (768, 768),
    'ffn_up.weight': (3072, 768), 'ffn_down.weight': (768, 3072),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def prepare(assets):
    weight = assets / WEIGHT_NAME
    require(weight.stat().st_size == WEIGHT_BYTES and sha(weight) == WEIGHT_SHA,
            'The complete pinned public small GGUF is required')
    for name, expected in SOURCE_HASHES.items():
        require(sha(assets / 'source' / name) == expected, f'Converter source changed: {name}')
    for name, expected in LICENSE_HASHES.items():
        require(sha(assets / 'hf' / name) == expected, f'Weight notice changed: {name}')
    require('torch' not in sys.modules and 'onnxruntime' not in sys.modules, 'Run in a fresh data-only process')
    return dict(kind='offline_gguf_tensor_recovery', cpp_commit=CPP_COMMIT,
        hf_repo='DamRsn/muscriptor-gguf', hf_revision=HF_REVISION,
        gguf_sha256=WEIGHT_SHA, gguf_bytes=WEIGHT_BYTES, config=CONFIG,
        converter_source_sha256=SOURCE_HASHES, weight_license_sha256=LICENSE_HASHES,
        runner_sha256=sha(__file__), original_float32_restored=False,
        precision='Published GGUF mixed F16/F32 preserved exactly',
        neural_model_loaded=False, inference_performed=False, application_changed=False)


def mapping_from_author(converter, np):
    rows = []
    require(len(converter._GLOBAL_MAP) == 10 and len(converter._LAYER_MAP) == 8,
            'Unexpected author mapping topology')
    for source, (name, keep_f32, transform) in converter._GLOBAL_MAP.items():
        require(transform is None or (name == 'cond.mel_fb.weight' and transform is np.transpose),
                f'Unsupported transform: {name}')
        rows.append(dict(gguf=name, source_key=source, gguf_shape=list(GLOBAL_SHAPES[name]),
                         dtype='float32' if keep_f32 else 'float16', transpose=transform is not None))
    for index in range(CONFIG['num_layers']):
        for source, (name, keep_f32) in converter._LAYER_MAP.items():
            rows.append(dict(gguf=f'blk.{index}.{name}', source_key=f'transformer.layers.{index}.{source}',
                             gguf_shape=list(LAYER_SHAPES[name]),
                             dtype='float32' if keep_f32 else 'float16', transpose=False))
    for row in rows:
        key = row['source_key']
        row['current_upstream_key'] = ('emb.' + key[len('emb.0.'):] if key.startswith('emb.0.') else
            'linear.' + key[len('linears.0.'):] if key.startswith('linears.0.') else key)
        row['restored_shape'] = row['gguf_shape'][::-1] if row['transpose'] else row['gguf_shape']
    require(len(rows) == len({r['gguf'] for r in rows}) == len({r['source_key'] for r in rows}) == 122,
            'Missing or duplicate author tensor mapping')
    return rows


def validate_arrays(arrays, rows, np, *, restored=False):
    key = 'source_key' if restored else 'gguf'
    shape = 'restored_shape' if restored else 'gguf_shape'
    expected = {row[key] for row in rows}
    require(set(arrays) == expected,
            f'Tensor names differ: missing={sorted(expected - set(arrays))}, extra={sorted(set(arrays) - expected)}')
    for row in rows:
        value = arrays[row[key]]
        require(value.shape == tuple(row[shape]), f'Wrong shape: {row[key]}')
        require(value.dtype == np.dtype(row['dtype']), f'Wrong dtype: {row[key]}')
        require(np.isfinite(value).all(), f'Nonfinite tensor: {row[key]}')


def validate_metadata(metadata):
    expected = {
        'general.architecture': 'muscriptor', 'general.license': 'cc-by-nc-4.0',
        'muscriptor.format_version': 1, 'muscriptor.embedding_length': 768,
        'muscriptor.block_count': 14, 'muscriptor.attention.head_count': 12,
        'muscriptor.attention.head_dim': 64, 'muscriptor.feed_forward_length': 3072,
        'muscriptor.vocab_size': 1393, 'muscriptor.initial_token_id': 1393,
        'muscriptor.logit_mask_start': 1393, 'muscriptor.audio.sample_rate': 16000,
        'muscriptor.audio.n_fft': 2048, 'muscriptor.audio.hop_length': 160,
        'muscriptor.audio.frame_rate': 100, 'muscriptor.audio.n_mels': 512,
        'muscriptor.source.weight_dtype': 'f16',
    }
    for key, value in expected.items():
        require(metadata.get(key) == value, f'Unexpected GGUF metadata: {key}')
    for key, value in {'muscriptor.audio.log_eps': 1e-6,
                       'muscriptor.attention.layer_norm_epsilon': 1e-5,
                       'muscriptor.position_embedding.max_period': 10000.}.items():
        require(isinstance(metadata.get(key), (int, float)) and
                math.isclose(metadata[key], value, rel_tol=1e-7), f'Unexpected GGUF float: {key}')
    require(re.fullmatch('[0-9a-f]{40}', metadata.get('muscriptor.source.revision', '')),
            'Missing original checkpoint revision metadata')
    require(re.fullmatch('[0-9a-f]{64}', metadata.get('muscriptor.source.sha256', '')),
            'Missing original checkpoint hash metadata')
    require(bool(metadata.get('muscriptor.license.notice')), 'Missing embedded license notice')


def corruption_controls(np):
    rows = [dict(gguf='sentinel', source_key='source', gguf_shape=[2, 3], restored_shape=[3, 2], dtype='float32')]
    valid = {'sentinel': np.zeros((2, 3), dtype=np.float32)}
    validate_arrays(valid, rows, np)
    nan = valid['sentinel'].copy(); nan[0, 0] = np.nan
    cases = {'missing': {}, 'extra': {**valid, 'unexpected': valid['sentinel']},
             'shape': {'sentinel': np.zeros((3, 2), dtype=np.float32)},
             'dtype': {'sentinel': np.zeros((2, 3), dtype=np.float16)},
             'nonfinite': {'sentinel': nan}}
    results = []
    for name, arrays in cases.items():
        try:
            validate_arrays(arrays, rows, np)
        except ValueError as error:
            results.append(dict(case=name, rejected=True, reason=str(error)))
        else:
            raise ValueError(f'Corruption control was accepted: {name}')
    return results


def run(assets, output):
    require(not output.exists(), 'Preserve previous complete or partial evidence; use a fresh output directory')
    plan = prepare(assets)
    sys.path.insert(0, str(assets / 'deps'))
    sys.path.insert(0, str(assets / 'source'))
    import numpy as np
    import gguf
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    converter = importlib.import_module('msl.convert_gguf')
    require(Path(converter.__file__).resolve() == (assets / 'source/msl/convert_gguf.py').resolve(),
            'Imported converter did not come from the pinned source')
    require(importlib.metadata.version('gguf') == '0.19.0' and
            importlib.metadata.version('safetensors') == '0.5.3', 'Unexpected data-reader version')
    rows = mapping_from_author(converter, np)
    reader = gguf.GGUFReader(assets / WEIGHT_NAME, mode='r')
    metadata = {name: field.contents() for name, field in reader.fields.items()}
    validate_metadata(metadata)
    require(len(reader.tensors) == len({tensor.name for tensor in reader.tensors}) == 122,
            'Duplicate, missing or extra GGUF tensors')
    original = {tensor.name: tensor.data for tensor in reader.tensors}
    validate_arrays(original, rows, np)
    for tensor in reader.tensors:
        require(tensor.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32),
                f'Unexpected quantization: {tensor.name}')
    restored = {row['source_key']: np.ascontiguousarray(
        original[row['gguf']].T if row['transpose'] else original[row['gguf']]) for row in rows}
    validate_arrays(restored, rows, np, restored=True)

    def array_sha(array):
        return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()

    # Execute the author's unchanged forward mapping against our restored keys.
    forward = converter._translate(restored, CONFIG['num_layers'], 'f16')
    validate_arrays(forward, rows, np)
    for row in rows:
        before, after = original[row['gguf']], forward[row['gguf']]
        require(array_sha(before) == array_sha(after), f'Author round-trip differs: {row["gguf"]}')
        row.update(gguf_data_sha256=array_sha(before), restored_data_sha256=array_sha(restored[row['source_key']]),
                   author_forward_roundtrip_bit_exact=True, elements=int(before.size), bytes=int(before.nbytes))
    del forward
    controls = corruption_controls(np)
    require('torch' not in sys.modules and 'onnxruntime' not in sys.modules, 'Unexpected neural runtime import')
    output.mkdir(parents=True)
    save(output / 'plan.json', plan)
    save(output / 'config.json', CONFIG)
    save(output / 'gguf-metadata.json', metadata)
    save(output / 'tensor-mapping.json', rows)
    save(output / 'corruption-controls.json', controls)
    safetensor_metadata = dict(format='pt', source_gguf_sha256=WEIGHT_SHA,
        source_gguf_repo='DamRsn/muscriptor-gguf', source_gguf_revision=HF_REVISION,
        original_float32_restored='false', precision='published F16/F32 values preserved', license='cc-by-nc-4.0')
    checkpoint = output / 'muscriptor-small-recovered-f16.safetensors'
    save_file(restored, str(checkpoint), metadata=safetensor_metadata)
    del restored
    reread = load_file(str(checkpoint))
    validate_arrays(reread, rows, np, restored=True)
    for row in rows:
        require(array_sha(reread[row['source_key']]) == row['restored_data_sha256'],
                f'Safetensors readback differs: {row["source_key"]}')
        row['safetensors_readback_bit_exact'] = True
    with safe_open(str(checkpoint), framework='numpy') as handle:
        require(handle.metadata() == safetensor_metadata, 'Safetensors metadata changed')
    # Keep the first mapping artifact immutable; readback evidence is separate.
    save(output / 'readback.json', dict(tensors=122, all_bit_exact=True,
        tensors_sha256={row['source_key']: row['restored_data_sha256'] for row in rows}))
    for source, name in [(assets / 'hf/LICENSE', 'WEIGHTS-LICENSE.txt'),
                         (assets / 'hf/NOTICE.md', 'WEIGHTS-NOTICE.md'),
                         (assets / 'source/LICENSE', 'CONVERTER-LICENSE.txt')]:
        shutil.copyfile(source, output / name)
        require(sha(source) == sha(output / name), f'Copied notice changed: {name}')
    save(output / 'license-summary.json', dict(weights='CC BY-NC 4.0', converter='MIT',
        original_authors='Mirelo and Kyutai', gguf_converter='Damien Ronssin',
        notices=['WEIGHTS-LICENSE.txt', 'WEIGHTS-NOTICE.md', 'CONVERTER-LICENSE.txt'],
        changes='Inverse tensor names and mel transpose only; original GGUF mixed precision retained.',
        original_checkpoint_identity='Recorded from pinned GGUF metadata; original gated file was not obtained or verified.'))
    require(prepare(assets) == plan, 'Input identity changed during recovery')
    require('torch' not in sys.modules and 'onnxruntime' not in sys.modules, 'Unexpected neural runtime import')
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    report = dict(completed=True, plan=plan, plan_sha256=artifacts['plan.json'], tensors=122,
        tensor_values=sum(row['elements'] for row in rows), tensor_bytes=sum(row['bytes'] for row in rows),
        dtypes={name: sum(row['dtype'] == name for row in rows) for name in ('float16', 'float32')},
        transposed_tensors=[row['gguf'] for row in rows if row['transpose']],
        author_forward_roundtrip_bit_exact=True, safetensors_readback_bit_exact=True,
        controls_rejected=len(controls), artifacts=artifacts,
        runtime_versions={name: importlib.metadata.version(name) for name in ('numpy', 'gguf', 'safetensors')},
        original_source_revision=metadata['muscriptor.source.revision'],
        original_source_sha256=metadata['muscriptor.source.sha256'],
        neural_model_loaded=False, inference_performed=False, application_changed=False,
        limitations=['Original F32 information lost by the published GGUF conversion is not recovered.',
            'Tensor round-trip is not a Torch model strict-load or numerical inference check.',
            'No music, note, timing, instrument or LN quality evaluation has been performed.'])
    save(output / 'report.json', report)
    print(json.dumps(dict(output=str(output), report_sha256=sha(output / 'report.json'),
                         checkpoint_sha256=artifacts[checkpoint.name], tensors=122, bit_exact=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        print(json.dumps(prepare(args.assets.resolve()), indent=2))
    elif args.output is None:
        parser.error('--output is required')
    else:
        run(args.assets.resolve(), args.output.resolve())
