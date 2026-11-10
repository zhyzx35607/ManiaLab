"""Fixed full-recording OaF drum regression; research output only."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMBA_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.probe_adt_drums import roster
from scripts.oaf_drums_runtime import (SOURCE, SAMPLE_RATE, HOP, build_graph,
    prepare_audio, spectrogram, decode, verify_weights, hparams)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fixed_source():
    base = ROOT/'test-data/oaf-drums'
    manifest = read(SOURCE/'source-manifest.json')
    if manifest['commit'] != 'c15687ebd1c1cc12658cd468c737d3c24c02b918':
        raise ValueError('Unexpected upstream source')
    files = {r['path']: r['sha256'] for r in manifest['files']}
    extra = read(base/'additional-source-manifest.json')
    rnn = next(r for r in extra if r['folder'] == 'source' and r['path'] == 'magenta/contrib/rnn.py')
    if rnn['revision'] != manifest['commit']:
        raise ValueError('Recurrent source revision differs')
    files[rnn['path']] = rnn['sha256']
    for name, digest in files.items():
        if sha256(SOURCE/name) != digest:
            raise ValueError('Upstream source changed: '+name)
    checkpoint = read(base/'checkpoint-identity.json')
    if checkpoint['zip_sha256'] != '09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0':
        raise ValueError('Wrong model archive')
    if sha256(base/'e-gmd_checkpoint.zip') != checkpoint['zip_sha256']:
        raise ValueError('Model archive changed')
    for row in checkpoint['files']:
        if sha256(base/'checkpoint'/row['name']) != row['sha256']:
            raise ValueError('Checkpoint changed: '+row['name'])
    return files, checkpoint


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local research directory')
    source, checkpoint_identity = fixed_source()
    records, manifest_hash, roster_hash = roster()
    paths = {'runner': Path(__file__), 'runtime': ROOT/'scripts/oaf_drums_runtime.py',
             'protocol': ROOT/'docs/oaf-drums-protocol.md'}
    identity = dict(source_commit='c15687ebd1c1cc12658cd468c737d3c24c02b918',
        source_hashes=source, checkpoint=checkpoint_identity,
        code={k: sha256(p) for k, p in paths.items()},
        hparams=vars(hparams()), dependencies={n: importlib.metadata.version(n)
        for n in ('tensorflow-intel', 'tf-slim', 'librosa', 'resampy', 'numpy', 'scipy', 'setuptools')},
        input_roster_sha256=roster_hash, manifest_sha256=manifest_hash, inputs=records,
        graph_mode='full_recording_no_padding', threads=2, inter_threads=1,
        oneDNN=False, application_enabled=False, training_performed=False,
        reserved_piano_test_accessed=False, direct_listening_performed=False)
    output.mkdir(parents=True)
    save(output/'identity.json', identity)
    for name, path in paths.items():
        shutil.copyfile(path, output/(name+'-snapshot'+path.suffix))
    save(output/'progress.json', [])
    graph, session, x, length, onset, velocity, variables, saver = build_graph()
    checkpoint = ROOT/'test-data/oaf-drums/checkpoint/model.ckpt-569400'
    saver.restore(session, str(checkpoint))
    proof = verify_weights(session, variables, checkpoint)
    save(output/'model-proof-before.json', proof)
    print(json.dumps(dict(loaded=True, model_variables=len(proof))), flush=True)
    progress = []
    import numpy as np
    for record in records:
        started = time.monotonic()
        folder = output/record['id']
        folder.mkdir()
        pcm, metadata = prepare_audio(record['audio'])
        spec = spectrogram(pcm)
        # data.wav_to_num_frames: official numpy length before model trimming.
        frame_count = int(np.int32(len(pcm)/SAMPLE_RATE*(SAMPLE_RATE/HOP)))
        metadata.update(full_mel_frames=len(spec), model_frames=frame_count,
                        mel_sha256=__import__('hashlib').sha256(spec.tobytes()).hexdigest())
        save(folder/'frontend.json', metadata)
        print(json.dumps(dict(id=record['id'],stage='prepared',frames=frame_count)), flush=True)
        predicted, velocities = session.run([onset, velocity],
            {x: spec[None, :frame_count, :, None], length: [frame_count]})
        del pcm, spec
        probabilities, velocities = predicted[0], velocities[0]
        events = decode(probabilities, velocities)
        np.savez_compressed(folder/'outputs.npz', onset=probabilities, velocity=velocities)
        save(folder/'events.json', events)
        row = dict(**record, duration_seconds=metadata['source_samples']/metadata['source_rate'],
                   frames=frame_count, events=len(events), kinds=dict(Counter(e['kind'] for e in events)),
                   elapsed_seconds=time.monotonic()-started,
                   artifacts_sha256={p.name: sha256(p) for p in folder.iterdir()})
        progress.append(row)
        save(output/'progress.json', progress)
        print(json.dumps(row), flush=True)
        del predicted, probabilities, velocities, events
    after = verify_weights(session, variables, checkpoint)
    if after != proof:
        raise ValueError('Inference mutated model weights')
    save(output/'model-proof-after.json', after)
    session.close()
    save(output/'report.json', dict(records=progress, frozen_model_unchanged=True,
         identity_sha256=sha256(output/'identity.json'), complete_corpus=len(progress) == 24,
         artifacts_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(complete=True, records=len(progress))), flush=True)


if __name__ == '__main__':
    main()
