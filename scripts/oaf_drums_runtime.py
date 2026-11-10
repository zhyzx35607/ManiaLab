"""Research-only CPU adapter for fixed Magenta Onsets and Frames drum weights.

Loads selected functions directly from separately acquired Apache-2.0 upstream
source; does not vendor, retrain, or enable the model in the application.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'test-data/oaf-drums/source'
MODEL_REL = Path('magenta/models/onsets_frames_transcription')
SAMPLE_RATE = 44100
HOP = 512
PITCH_GROUPS = {36: 'kick', 38: 'snare', 48: 'tom', 46: 'hi-hat',
                51: 'cymbal', 53: 'other', 49: 'cymbal', 75: 'other'}


def prepare_audio(path):
    """Reproduce the paper-era CLI's mono, peak and PCM16 preprocessing.

    Float WAV samples (including values outside +/-1) survive until peak
    normalization. The int16 round trip uses 32767 and truncation as upstream.
    """
    import numpy as np
    import librosa
    from scipy.io import wavfile
    rate, y = wavfile.read(path)
    info = dict(source_rate=int(rate), source_samples=len(y), source_dtype=str(y.dtype),
                source_shape=list(y.shape))
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32767
    elif y.dtype != np.float32:
        raise ValueError('Official reader supports PCM16 or float32 WAV only')
    if y.ndim == 2 and y.shape[1] == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected nonempty finite mono/stereo PCM')
    info['source_mono_peak'] = float(np.max(np.abs(y)))
    if rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=rate, target_sr=SAMPLE_RATE,
                             res_type='kaiser_best', fix=True, scale=False)
    info['resampled_peak_before_normalization'] = float(np.max(np.abs(y)))
    y = librosa.util.normalize(y, norm=np.inf)
    pcm = (y * 32767).astype(np.int16)
    y = pcm.astype(np.float32) / 32767
    info.update(input_samples=len(y), input_rate=SAMPLE_RATE,
                pcm16_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                processed_float_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    return y, info


def spectrogram(y):
    import numpy as np
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=2048,
        hop_length=HOP, win_length=2048, window='hann', center=True,
        pad_mode='reflect', power=2.0, n_mels=250, fmin=30.0,
        fmax=SAMPLE_RATE/2, htk=True, norm='slaney', dtype=np.float32)
    return librosa.power_to_db(mel.astype(np.float32).T, ref=1.0,
                              amin=1e-10, top_db=80.0)


def decode(probabilities, velocities):
    """Official per-frame threshold rule: no peak pick, dedup or snapping.

    Every one of the 88 output pitches remains visible, including noncanonical
    pitches. Fixed MIDI display tails are intentionally not represented as LNs.
    """
    import numpy as np
    if probabilities.shape != velocities.shape or probabilities.ndim != 2 or probabilities.shape[1] != 88:
        raise ValueError('Expected frame by 88 probabilities and velocities')
    if not np.isfinite(probabilities).all() or not np.isfinite(velocities).all():
        raise ValueError('Nonfinite model output')
    events = []
    for frame, index in np.argwhere(probabilities > 0.5):
        pitch = int(index) + 21
        velocity = int(np.clip(velocities[frame, index], 0, 1) * 127)
        events.append(dict(frame=int(frame), pitch=pitch,
                           onset=float(frame * HOP / SAMPLE_RATE),
                           kind=PITCH_GROUPS.get(pitch, 'unknown'),
                           probability=float(probabilities[frame, index]),
                           velocity_value=float(velocities[frame, index]), velocity=velocity))
    return events


def functions(path, names, namespace):
    """Execute only named, unmodified upstream function definitions."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError('Missing upstream function')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def hparams():
    """Merge literal model and drums configuration, preserving spec_hop_length."""
    model = ast.parse((SOURCE / MODEL_REL / 'model_tpu.py').read_text())
    defaults = next(n for n in model.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'get_default_hparams')
    values = {k.arg: ast.literal_eval(k.value) for k in defaults.body[-1].value.keywords}
    config = ast.parse((SOURCE / MODEL_REL / 'configs.py').read_text())
    calls = [n for n in ast.walk(config) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'HParams']
    general = next(n for n in calls if any(k.arg == 'spec_hop_length' for k in n.keywords))
    drums = next(n for n in calls if any(k.arg == 'drum_data_map' for k in n.keywords)
                 and any(k.arg == 'drums_only' for k in n.keywords))
    values = {**{k.arg: ast.literal_eval(k.value) for k in general.keywords}, **values,
              **{k.arg: ast.literal_eval(k.value) for k in drums.keywords}}
    if (values['sample_rate'], values['spec_hop_length'], values['spec_n_bins']) != (44100, 512, 250):
        raise ValueError('Unreviewed frontend configuration')
    return SimpleNamespace(**values)


def build_graph():
    import tensorflow.compat.v1 as tf
    import tf_slim as slim
    from tensorflow.python.framework import ops
    # tf_slim 1.1.0 performs an isinstance check through an alias removed in
    # TF 2.15. Restore only that alias; TensorFlow operations are unchanged.
    if not hasattr(ops, 'Tensor'):
        ops.Tensor = tf.Tensor
    tf.disable_v2_behavior()
    hp = hparams()
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(42)
        recurrent = dict(tf=tf)
        functions(SOURCE/'magenta/contrib/rnn.py', ['stack_bidirectional_dynamic_rnn'], recurrent)
        namespace = dict(tf=tf, slim=slim, constants=SimpleNamespace(MIDI_PITCHES=88),
                         contrib_rnn=SimpleNamespace(**recurrent))
        functions(SOURCE/MODEL_REL/'model_tpu.py',
                  ['conv_net', 'lstm_layer', 'acoustic_model', 'build_model'], namespace)
        spec = tf.placeholder(tf.float32, [1, None, 250, 1], name='research_spec')
        length = tf.placeholder(tf.int32, [1], name='research_length')
        # TF Slim's own fused inference path avoids removed Keras legacy imports.
        # No training-statistic updates occur with is_training=False.
        with slim.arg_scope([slim.batch_norm], fused=True, updates_collections=None):
            _, onset, _, velocity = namespace['build_model'](spec, length, hp, False)
        probability = tf.sigmoid(onset)
        variables = tf.global_variables()
        saver = tf.train.Saver(variables)
    config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1,
                            device_count={'GPU': 0})
    return graph, tf.Session(graph=graph, config=config), spec, length, probability, velocity, variables, saver


def verify_weights(session, variables, checkpoint):
    """Check every inference variable against the stored tensor, before/after."""
    import numpy as np
    import tensorflow.compat.v1 as tf
    reader = tf.train.load_checkpoint(str(checkpoint))
    rows = []
    for variable in variables:
        name = variable.name.removesuffix(':0')
        actual = session.run(variable)
        expected = reader.get_tensor(name)
        if not np.array_equal(actual, expected):
            raise ValueError('Checkpoint tensor mismatch: ' + name)
        rows.append(dict(name=name, shape=list(actual.shape),
                         sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    model_names = {n for n, _ in tf.train.list_variables(str(checkpoint))
                   if n.startswith(('onsets/', 'offsets/', 'velocity/', 'frame/'))
                   and '/Adam' not in n}
    if {r['name'] for r in rows} != model_names:
        raise ValueError('Model variable inventory differs from checkpoint')
    return rows
