"""Research piano runtime with explicit release provenance; not a default engine.

The ONNX weights represent Kong et al.'s onset/offset regression model. This
module uses sample-clock overlap stitching and a separate NumPy event decoder.
Pedal-extended acoustic intervals are evidence, not authored LN release choices.
"""
from math import ceil, gcd
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

RATE, WINDOW, HOP, FPS = 16000, 160000, 80000, 100
OUTPUTS = ('reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output')
MODEL_SHA256 = '7488435a431ba2863de16c56be24f3626dc75e38cc22f3103e193a677494fafa'


def infer(samples, sample_rate, session, progress=lambda p,s:None, cancelled=lambda:False,
          continuation_trace=None):
    from .piano_continuity import continue_overlap
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('钢琴输入必须是有限的单声道采样')
    if not isinstance(sample_rate,int) or sample_rate <= 0:
        raise ValueError('钢琴采样率无效')
    duration = len(samples)/sample_rate
    if sample_rate != RATE:
        common = gcd(sample_rate, RATE)
        samples = resample_poly(samples, RATE//common, sample_rate//common).astype(np.float32)
    padded_length = ceil(len(samples)/WINDOW)*WINDOW
    starts = list(range(0, padded_length-WINDOW+1, HOP))
    pieces = {key:[] for key in OUTPUTS}
    previous = None
    for index,start in enumerate(starts):
        if cancelled():
            raise InterruptedError('已取消钢琴分析')
        chunk = np.zeros(WINDOW,dtype=np.float32)
        available = samples[start:start+WINDOW]
        chunk[:len(available)] = available
        outputs = session.run(list(OUTPUTS),{'audio':chunk[None]})
        for output in outputs:
            if output.shape != (1,1001,88) or not np.isfinite(output).all():
                raise ValueError('钢琴网络输出形状或数值无效')
        raw = {key:output[0] for key,output in zip(OUTPUTS,outputs)}
        if previous is None:
            current = raw
        else:
            current, changes = continue_overlap(previous, raw,
                owned_left=0 if index == 1 else 250, require_release_regression=True)
            if continuation_trace is not None:
                continuation_trace.extend(dict(window_start_seconds=start/RATE, **change) for change in changes)
        for key,output in current.items():
            # Keep the central five seconds; retain context at the song edges.
            left = 0 if index == 0 else 250
            right = 1001 if len(starts) == 1 else 1000 if index == len(starts)-1 else 750
            pieces[key].append(output[left:right])
        previous = raw
        progress(round((index+1)*100/len(starts)),f'钢琴起音与持续分析 {index+1}/{len(starts)}')
    valid = ceil(duration*FPS)
    return {key:np.concatenate(rows)[:valid] for key,rows in pieces.items()}


def regression_peaks(values, threshold, radius):
    """Monotonic regression peaks with a sub-frame triangular interpolation."""
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Invalid piano regression values')
    selected = np.zeros(values.shape,dtype=bool)
    shifts = np.zeros(values.shape,dtype=np.float64)
    if len(values) < 2*radius+1:
        return selected, shifts
    centers = np.arange(radius,len(values)-radius)
    mask = values[centers] > threshold
    for step in range(radius):
        mask &= values[centers-step] >= values[centers-step-1]
        mask &= values[centers+step] >= values[centers+step+1]
    selected[centers] = mask
    left, peak, right = values[centers-1], values[centers], values[centers+1]
    denominator = peak-np.where(left>right,right,left)
    estimate = np.divide((right-left)*.5,denominator,
                         out=np.zeros(denominator.shape,dtype=np.float64),where=denominator>0)
    shifts[centers] = np.where(mask,estimate,0.)
    return selected, shifts


def decode(activations, *, onset_threshold=.3, offset_threshold=.3, frame_threshold=.1):
    """Return physical candidates, distinguishing detected and forced endings.

    A reattack, a six-second decoder limit or the available input boundary must
    never be represented as a learned acoustic release without this provenance.
    """
    arrays = [np.asarray(activations[key]) for key in OUTPUTS]
    shape = arrays[0].shape
    if (len(shape) != 2 or shape[1] != 88
            or any(a.shape != shape or not np.isfinite(a).all() for a in arrays)):
        raise ValueError('Invalid piano activations')
    onset, onset_shift = regression_peaks(arrays[0],onset_threshold,2)
    release, release_shift = regression_peaks(arrays[1],offset_threshold,4)
    events = []
    last = shape[0]-1
    for key in range(88):
        attacks = np.flatnonzero(onset[:,key])
        fades = np.flatnonzero(arrays[2][:,key] <= frame_threshold)
        releases = np.flatnonzero(release[:,key])
        for index,start in enumerate(attacks):
            next_start = int(attacks[index+1]) if index+1 < len(attacks) else last+1
            boundary = min(int(start)+600,last,next_start-1)
            if boundary <= start:
                continue
            fade_index = np.searchsorted(fades,start+1)
            fade = int(fades[fade_index]) if fade_index < len(fades) else last+1
            shift = 0.
            if fade <= boundary:
                release_index = np.searchsorted(releases,start+1)
                first_release = int(releases[release_index]) if release_index < len(releases) else last+1
                if first_release <= fade and first_release-start > fade-first_release:
                    end,reason = first_release,'release_regression'
                else:
                    end,reason = fade,'frame_decay'
                shift = release_shift[end,key]
                # A low frame immediately after an onset is not a decay if
                # this pitch was never active. Keep the onset and raw endpoint
                # for review, but do not claim an observed sustain/release.
                if not np.any(arrays[2][start:fade,key] > frame_threshold):
                    reason = 'frame_inactive'
            elif next_start <= min(int(start)+600,last):
                end,reason = next_start-1,'reattack'
            else:
                end = boundary
                reason = 'duration_limit' if end-start >= 600 else 'input_boundary'
                shift = release_shift[end,key]
            events.append(dict(onset=float(np.float32((start+onset_shift[start,key])/FPS)),
                               offset=float(np.float32((end+shift)/FPS)),pitch=key+21,
                               velocity=int(arrays[3][start,key]*128),
                               onset_score=float(arrays[0][start,key]),
                               release_reason=reason,release_observed=reason in ('frame_decay','release_regression')))
    return events


class PianoDetector:
    """Optional, explicitly selected piano-only detector with independent caches."""
    def __init__(self,resources):
        self.model=Path(resources)/'assets/deep/piano-regression.onnx'

    def status(self):
        return dict(available=self.model.is_file(),label='钢琴专用采音（测试）',
                    voice='piano',default=False)

    def analyze(self,samples,sample_rate,cache,stem_identity,progress=lambda p,s:None,cancelled=lambda:False):
        from .domain import Note
        if not self.model.is_file():
            raise ValueError('当前安装包未包含钢琴专用采音模型')
        with self.model.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=MODEL_SHA256:
                raise ValueError('钢琴专用模型校验失败，请使用完整测试包')
        identity=dict(revision=2,model_sha256=MODEL_SHA256,stem_identity=stem_identity,
                      pcm_sha256=hashlib.sha256(np.asarray(samples,dtype='<f4').tobytes()).hexdigest(),
                      sample_rate=sample_rate,samples=len(samples))
        signature=json.dumps(identity,sort_keys=True)
        cache=Path(cache);activations=None;continuations=[]
        if cache.exists():
            try:
                with np.load(cache,allow_pickle=False) as data:
                    if str(data['identity'])==signature:
                        activations={key:data[key] for key in OUTPUTS}
                        continuations=json.loads(str(data['continuations']))
            except (ValueError,KeyError,OSError,EOFError):
                activations=None;continuations=[]
        cached=activations is not None
        if not cached:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            session=ort.InferenceSession(str(self.model),sess_options=options,providers=['CPUExecutionProvider'])
            activations=infer(samples,sample_rate,session,progress,cancelled,continuations)
            del session
            if cancelled():raise InterruptedError('已取消钢琴分析')
            temporary=cache.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,identity=signature,continuations=json.dumps(continuations),**activations)
            temporary.replace(cache)
        if cancelled():raise InterruptedError('已取消钢琴分析')
        events=decode(activations)
        notes=[Note(start=round(event['onset']*1000),end=round(event['offset']*1000),
                    pitch=event['pitch'],strength=event['onset_score'],source='pitch',voice='piano',
                    id=f'specialist:{i}',release_reason=event['release_reason']) for i,event in enumerate(events)]
        return notes,dict(identity=identity,cached=cached,engine='piano-onset-offset-regression',
                          window_context_continuations=continuations,
                          release_reasons={reason:sum(n.release_reason==reason for n in notes)
                              for reason in sorted({n.release_reason for n in notes})},
                          notice='钢琴持续区间包含踏板延音；强制截断的音尾不能自动作为 LN 放手。')
