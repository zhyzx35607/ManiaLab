"""Fixed development inputs and physical-clock labels for piano adaptation.

This module performs no model execution and never opens reserved test files.
MIDI key endpoints, pedal sustain and future chart LN decisions stay separate.
"""
import hashlib
import json
from math import isfinite
from numbers import Integral
from pathlib import Path
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SELECTION_SHA256 = '34947943dcf11daf1b68939d005e05fd7331fe40413e575e70140a88248eb1aa'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _midi_labels(path, duration):
    # Keep optional MIDI dependencies out of target/window arithmetic imports.
    from scripts.prepare_piano_benchmark import midi_events
    from scripts.piano_pedal_labels import sustain_labels
    notes, controls = midi_events(path)
    if any(not 21 <= n['pitch'] <= 108 or not 0 <= n['onset'] < n['offset'] <= duration for n in notes):
        raise ValueError('Development MIDI has an invalid pitch or physical interval')
    sustained, pedal_audit = sustain_labels(notes, controls, duration)
    return notes, controls, sustained, pedal_audit


def load_development(root):
    """Return the frozen ten development records in selection order.

    Each dict contains absolute Path audio/midi, their SHA256 identities, notes,
    controls, sustain_notes and pedal_audit. The caller supplies mido in its
    research environment. Reserved test metadata is skipped before file access.
    """
    root = Path(root).resolve()
    selection_path = root/'selection.json'
    if _sha(selection_path) != SELECTION_SHA256:
        raise ValueError('Frozen MAESTRO development selection changed')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    downloads = json.loads((root/'download-report.json').read_text(encoding='utf-8'))
    if downloads['selection_sha256'] != SELECTION_SHA256:
        raise ValueError('Downloads refer to a different selection')
    selected = [row for row in selection['rows'] if row['split'] in ('train', 'validation')]
    if (len(selected) != 10 or sum(row['split']=='train' for row in selected) != 6
            or sum(row['split']=='validation' for row in selected) != 4):
        raise ValueError('Expected exactly six train and four validation recordings')
    by_id = {row['id']:row for row in selected}
    if len(by_id) != len(selected):
        raise ValueError('Duplicate development recording identity')
    if len(downloads['files']) != 20:
        raise ValueError('Expected exactly twenty development files')
    # Validate the complete roster before opening any advertised file. In
    # particular, a test item anywhere in the report stops before all hashes.
    entries = {}
    for item in downloads['files']:
        split, recording_id, kind = item['split'], item['id'], item['kind']
        if split not in ('train', 'validation') or recording_id not in by_id:
            raise ValueError('Only selected development files may be accessed')
        row = by_id[recording_id]
        if split != row['split'] or kind not in ('audio', 'midi'):
            raise ValueError('Development split or file kind mismatch')
        identity = (recording_id,kind)
        if identity in entries:
            raise ValueError('Duplicate development file')
        path = Path(item['path']).resolve()
        if not path.is_relative_to(root/'development'/split):
            raise ValueError('Development file escaped its split directory')
        member = row['members'][kind]
        if item['member'] != member or item['size'] != member['size']:
            raise ValueError('Download member differs from frozen selection')
        if not item.get('zip_crc_verified'):
            raise ValueError('Download archive CRC was not verified')
        entries[identity] = (path,item)
    if set(entries) != {(row['id'],kind) for row in selected for kind in ('audio','midi')}:
        raise ValueError('Incomplete development file roster')
    for path,item in entries.values():
        if path.stat().st_size != item['size'] or _sha(path) != item['sha256']:
            raise ValueError('Development source file identity changed: '+path.name)
    result = []
    for row in selected:
        audio,audio_id = entries[row['id'],'audio']
        midi,midi_id = entries[row['id'],'midi']
        with wave.open(str(audio),'rb') as stream:
            if (stream.getframerate(),stream.getnchannels(),stream.getsampwidth(),stream.getcomptype()) != (44100,2,2,'NONE'):
                raise ValueError('Expected original 44100 Hz stereo int16 PCM')
            duration = stream.getnframes()/stream.getframerate()
        if duration < 180:
            raise ValueError('Development recording is shorter than 180 seconds')
        notes,controls,sustained,pedal_audit = _midi_labels(midi,duration)
        result.append(dict(id=row['id'],split=row['split'],year=row['year'],
            audio=audio,midi=midi,audio_sha256=audio_id['sha256'],midi_sha256=midi_id['sha256'],
            audio_duration=duration,selection_sha256=SELECTION_SHA256,
            notes=notes,controls=controls,sustain_notes=sustained,pedal_audit=pedal_audit))
    return result


def read_pcm(path, seconds=180):
    """Read the original prefix as 16k float32 mono, without gain normalization."""
    if isinstance(seconds,bool) or not isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive finite input duration required')
    rate,raw = wavfile.read(path,mmap=True)
    if rate != 44100 or raw.dtype != np.int16 or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError('Expected original 44100 Hz stereo int16 PCM')
    samples = round(seconds*rate)
    if len(raw) < samples:
        raise ValueError('Source is shorter than the fixed input duration')
    source = raw[:samples].astype(np.float32).mean(axis=1)/32768
    result = resample_poly(source,160,441).astype(np.float32)
    if len(result) != round(seconds*16000) or not np.isfinite(result).all():
        raise ValueError('Input duration does not produce the expected 16k clock')
    return result


def onset_targets(notes, start, frames=1001, fps=100):
    """Sample max triangular evidence around true MIDI onsets on the frame clock.

    Notes just outside the window contribute when within 50ms of a frame.
    Input timestamps are never rounded to frame centers or beat positions.
    """
    if (isinstance(start,bool) or not isfinite(start) or start < 0
            or isinstance(frames,bool) or not isinstance(frames,Integral) or frames < 1
            or isinstance(fps,bool) or not isinstance(fps,Integral) or fps <= 0):
        raise ValueError('Invalid physical frame clock')
    times = start+np.arange(frames,dtype=np.float64)/fps
    target = np.zeros((frames,88),dtype=np.float32)
    for note in notes:
        pitch,onset = note['pitch'],note['onset']
        if (isinstance(pitch,bool) or not isinstance(pitch,Integral) or not 21 <= pitch <= 108
                or isinstance(onset,bool) or not isfinite(onset) or onset < 0):
            raise ValueError('Invalid MIDI pitch or onset')
        if onset < times[0]-.05 or onset > times[-1]+.05:
            continue
        values = np.maximum(0.,1.-np.abs(times-onset)/.05).astype(np.float32)
        np.maximum(target[:,pitch-21],values,out=target[:,pitch-21])
    return target


def window_slices(count=35):
    """Unique frame ownership for ten-second windows stepped by five seconds.

    For 35 windows, their physical starts are 0,5,...,170 seconds; returned
    half-open slices cover exactly the 18000 frames in [0,180), once each.
    """
    if isinstance(count,bool) or not isinstance(count,Integral) or count < 1:
        raise ValueError('Positive integer window count required')
    if count == 1:
        return [(0,1000)]
    return [(0 if index==0 else 250,1000 if index==count-1 else 750) for index in range(count)]


def stitch(windows):
    """Stitch [window,1001,...] arrays, lists thereof, or a dict of such values.

    Returns [unique_frame,...], preserving dtype and trailing dimensions.
    A dict returns a dict with matching keys and equal window/frame counts.
    """
    if isinstance(windows,dict):
        if not windows:
            raise ValueError('No window arrays supplied')
        lengths = {len(value) for value in windows.values()}
        if len(lengths) != 1:
            raise ValueError('Output arrays have different window counts')
        return {key:stitch(value) for key,value in windows.items()}
    arrays = np.asarray(windows)
    if arrays.ndim < 2 or arrays.shape[1] != 1001:
        raise ValueError('Expected [window,1001,...] frame arrays')
    slices = window_slices(len(arrays))
    return np.concatenate([arrays[index,left:right] for index,(left,right) in enumerate(slices)],axis=0)
