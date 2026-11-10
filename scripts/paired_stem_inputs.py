"""Verify cached source identity and retain absolute time for cropped audio."""
import json
from pathlib import Path
import numpy as np

from manialab.stem_release_evidence import ROLES, observe, EVIDENCE_REVISION
from scripts.probe_mapper_adapter import sha


def prepare_observations(record, output):
    audio = Path(record['audio']); chart = Path(record['chart'])
    if sha(audio) != record['audio_sha256'] or sha(chart) != record['chart_sha256']:
        raise ValueError('Original audio/chart hash changed')
    analysis_path = audio.parent/'deep/analysis.json'
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    separation = analysis['separation']; clip = analysis.get('research_clip')
    if clip:
        if (clip['original_audio_sha256'] != record['audio_sha256']
                or clip['chart_sha256'] != record['chart_sha256']):
            raise ValueError('Cropped audio identity does not match its original chart/audio')
        start = float(clip['start_ms']); margin = max(3000., float(clip['excluded_edge_ms']))
        if abs(float(clip['end_ms'])-start-analysis['duration']*1000) > 1:
            raise ValueError('Cropped duration does not match its absolute clock')
    else:
        if separation['audio_sha256'] != record['audio_sha256']:
            raise ValueError('Separated audio identity changed')
        start = 0.; margin = 0.
    stem_root = analysis_path.parent/'stems'
    own = json.loads((stem_root/'separation.json').read_text(encoding='utf-8'))
    for name in ('audio_sha256', 'model_sha256', 'sample_rate', 'frames'):
        if own[name] != separation[name]: raise ValueError('Cached separation metadata disagrees')
    if separation['sample_rate'] != 22050: raise ValueError('Unexpected stem sample rate')
    evidence = {}; identities = {}
    for role in ROLES:
        path = stem_root/(role+'.npy')
        y = np.load(path, mmap_mode='r', allow_pickle=False)
        if y.shape != (separation['frames'],): raise ValueError('Stem clocks differ')
        if abs(len(y)/22050-analysis['duration']) > .001: raise ValueError('Stem duration differs from analysis')
        identities[role] = sha(path)
        evidence[role] = observe(y, 22050)
    output.mkdir(parents=True)
    saved = output/'observations.npz'
    np.savez_compressed(saved, **{role+'_'+key:value for role,row in evidence.items() for key,value in row.items()})
    identity = dict(revision=EVIDENCE_REVISION, audio_sha256=record['audio_sha256'],
        chart_sha256=record['chart_sha256'], analysis_sha256=sha(analysis_path),
        separator_sha256=separation['model_sha256'], raw_stem_sha256=identities,
        observations_sha256=sha(saved), audio_start_ms=start,
        supported_bounds=[start+margin, start+analysis['duration']*1000-margin],
        source_kind='cached_four_stem_hybrid_demucs', normalisation='original_npy_PCM_scale')
    (output/'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    return evidence, identity


def load_observations(path, expected):
    identity = json.loads((path/'identity.json').read_text(encoding='utf-8'))
    if identity != expected or sha(path/'observations.npz') != identity['observations_sha256']:
        raise ValueError('Frozen audio observations changed')
    with np.load(path/'observations.npz', allow_pickle=False) as saved:
        evidence = {role:{key:saved[role+'_'+key] for key in ('times_ms', 'level', 'flux', 'low', 'high')}
                    for role in ROLES}
    return evidence
