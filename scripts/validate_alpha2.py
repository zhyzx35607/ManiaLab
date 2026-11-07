"""Reproducible timing benchmark and locally available beatmap invariants.

No user music is uploaded. Synthetic audio is deliberately a controlled timing
test, not a measurement of real-song transcription or chart quality.
"""
from pathlib import Path
import json
import subprocess
import sys
import time
import types

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.audio import rhythmic_evidence, SR
from manialab.arrange import assign_lanes
from manialab.domain import parse_osu, export_osu, invariant, validate


def run():
    baseline = types.ModuleType('manialab._baseline')
    baseline.__package__ = 'manialab'
    source = subprocess.check_output(['git', 'show', '5dacd9bc4e1427e83b871f57cc103de3afad3381:manialab/audio.py'], cwd=ROOT)
    exec(compile(source, 'alpha1_audio.py', 'exec'), baseline.__dict__)
    rows = []
    for bpm in (90, 120, 137, 174.5, 273.25):
        y = np.zeros(SR * 180, np.float32)
        expected = np.arange(.347, 179, 60 / bpm) * 1000
        t = np.arange(500) / SR
        click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 120)).astype(np.float32)
        for at in expected:
            i = round(at / 1000 * SR)
            y[i:i + len(click)] += click
        _, old = baseline.rhythmic_evidence(y)
        _, new = rhythmic_evidence(y)
        period = 60000 / new['bpm']
        errors = (expected - new['offset'] + period / 2) % period - period / 2
        rows.append(dict(true_bpm=bpm, alpha1_bpm=old['bpm'], alpha2_bpm=new['bpm'],
                         alpha2_max_beat_error_ms=round(float(np.max(np.abs(errors))), 3),
                         alpha2_offset=new['offset'], true_phase_ms=round(347 % (60000 / bpm), 3),
                         stable=new['stable']))
        assert abs(new['bpm'] - bpm) < .01 and np.max(np.abs(errors)) < 5
    maps = []
    for path in sorted((ROOT / 'tests/artifacts/public-maps').glob('*.osu')):
        source = path.read_text(encoding='utf-8-sig')
        parsed = parse_osu(source)
        start = time.perf_counter()
        out = assign_lanes(parsed['notes'], {'keys': parsed['keys']})
        exported = parse_osu(export_osu(out, parsed['keys'], ROOT / 'vendor/mapperatorinator/template.osu', original=source))
        preserved = invariant(out) == invariant(parsed['notes'])
        assert preserved and exported['timing'] == parsed['timing'] and validate(out, parsed['keys'])['valid']
        assert sorted(n.raw.split(',', 1)[1] for n in exported['notes']) == sorted(n.raw.split(',', 1)[1] for n in parsed['notes'])
        maps.append(dict(id=path.stem, notes=len(out), ln=sum(n.hold for n in out),
                         seconds=round(time.perf_counter() - start, 3), preserved=preserved))
    result = dict(timing=rows, maps=maps, note='Controlled timing/structure tests; no real-song listening or gameplay quality claim.')
    (ROOT / 'docs/alpha2-benchmark.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
