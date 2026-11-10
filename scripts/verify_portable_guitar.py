"""Real guitar HTTP checks shared by source and packaged application runs.

The fixed float probe supplies raw model evidence, not authored chart targets.
All audio and generated charts remain in the caller's local verification folder.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from manialab.audio import decode, source_evidence, SR
from manialab.domain import Note, UNOBSERVED_RELEASES, parse_osu, sections, validate, values
from manialab.guitar import MODEL_SHA256
from manialab.timing import grid_report

ROOT = Path(__file__).resolve().parents[1]
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fixtures(probe, work):
    """Freeze the first complete recording and both native stems, without NN."""
    probe, work = Path(probe), Path(work)
    report = read(probe/'report.json')
    assert report['derived_cases'] == 60 and len(report['derived_rows']) == 60
    assert report['native_unlabelled_cases'] == 2 and len(report['native_diagnostics']) == 2
    assert report['plan']['graph_sha256'] == MODEL_SHA256
    assert report['plan']['code_sha256']['manialab/guitar.py'] == sha(ROOT/'manialab/guitar.py')
    rows = [report['derived_rows'][0], *report['native_diagnostics']]
    assert [r['name'] for r in rows] == [
        'guitar-00_BN1-129-Eb_comp', 'sakura-nova-guitar', 'youdiantian-htdemucs6-guitar']
    mixtures = [None, ROOT/'test-data/sakura-benchmark/bundled/mix.mp3',
        Path('C:/Users/31270/Downloads/audio.mp3')]
    mix_hashes = [None, '2a82d46c43e43acb4f8e21de71573be1ae6b71b8cf5485364c9ed1d5d3b12765',
        '08b6aff5db874fca7718c6bab0b7c97e01b59648531bfd0c53d5b1628346010d']
    result = []
    for index, row in enumerate(rows):
        folder = probe/('derived-cohort' if index == 0 else 'native-diagnostics')/row['name']
        for name, digest in row['artifacts'].items():
            assert sha(folder/name) == digest, (row['name'], name)
        pcm = np.load(folder/'input-22050.npy', allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.ndim == 1 and np.isfinite(pcm).all()
        target = work/row['name']
        target.mkdir(parents=True, exist_ok=False)
        stem = target/'guitar-float.wav'
        wavfile.write(stem, SR, pcm)
        mixture = mixtures[index] or stem
        assert index == 0 or sha(mixture) == mix_hashes[index]
        mix = decode(mixture, target)
        assert len(mix) == len(pcm), 'Do not trim or pad the frozen fixtures'
        assert np.array_equal(decode(stem, target), pcm), 'Float WAV import changed the original stem'
        result.append(dict(name=row['name'], source=row, folder=folder, directory=target,
            stem=stem, mixture=mixture, pcm=pcm, mix=mix,
            expected=read(folder/'runtime-candidates.json'), modes=(False, True) if index == 0 else (False,)))
    return result


def check_guitar(request, wait, start, stop, work, probe, checks):
    status = request('/api/status')['guitar']
    assert status['available'] and status['default'] is False and status['predicted_releases_default'] is False
    prepared = fixtures(probe, Path(work)/'guitar-fixtures')
    for fixture in prepared:
        audio = fixture['mixture'].read_bytes()
        imported = request('/api/import', upload=('files', fixture['mixture'].name, audio))
        base = '/api/projects/'+imported['id']
        job = wait(request(base+'/provided-stems', upload=('guitar', '吉他浮点声轨.wav', fixture['stem'].read_bytes())))
        assert job['status'] == 'done', job
        modes = []
        for allow in fixture['modes']:
            options = dict(engine='provided', guitar_engine='fl', allow_guitar_predicted_releases=allow,
                settings=dict(keys=10, density=24, chord_size=10, min_ln=100, ln_ratio=1.))
            job = wait(request(base+'/analyze', options), timeout=900)
            assert job['status'] == 'done', job
            assert job['identity']['guitar_engine'] == 'fl'
            assert job['identity']['allow_guitar_predicted_releases'] is allow
            state = request(base)
            analysis = state['analysis']
            raw = analysis['candidates']
            assert analysis['guitar_engine'] == 'fl' and analysis['allow_guitar_predicted_releases'] is allow
            assert analysis['instrument_engine'] == analysis['piano_engine'] == 'general'
            summary = analysis['voices'][0]
            evidence = summary['release_refinement']
            assert summary['detector'] == 'guitar-fl' and summary['cross_source_bleed_checked'] is False
            assert evidence['cached'] is False
            assert evidence['generic_release_refinement'] is False and evidence['generic_attack_fusion'] is False
            assert evidence['identity']['pcm_sha256'] == hashlib.sha256(fixture['pcm'].astype('<f4').tobytes()).hexdigest()
            stored = Path(evidence['evidence_cache']).parents[2]/'guitar.npy'
            assert np.array_equal(np.load(stored, allow_pickle=False), fixture['pcm'])
            expected = fixture['expected']['predicted_releases' if allow else 'default_onsets']
            # The research scorer adds three labels outside the actual decoder.
            # Strip only those annotations before comparing every raw event field.
            expected_trace = [dict(item, raw_event={k: v for k, v in item['raw_event'].items()
                if k not in ('program', 'is_drum', 'voice')}) for item in expected['trace']]
            assert evidence['candidate_trace'] == expected_trace, 'HTTP detector changed frozen raw model evidence'
            predicted = [Note(**note) for note in expected['candidates']]
            retained, rejected = source_evidence(predicted, fixture['pcm'], fixture['mix'])
            assert raw == [n.data() for n in sorted(retained, key=lambda n: (n.start, n.pitch, n.voice))]
            assert summary['rejected_low_energy_pitch'] == rejected
            assert evidence['source_energy_filter']['rejected_ids'] == sorted({n.id for n in predicted}-{n.id for n in retained})
            if not allow:
                assert all(n['release_reason'] in UNOBSERVED_RELEASES for n in raw)
            assert request(base+'/stem/guitar')[:4] == b'RIFF'
            assert request(base+'/waveform?voice=guitar&start_ms=0&duration_ms=1000')['sample_rate'] == SR
            plan = state['expression']
            generated = []
            # The native weak stem may contain no accepted notes. Do not invent
            # notes or choose a louder excerpt just to satisfy an export check.
            if raw:
                for section in plan['sections']:
                    section['voices']['guitar']['mode'] = 'sustain'
                request(base+'/expression', dict(expression=plan))
                for keys in range(6, 11):
                    job = wait(request(base+'/generate', dict(mode='audio', bpm=120, offset=0, division=4,
                        audio_shift_ms=0, timing_confirmed=True, settings=options['settings']|dict(keys=keys))), timeout=240)
                    assert job['status'] == 'done', job
                    state = request(base)
                    assert state['analysis']['candidates'] == raw
                    exported = request(base+'/export/osu')
                    parsed = parse_osu(exported.decode('utf-8'))
                    bookmarks = values(sections(exported.decode('utf-8'))['Editor']).get('Bookmarks', '')
                    assert all(int(t) >= 0 for t in bookmarks.split(',') if t.strip()), bookmarks
                    structural = validate(parsed['notes'], keys)
                    assert parsed['keys'] == keys and structural['valid'] and structural['notes'] > 0, structural
                    assert grid_report(parsed['notes'], dict(bpm=120, offset=0, division=4))['off_grid'] == 0
                    selected = state['variants'][state['active']]['notes']
                    assert [(n.start, n.end, n.lane) for n in parsed['notes']] == [
                        (n['start'], n['end'], n['lane']) for n in selected]
                    assert all(n['start'] == n['end'] or n['release_reason'] not in UNOBSERVED_RELEASES for n in selected)
                    if not allow:
                        assert structural['ln'] == 0
                    (fixture['directory']/f'release-{allow}-{keys}k.osu').write_bytes(exported)
                    generated.append(dict(keys=keys, notes=structural['notes'], ln=structural['ln']))
                archive_bytes = request(base+'/export/osz')
                with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                    names = archive.namelist()
                    chart = next(name for name in names if name.endswith('.osu'))
                    assert len(names) == 2 and archive.read(chart) == exported
                    assert hashlib.sha256(archive.read(next(n for n in names if n != chart))).hexdigest() == hashlib.sha256(audio).hexdigest()
                (fixture['directory']/f'release-{allow}.osz').write_bytes(archive_bytes)
                assert request(base+'/audition', dict(mode='chart', start_ms=0, duration_ms=5000))[:4] == b'RIFF'
            if fixture['name'].startswith('guitar-00_'):
                assert generated and (not allow or all(row['ln'] > 0 for row in generated))
            before = request(base)
            stop(); start()
            restored = request(base)
            assert restored['analysis'] == before['analysis']
            assert restored['analysis_options'] == before['analysis_options']
            assert restored['expression'] == before['expression'] and restored['variants'] == before['variants']
            job = wait(request(base+'/analyze', options), timeout=900)
            assert job['status'] == 'done', job
            warm = request(base)['analysis']
            assert warm['candidates'] == raw and warm['voices'][0]['release_refinement']['cached'] is True
            assert warm['voices'][0]['release_refinement']['candidate_trace'] == evidence['candidate_trace']
            result = dict(allow_predicted_releases=allow, raw_model_events=len(predicted), retained=len(raw),
                rejected_low_energy=rejected, source_pcm_equal=True, raw_trace_equal=True,
                cache_reused_after_restart=True, cold_seconds=analysis['elapsed_seconds'],
                warm_seconds=warm['elapsed_seconds'], generated=generated)
            modes.append(result)
            (fixture['directory']/f'analysis-release-{allow}.json').write_text(
                json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(stage='guitar_http_mode', case=fixture['name'], **result)), flush=True)
        checks.append(dict(check='guitar_real_audio_http', case=fixture['name'], modes=modes,
            fixture_pcm_sha256=hashlib.sha256(fixture['pcm'].astype('<f4').tobytes()).hexdigest(),
            musical_acceptance=False, timing='Fixed verification grid, not musical timing acceptance'))
    # Cancel a genuinely cold request, then check that no partial analysis was saved.
    fixture = prepared[0]
    imported = request('/api/import', upload=('files', '取消吉他采音.wav', fixture['stem'].read_bytes()))
    base = '/api/projects/'+imported['id']
    assert wait(request(base+'/provided-stems', upload=('guitar', 'guitar.wav', fixture['stem'].read_bytes())))['status'] == 'done'
    job = request(base+'/analyze', dict(engine='provided', guitar_engine='fl'))
    deadline = time.monotonic()+60
    while True:
        current = request('/api/jobs/'+job['id'])
        assert current['status'] in ('queued', 'running'), current
        if current['status'] == 'running' and current['progress'] >= 31:
            break
        assert time.monotonic() < deadline
        time.sleep(.05)
    request('/api/jobs/'+job['id']+'/cancel', {})
    assert wait(job, timeout=60)['status'] == 'cancelled'
    assert not request(base).get('analysis')
    assert request('/api/status')['guitar']['available']
    checks.append(dict(check='guitar_cold_cancel', partial_analysis_saved=False, service_alive=True))
