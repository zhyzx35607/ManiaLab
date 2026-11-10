"""Actual HTTP/EXE tests for the optional instrument path; no mock inference."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

from manialab.domain import UNOBSERVED_RELEASES, parse_osu, validate
from manialab.timing import grid_report


def check_instruments(request, wait, start, stop, work, fixtures, checks):
    status = request('/api/status')['instruments']
    assert status['available'] and status['default'] is False and status['predicted_releases_default'] is False, status
    fixtures = Path(fixtures)
    manifest = json.loads((fixtures/'manifest.json').read_text(encoding='utf-8'))
    sources = {c['model']: c for c in manifest['cases']}
    for voice, model in (('bass','bass_v2'),('other','other_v1_5')):
        source = sources[model]
        audio = Path(source['audio']).read_bytes()
        assert hashlib.sha256(audio).hexdigest() == source['audio_sha256']
        project = request('/api/import',upload=('files','乐器原声-'+voice+'.wav',audio))
        base = '/api/projects/'+project['id']
        imported = wait(request(base+'/provided-stems',upload=(voice,'声轨-'+voice+'.wav',audio)))
        assert imported['status'] == 'done', imported
        mode_results = []
        for allow in (False,True):
            options = dict(engine='provided',instrument_engine='tsumugi',allow_predicted_releases=allow,
                settings=dict(keys=10,density=24,chord_size=10,min_ln=100))
            job = wait(request(base+'/analyze',options),timeout=600)
            assert job['status'] == 'done', job
            assert job['identity']['instrument_engine'] == 'tsumugi' and job['identity']['allow_predicted_releases'] is allow
            state = request(base)
            analysis = state['analysis']
            raw = analysis['candidates']
            assert raw and all(n['voice']==voice and 'tsumugi:' in n['instrument_id'] for n in raw)
            assert analysis['instrument_engine']=='tsumugi' and analysis['allow_predicted_releases'] is allow
            summary = analysis['voices'][0]
            detector = summary['release_refinement']
            assert summary['detector']=='tsumugi' and summary['cross_source_bleed_checked'] is False
            assert detector['cached'] is False and detector['source_semantics']=='unit_weight_not_confidence'
            assert detector['generic_release_refinement'] is False and detector['generic_attack_fusion'] is False
            if not allow:
                assert all(n['release_reason'] in UNOBSERVED_RELEASES for n in raw)
            assert request(base+'/stem/'+voice)[:4]==b'RIFF'
            waveform = request(base+'/waveform?voice='+voice+'&start_ms=0&duration_ms=1000')
            assert waveform['sample_rate']==22050
            plan=state['expression']
            for section in plan['sections']:
                section['voices'][voice]['mode']='sustain'
            request(base+'/expression',dict(expression=plan))
            generated=[]
            for keys in range(6,11):
                job=wait(request(base+'/generate',dict(mode='audio',bpm=120,offset=0,division=4,
                    audio_shift_ms=0,timing_confirmed=True,settings=options['settings']|dict(keys=keys))),timeout=240)
                assert job['status']=='done',job
                state=request(base)
                assert state['analysis']['candidates']==raw
                exported=request(base+'/export/osu')
                parsed=parse_osu(exported.decode('utf-8'))
                structural=validate(parsed['notes'],keys)
                assert parsed['keys']==keys and structural['valid'] and structural['notes']>0,structural
                assert grid_report(parsed['notes'],dict(bpm=120,offset=0,division=4))['off_grid']==0
                selected=state['variants'][state['active']]['notes']
                assert [(n.start,n.end,n.lane) for n in parsed['notes']]==[
                    (n['start'],n['end'],n['lane']) for n in selected], 'Export changed selected notes'
                assert all(n['start']==n['end'] or n['release_reason'] not in UNOBSERVED_RELEASES for n in selected)
                if not allow:
                    assert structural['ln']==0,structural
                else:
                    assert structural['ln']>0,structural
                (work/f'{voice}-release-{allow}-{keys}k.osu').write_bytes(exported)
                generated.append(dict(keys=keys,notes=structural['notes'],ln=structural['ln']))
            archive_bytes=request(base+'/export/osz')
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                names=archive.namelist()
                chart_names=[p for p in names if p.endswith('.osu')]
                assert len(names)==2 and len(chart_names)==1
                assert archive.read(chart_names[0])==exported
                audio_name=next(p for p in names if p!=chart_names[0])
                assert hashlib.sha256(archive.read(audio_name)).hexdigest()==source['audio_sha256']
            (work/f'{voice}-release-{allow}.osz').write_bytes(archive_bytes)
            audition=request(base+'/audition',dict(mode='chart',start_ms=0,duration_ms=5000))
            assert audition[:4]==b'RIFF'
            notes_before=state['variants'][state['active']]['notes']
            stop();start()
            restored=request(base)
            assert restored['analysis']['candidates']==raw
            assert restored['analysis_options']==state['analysis_options']
            assert restored['variants'][restored['active']]['notes']==notes_before
            job=wait(request(base+'/analyze',options),timeout=600)
            assert job['status']=='done',job
            warmed=request(base)
            assert warmed['analysis']['candidates']==raw
            assert warmed['analysis']['voices'][0]['release_refinement']['cached'] is True
            mode_results.append(dict(allow_predicted_releases=allow,candidates=len(raw),
                cold_seconds=analysis['elapsed_seconds'],warm_seconds=warmed['analysis']['elapsed_seconds'],
                generated=generated,cache_reused_after_restart=True,source_preserved=True))
        checks.append(dict(check='instrument_portable_real_audio',voice=voice,model=model,
            fixture_sha256=source['audio_sha256'],modes=mode_results,
            scope='Clean PATH and unreachable download proxies (not a firewall), real worker through HTTP, 6-10K export and restart cache',
            musical_acceptance=False,timing='Fixed verification grid; not evaluated musical timing'))
        print(json.dumps(dict(stage='instrument_http_passed',voice=voice,modes=mode_results)),flush=True)
