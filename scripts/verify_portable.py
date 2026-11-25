"""Exercise the actual Windows EXE with an isolated data directory and clean PATH."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
from urllib.parse import urlencode
import urllib.request
import uuid
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab import __version__
from manialab.domain import parse_osu, validate
from manialab.timing import grid_report
from scripts.build_provenance import (
    sha256,
    verify_worker,
    verify_instrument_worker,
    verify_piano_model,
)
from scripts.package_assets import capabilities, verify_core_model_files


def verify_piano_onnx_runtime(model_path, identity):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [(item.name, item.shape) for item in session.get_inputs()] != [
        ("audio", [1, identity["input_samples"]])
    ]:
        raise AssertionError("Bundled piano model input contract changed")
    names = [item.name for item in session.get_outputs()]
    expected = [
        "reg_onset_output",
        "reg_offset_output",
        "frame_output",
        "velocity_output",
        "reg_pedal_onset_output",
        "reg_pedal_offset_output",
        "pedal_frame_output",
    ]
    if names != expected:
        raise AssertionError("Bundled piano model outputs changed")
    outputs = session.run(names, {"audio": np.zeros((1, identity["input_samples"]), dtype=np.float32)})
    shapes = [list(output.shape) for output in outputs]
    expected_shapes = [[1, 1001, 88]] * 4 + [[1, 1001, 1]] * 3
    if shapes != expected_shapes or not all(
        np.isfinite(output).all() for output in outputs
    ):
        raise AssertionError("Bundled piano model CPU smoke test failed")
    return dict(
        check="piano_onnx_runtime",
        sha256=identity["sha256"],
        bytes=identity["bytes"],
        providers=session.get_providers(),
        output_shapes=shapes,
        input="all-zero synthetic 10-second tensor; runtime smoke test only",
    )


def synthetic(path, seconds=30):
    rate = 22050
    windows = [(1,3,261.63),(5,9,329.63),(12,13,392.),(20,23,440.),(26,28,523.25)]
    data = bytearray(rate*seconds*2)
    for start,end,pitch in windows:
        for i in range(round(start*rate), min(round(end*rate),rate*seconds)):
            t=i/rate-start
            fade=min(1.,t/.005,(end-i/rate)/.005)
            value=.6*math.sin(2*math.pi*pitch*t)*fade
            struct.pack_into('<h',data,i*2,round(value*32767))
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(rate);stream.writeframes(data)


def verify_manifest_components(manifest, declared, bundle_root, source_root=ROOT):
    bundle_root = Path(bundle_root)
    worker = None
    piano_identity = None
    if declared["separation_four"] or declared["separation_six"]:
        worker_path = bundle_root / "_internal/assets/deep/ManiaLabAudio/ManiaLabAudio.exe"
        assert manifest["worker_sha256"] == hashlib.sha256(worker_path.read_bytes()).hexdigest()
        worker = verify_worker(source_root, bundle_root / "_internal/assets/deep")
        assert manifest["worker_sources"] == worker["worker_sources"]
        assert manifest["worker_files"] == worker["worker_files"]
    else:
        for field in ("worker_sha256", "worker_sources", "worker_files"):
            assert field in manifest and manifest[field] is None, (
                f"Core release must declare {field}=null"
            )
    if declared["piano"]:
        piano_identity = verify_piano_model(bundle_root / "_internal")
        assert manifest.get("piano_model") == piano_identity
    else:
        assert "piano_model" in manifest and manifest["piano_model"] is None, (
            "Unavailable piano_model must be explicitly null"
        )
    assert bool(manifest.get("guitar_model")) == declared["guitar"]
    assert bool(manifest.get("instrument_worker")) == declared["instruments"]
    if not declared["guitar"]:
        assert "guitar_model" in manifest and manifest["guitar_model"] is None, (
            "Unavailable guitar_model must be explicitly null"
        )
    elif manifest.get("guitar_model"):
        from scripts.build_provenance import verify_guitar_model
        assert verify_guitar_model(bundle_root / "_internal") == manifest["guitar_model"]
    if not declared["instruments"]:
        assert "instrument_worker" in manifest and manifest["instrument_worker"] is None, (
            "Unavailable instrument_worker must be explicitly null"
        )
    elif manifest.get("instrument_worker"):
        assert verify_instrument_worker(
            source_root, bundle_root / "_internal/assets/instruments"
        ) == manifest["instrument_worker"]
    return worker, piano_identity


def verify_torosu_cli(bundle_root, identity, work_root):
    bundle_root = Path(bundle_root)
    relative = "_internal/engines/torosu_cli/torosu-cli.exe"
    if identity.get("path") != relative:
        raise AssertionError("Torosu CLI path differs from the native runtime contract")
    binary = bundle_root / relative
    if not binary.is_file():
        raise AssertionError("Packaged Torosu CLI is missing")
    if identity.get("bytes") != binary.stat().st_size:
        raise AssertionError("Packaged Torosu CLI size differs from the release manifest")
    if identity.get("sha256") != hashlib.sha256(binary.read_bytes()).hexdigest():
        raise AssertionError("Packaged Torosu CLI hash differs from the release manifest")
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    smoke_root = work_root / "isolated"
    smoke_root.mkdir(exist_ok=True)
    environment = {
        key: value for key, value in os.environ.items()
        if key not in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
    }
    environment["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    environment["TEMP"] = str(smoke_root)
    environment["TMP"] = str(smoke_root)
    chart = """osu file format v14

[General]
AudioFilename: smoke.ogg
Mode: 3

[Metadata]
Title: Torosu CLI smoke
Artist: ManiaLab
Creator: ManiaLab
Version: Smoke
BeatmapSetID: 24

[Difficulty]
CircleSize:7

[TimingPoints]
0,500,4,1,0,70,1,0

[HitObjects]
36,192,0,1,8,0:0:0:0:
109,192,250,128,0,1000:0:0:0:0:
183,192,500,1,0,0:0:0:0:
256,192,750,1,0,0:0:0:0:
329,192,1000,1,0,0:0:0:0:
402,192,1250,1,0,0:0:0:0:
475,192,1500,1,0,0:0:0:0:
"""
    with tempfile.TemporaryDirectory(prefix="torosu-cli-smoke-", dir=smoke_root) as scratch:
        scratch = Path(scratch)
        source = scratch / "source.osu"
        output = scratch / "converted.osu"
        source.write_text(chart, encoding="utf-8")
        converted = subprocess.run(
            [
                str(binary), "convert", "--input", str(source), "--output", str(output),
                "--keys", "10", "--density", "none", "--strict",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, env=environment, cwd=scratch,
        )
        if converted.returncode:
            raise AssertionError(
                f"Packaged Torosu convert smoke failed: {converted.stderr[-2000:]}"
            )
        analysis = subprocess.run(
            [str(binary), "analyze", "--input", str(output), "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, env=environment, cwd=scratch,
        )
        if analysis.returncode:
            raise AssertionError(
                f"Packaged Torosu analyze smoke failed: {analysis.stderr[-2000:]}"
            )
        result = json.loads(analysis.stdout)
        if result.get("keys") != 10 or result.get("objects") != 7 or result.get("ln") != 1:
            raise AssertionError("Packaged Torosu CLI returned incorrect smoke analysis")
    return identity


def _required_bool(value, label):
    assert type(value) is bool, f"{label} must be a boolean"
    return value


def verify_runtime_capabilities(status, declared):
    separator = status["separator"]
    profiles = separator["profiles"]
    actual = {}
    expected_voices = {
        "four": ["drums", "bass", "vocals", "other"],
        "six": ["drums", "bass", "vocals", "other", "guitar", "piano"],
        "four-piano": ["drums", "piano", "vocals", "other"],
    }
    for profile, capability in (
        ("four", "separation_four"),
        ("six", "separation_six"),
        ("four-piano", "separation_six"),
    ):
        assert profile in profiles, f"Portable status omitted separator profile {profile}"
        actual[profile] = _required_bool(
            profiles[profile]["available"], f"Runtime {profile} availability"
        )
        assert profiles[profile].get("voices") == expected_voices[profile], (
            f"Runtime {profile} voices do not match the declared profile"
        )
        expected = _required_bool(declared[capability], f"Manifest {capability}")
        assert actual[profile] == expected, (
            f"Manifest declares {capability}={expected}, "
            f"but runtime reports {actual[profile]}"
        )
    separator_available = _required_bool(separator["available"], "Runtime separator availability")
    assert separator_available == _required_bool(
        declared["separation_four"], "Manifest separation_four"
    )
    for component in ("piano", "guitar", "instruments"):
        runtime_available = _required_bool(
            status[component]["available"], f"Runtime {component} availability"
        )
        expected = _required_bool(declared[component], f"Manifest {component}")
        assert runtime_available == expected, (
            f"Manifest declares {component}={expected}, "
            f"but runtime reports {runtime_available}"
        )
    for component, status_field, capability in (
        ("pattern_prior", "available", "pattern_prior"),
        ("engine", "install_available", "mapperatorinator_install"),
    ):
        runtime_available = _required_bool(
            status[component][status_field], f"Runtime {component} availability"
        )
        expected = _required_bool(declared[capability], f"Manifest {capability}")
        assert runtime_available == expected, (
            f"Manifest declares {capability}={expected}, "
            f"but runtime reports {runtime_available}"
        )
    assert _required_bool(status["piano"]["default"], "Runtime piano default") is False
    return actual


def build_verification_identity(executable, manifest_path, manifest, verification_run_id):
    return {
        "build_profile": manifest["build_profile"],
        "capabilities": manifest["capabilities"],
        "executable_sha256": sha256(executable),
        "release_manifest_sha256": sha256(manifest_path),
        "verification_run_id": verification_run_id,
    }


def append_verification_evidence(checks, identity, row):
    """Bind a freshly completed check, without re-signing previously collected rows."""
    if not isinstance(row, dict) or not isinstance(row.get("check"), str):
        raise ValueError("Verification evidence must be a named check row")
    if any(field in row and row[field] != value for field, value in identity.items()):
        raise ValueError("Verification evidence already belongs to another run")
    checks.append(dict(row, **deepcopy(identity)))


def bind_new_specialist_evidence(checks, first_new_index, identity):
    """Bind only specialist rows appended by the current check invocation."""
    if type(first_new_index) is not int or not 0 <= first_new_index <= len(checks):
        raise ValueError("Specialist evidence start index is invalid")
    specialist_checks = {
        "guitar_real_audio_http",
        "guitar_cold_cancel",
        "instrument_portable_real_audio",
    }
    for item in checks[first_new_index:]:
        if isinstance(item, dict) and item.get("check") in specialist_checks:
            if any(field in item and item[field] != value for field, value in identity.items()):
                raise ValueError("Specialist evidence already belongs to another run")
            item.update(deepcopy(identity))


def verify_analysis_stem_playback(request, base, state, voices):
    """Check the fresh analysis without substituting a selected candidate's source."""
    identity = state['analysis']['separation'].get('playback_identity')
    assert isinstance(identity, str) and identity, 'Current analysis has no playback source identity'
    context = dict(context='evidence', source_identity=identity, audio=state.get('audio') or '')
    for voice in voices:
        assert request(base + '/stem/' + voice + '?' + urlencode(context))[:4] == b'RIFF'
        query = urlencode(dict(context, voice=voice, start_ms=900, duration_ms=500))
        assert request(base + '/waveform?' + query)['sample_rate'] == 22050


def basic_analysis_evidence(analysis):
    return dict(check="basic_analysis", engine="basic",
                analysis_engine=analysis["engine"],
                candidates=len(analysis["candidates"]),
                seconds=analysis["elapsed_seconds"])


def verify_core_workflow(request, wait, start, stop, work, source, record):
    imported = request("/api/import", upload=("files", source.name, source.read_bytes()))
    base = "/api/projects/" + imported["id"]
    job = wait(request(base + "/analyze", dict(engine="basic", settings={})))
    assert job["status"] == "done", job
    state = request(base)
    analysis = state["analysis"]
    assert isinstance(analysis.get("candidates"), list), analysis
    record(basic_analysis_evidence(analysis))

    last_notes = None
    for keys in range(6, 11):
        payload = dict(mode="audio", bpm=120, offset=-22, division=4,
                       audio_shift_ms=22.5, timing_confirmed=True,
                       settings={"keys": keys, "ln_ratio": .6})
        job = wait(request(base + "/generate", payload))
        assert job["status"] == "done", job
        generated_state = request(base)
        parsed = parse_osu(request(base + "/export/osu").decode("utf-8"))
        report = validate(parsed["notes"], keys)
        assert report["valid"], report
        grid = grid_report(parsed["notes"], dict(bpm=120, offset=-22, division=4))
        assert grid["off_grid"] == 0, grid
        last_notes = generated_state["variants"][generated_state["active"]]["notes"]
        record(dict(check="generate_export", keys=keys,
                           notes=report["notes"], ln=report["ln"], off_grid=0))

    provided = request("/api/import", upload=("files", source.name, source.read_bytes()))
    provided_base = "/api/projects/" + provided["id"]
    upload = request(provided_base + "/provided-stems",
                     upload=("other", "合成其他声轨.wav", source.read_bytes()))
    job = wait(upload)
    assert job["status"] == "done", job
    job = wait(request(provided_base + "/analyze",
                       dict(engine="provided", settings={})))
    assert job["status"] == "done", job
    provided_state = request(provided_base)
    voice_names = sorted(voice["voice"] for voice in provided_state["analysis"]["voices"])
    assert voice_names == ["other"], voice_names
    generated = wait(request(provided_base + "/generate",
        dict(mode="audio", bpm=120, offset=-22, division=4, audio_shift_ms=22.5,
             timing_confirmed=True, settings={"keys": 10, "ln_ratio": .6})))
    assert generated["status"] == "done", generated
    provided_chart = parse_osu(
        request(provided_base + "/export/osu").decode("utf-8")
    )
    provided_report = validate(provided_chart["notes"], 10)
    assert provided_report["valid"], provided_report
    provided_grid = grid_report(
        provided_chart["notes"], dict(bpm=120, offset=-22, division=4)
    )
    assert provided_grid["off_grid"] == 0, provided_grid
    record(dict(check="provided_stem_analysis_export", voices=voice_names,
                       candidates=len(provided_state["analysis"]["candidates"]),
                       notes=provided_report["notes"], ln=provided_report["ln"],
                       off_grid=0))

    unavailable = {}
    for engine in ("deep", "deep6"):
        job = wait(request(base + "/analyze", dict(engine=engine, settings={})), timeout=30)
        assert job["status"] == "error", job
        assert "分离组件未就绪" in job["message"], job
        unavailable[engine] = job["status"]
    unavailable_status = request("/api/status")["separator"]["profiles"]
    assert unavailable_status["six"]["available"] is False
    assert unavailable_status["four-piano"]["available"] is False
    record(dict(check="unavailable_engines", engines=unavailable,
                       reason="bundled separation assets are not included"))

    stop()
    start()
    restored = request(base)
    restored_notes = restored["variants"][restored["active"]]["notes"]
    assert restored_notes == last_notes
    record(dict(check="restart", preserved=True, keys=10,
                       unicode_audio=source.name))

    bad = request("/api/import", upload=("files", "damaged.wav", b"invalid audio"))
    bad_base = "/api/projects/" + bad["id"]
    failure = wait(request(bad_base + "/analyze", dict(engine="basic", settings={})))
    assert failure["status"] == "error" and failure.get("message"), failure
    assert not request(bad_base).get("analysis")
    assert request("/api/status")["version"] == __version__
    record(dict(check="bad_audio_recovery", analysis=failure["status"],
                       service_alive=True))


def run(executable, work, piano_audio=None, instrument_fixtures=None, instruments_only=False,
        guitar_probe=None, guitar_only=False):
    executable, work = executable.resolve(), work.resolve()
    verification_run_id = uuid.uuid4().hex
    work.mkdir(parents=True,exist_ok=True)
    data=work/'独立项目';data.mkdir(exist_ok=True)
    source=work/'音频测试.wav';synthetic(source)
    environment={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')
                 and not k.startswith(('MANIALAB_DEEP_','MANIALAB_INSTRUMENT_'))}
    environment['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    environment.update(HTTP_PROXY='http://127.0.0.1:9',HTTPS_PROXY='http://127.0.0.1:9',NO_PROXY='127.0.0.1,localhost',PYTHONUTF8='1')
    process=None;log=None;url='';token='';checks=[];launch_number=0
    local_http=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def record(row):
        append_verification_evidence(checks,verification_identity,row)

    def start():
        nonlocal process,log,url,token,launch_number
        marker=data/'current-url.txt';marker.unlink(missing_ok=True)
        log=(work/f'launch-{launch_number}.log').open('x',encoding='utf-8')
        launch_number+=1
        process=subprocess.Popen([str(executable),'--no-browser','--data-dir',str(data)],cwd=work,
            env=environment,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+60
        while not marker.exists():
            if process.poll() is not None:raise RuntimeError('Packaged process exited; inspect launch log')
            if time.monotonic()>deadline:raise TimeoutError('Packaged startup exceeded 60s')
            time.sleep(.2)
        url=marker.read_text(encoding='utf-8').strip()
        while True:
            if process.poll() is not None:raise RuntimeError('Packaged process exited before HTTP readiness; inspect launch log')
            try:
                html=local_http.open(url,timeout=2).read().decode()
                break
            except (urllib.error.URLError,TimeoutError):
                if time.monotonic()>deadline:raise TimeoutError('Packaged HTTP service did not become ready')
                time.sleep(.2)
        token=re.search(r'name="manialab-token" content="([^"]+)"',html).group(1)

    def stop():
        if process is not None and process.poll() is None:
            process.terminate();process.wait(timeout=15)
        if log:log.close()

    def request(path,payload=None,upload=None):
        headers={};body=None
        if upload:
            field,name,content=upload;boundary='ManiaLabPortableVerificationBoundary'
            body=(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+content+f'\r\n--{boundary}--\r\n'.encode())
            headers['Content-Type']='multipart/form-data; boundary='+boundary
        elif payload is not None:
            body=json.dumps(payload).encode();headers['Content-Type']='application/json'
        if body is not None:headers['X-ManiaLab-Token']=token
        with local_http.open(urllib.request.Request(url+path,data=body,headers=headers),timeout=30) as response:
            raw=response.read()
            return json.loads(raw) if 'application/json' in response.headers.get('Content-Type','') else raw

    def wait(job,timeout=240):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            result=request('/api/jobs/'+job['id'])
            if result['status'] in ('done','error','cancelled'):return result
            time.sleep(.25)
        raise TimeoutError('Packaged job exceeded timeout')

    try:
        manifest_path=executable.parent/'release-manifest.json'
        release_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert release_manifest['version']==__version__
        assert release_manifest['executable_sha256']==hashlib.sha256(executable.read_bytes()).hexdigest()
        torosu_identity=release_manifest.get('torosu_cli')
        has_torosu_sources='engines/torosu_cli/Cargo.toml' in release_manifest['sources']
        if has_torosu_sources:
            assert torosu_identity is not None, 'Native Torosu source has no packaged CLI identity'
        for name,digest in release_manifest['sources'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Packaged source is stale: {name}'
        declared=capabilities(release_manifest['build_profile'])
        assert release_manifest['capabilities']==declared
        worker,piano_identity=verify_manifest_components(
            release_manifest,declared,executable.parent,ROOT
        )
        core_model_identity=verify_core_model_files(
            executable.parent/'_internal/assets'
        )
        assert release_manifest.get('core_models')==core_model_identity, (
            'Basic Pitch identity does not match the release manifest'
        )
        is_core=release_manifest['build_profile']=='core'
        verification_identity=build_verification_identity(
            executable,manifest_path,release_manifest,verification_run_id
        )
        record(dict(check='release_manifest',source_files=len(release_manifest['sources']),
            worker_files=0 if worker is None else len(worker['worker_files']),
            **verification_identity))
        record(dict(check='core_model',**core_model_identity))
        prior_path=executable.parent/'_internal/assets/pattern-prior.json'
        if declared['pattern_prior']:
            expected_prior=hashlib.sha256((ROOT/'assets/pattern-prior.json').read_bytes()).hexdigest()
            bundled_prior=hashlib.sha256(prior_path.read_bytes()).hexdigest()
            assert bundled_prior==expected_prior, 'Packaged layout model differs from the source release'
            record(dict(check='bundled_model',sha256=bundled_prior))
        else:
            assert not prior_path.exists(), 'Core profile must exclude the learned layout prior'
        expected_profiles=hashlib.sha256((ROOT/'assets/arrangement-profiles.json').read_bytes()).hexdigest()
        assert hashlib.sha256((executable.parent/'_internal/assets/arrangement-profiles.json').read_bytes()).hexdigest()==expected_profiles
        record(dict(check='bundled_arrangement_profiles',sha256=expected_profiles))
        if has_torosu_sources:
            torosu_smoke=verify_torosu_cli(
                executable.parent,torosu_identity,work/'torosu-cli-smoke'
            )
            record(dict(check='torosu_cli_smoke',
                path=torosu_smoke['path'],bytes=torosu_smoke['bytes'],
                sha256=torosu_smoke['sha256'],commands=['convert','analyze']))
        start();status=request('/api/status')
        assert status['version']==__version__,status
        separator_status=verify_runtime_capabilities(status,declared)
        if declared["piano"]:
            record(verify_piano_onnx_runtime(
                executable.parent/'_internal/assets/deep/piano-regression.onnx',
                piano_identity,
            ))
        record(dict(check='startup',version=status['version'],separator=status['separator'],
                           prior=status['pattern_prior'],engine=status['engine']))
        if is_core:
            if instruments_only or guitar_only or piano_audio is not None \
                    or instrument_fixtures is not None or guitar_probe is not None:
                raise ValueError("Core portable verification does not accept specialist-engine fixtures")
            record(dict(check='core_capability_status',separator=separator_status,
                               piano_available=status['piano']['available']))
            verify_core_workflow(request,wait,start,stop,work,source,record)
            print(json.dumps(dict(passed=True,scope='core',checks=checks),ensure_ascii=True),flush=True)
            return
        if guitar_only:
            if guitar_probe is None:
                raise ValueError('--guitar-only requires --guitar-probe')
            from scripts.verify_portable_guitar import check_guitar
            first_specialist_check=len(checks)
            check_guitar(request,wait,start,stop,work,guitar_probe,checks)
            bind_new_specialist_evidence(checks,first_specialist_check,verification_identity)
            print(json.dumps(dict(passed=True,scope='guitar_only',checks=checks),ensure_ascii=True),flush=True)
            return
        if instruments_only:
            if instrument_fixtures is None:
                raise ValueError('--instruments-only requires --instrument-fixtures')
            from scripts.verify_portable_instruments import check_instruments
            first_specialist_check=len(checks)
            check_instruments(request,wait,start,stop,work,instrument_fixtures,checks)
            bind_new_specialist_evidence(checks,first_specialist_check,verification_identity)
            print(json.dumps(dict(passed=True,scope='instruments_only',checks=checks),ensure_ascii=True),flush=True)
            return
        state=request('/api/import',upload=('files',source.name,source.read_bytes()));pid=state['id'];base='/api/projects/'+pid
        job=wait(request(base+'/analyze',dict(engine='deep',settings={},audio_shift_ms=22.5)))
        assert job['status']=='done',job
        state=request(base);raw=state['analysis']['candidates']
        assert len(raw)==5,(len(raw),raw)
        expected=[(1000,3000),(5000,9000),(12000,13000),(20000,23000),(26000,28000)]
        assert all(abs(n['start']-a)<=20 and abs(n['end']-b)<=20 for n,(a,b) in zip(raw,expected))
        record(dict(check='deep_analysis',seconds=state['analysis']['elapsed_seconds'],candidates=len(raw),engine=state['analysis']['engine']))
        for keys in range(6,11):
            payload=dict(mode='audio',bpm=120,offset=-22,division=4,audio_shift_ms=22.5,timing_confirmed=True,settings={'keys':keys,'ln_ratio':.6})
            job=wait(request(base+'/generate',payload));assert job['status']=='done',job
            text=request(base+'/export/osu').decode('utf-8');parsed=parse_osu(text)
            report=validate(parsed['notes'],keys)
            # Every isolated source event is sustained. The section policy
            # must preserve all five; a global 55% quota used to turn two taps.
            assert report['valid'] and report['notes']==5 and report['ln']==5,report
            grid=grid_report(parsed['notes'],dict(bpm=120,offset=-22,division=4));assert grid['off_grid']==0,grid
            record(dict(check='generate_export',keys=keys,notes=5,ln=5,off_grid=0))
        with zipfile.ZipFile(io.BytesIO(request(base+'/export/osz'))) as archive:
            assert set(archive.namelist())=={'ManiaLab.osu',source.name},archive.namelist()
        audio=request(base+'/audition',dict(mode='chart',start_ms=0,duration_ms=12000));assert audio[:4]==b'RIFF'
        window=request(base+'/waveform?start_ms=900&duration_ms=500');assert window['sample_rate']==22050
        evidence=json.dumps(request(base)['analysis'],sort_keys=True)
        proposals=request(base+'/recommendations',dict(settings={'keys':10},bpm=120))
        assert len(proposals['options'])==3 and proposals['reference']['available']
        for option in proposals['options']:
            chosen=request(base+'/apply-recommendation',dict(settings={'keys':10},bpm=120,choice=option['id'],
                evidence_signature=proposals['evidence_signature'],context_signature=proposals['context_signature']))
            payload=dict(mode='audio',bpm=120,offset=-22,division=4,audio_shift_ms=22.5,timing_confirmed=True,settings=chosen['settings'])
            job=wait(request(base+'/generate',payload));assert job['status']=='done',job
            parsed=parse_osu(request(base+'/export/osu').decode('utf-8'))
            assert validate(parsed['notes'],10)['valid']
            assert grid_report(parsed['notes'],dict(bpm=120,offset=-22,division=4))['off_grid']==0
            assert json.dumps(request(base)['analysis'],sort_keys=True)==evidence
            record(dict(check='local_recommendation',choice=option['id'],notes=len(parsed['notes']),evidence_preserved=True))
        stale=wait(request(base+'/generate',payload|dict(bpm=240)))
        assert stale['status']=='error' and 'BPM' in stale['message'],stale
        record(dict(check='stale_recommendation_rejected',reason='BPM changed'))
        state=request(base);saved_notes=state['variants'][state['active']]['notes']
        stop();start();restored=request(base)
        assert restored['variants'][restored['active']]['notes']==saved_notes
        assert restored['plan']['choice']=='layers'
        record(dict(check='restart',preserved=True,keys=10,unicode_audio=source.name))
        # Packaged status must expose the piano-focused four-stem profile;
        # it shares the already bundled six-source model and runtime.
        four_piano_status = status['separator']['profiles']['four-piano']
        assert four_piano_status['available'], status['separator']
        assert four_piano_status['voices'] == ['drums','piano','vocals','other']
        record(dict(check='four_piano_profile_status',
                    available=four_piano_status['available'],voices=four_piano_status['voices']))
        # Both packaged separators must run with no Python or model downloads.
        assert status['separator']['profiles']['six']['available'],status['separator']
        job=wait(request(base+'/analyze',dict(engine='deep6',settings={})),timeout=360)
        assert job['status']=='done',job
        state=request(base)
        assert state['analysis']['separation']['profile']=='six'
        assert {v['voice'] for v in state['analysis']['voices']}=={'drums','bass','vocals','other','guitar','piano'}
        verify_analysis_stem_playback(request,base,state,('guitar','piano'))
        job=wait(request(base+'/generate',dict(mode='audio',bpm=120,offset=-22,division=4,
            audio_shift_ms=22.5,timing_confirmed=True,settings={'keys':10,'guitar_weight':0})))
        assert job['status']=='done',job
        state=request(base);candidate=state['variants'][state['active']]
        assert candidate['report']['grid']['off_grid']==0 and all(n['voice']!='guitar' for n in candidate['notes'])
        record(dict(check='six_stem_analysis_export',seconds=state['analysis']['elapsed_seconds'],
            voices=[v['voice'] for v in state['analysis']['voices']],notes=len(candidate['notes']),off_grid=0))
        if piano_audio is not None:
            assert status['piano']['available'] and not status['piano']['default'],status['piano']
            piano_audio=piano_audio.resolve();content=piano_audio.read_bytes()
            imported=request('/api/import',upload=('files','真实钢琴.wav',content))
            piano_base='/api/projects/'+imported['id']
            job=wait(request(piano_base+'/provided-stems',upload=('piano','钢琴声轨.wav',content)))
            assert job['status']=='done',job
            analysis_payload=dict(engine='provided',piano_engine='specialist',settings={})
            job=wait(request(piano_base+'/analyze',analysis_payload),timeout=600)
            assert job['status']=='done',job
            piano_state=request(piano_base);original=piano_state['analysis']['candidates']
            assert original and all(n['voice']=='piano' for n in original)
            summary=piano_state['analysis']['voices'][0]['release_refinement']
            assert summary['engine']=='piano-onset-offset-regression' and not summary['cached'],summary
            record(dict(check='piano_cold_inference',input_sha256=hashlib.sha256(content).hexdigest(),
                model_sha256=summary['identity']['model_sha256'],candidates=len(original),
                seconds=piano_state['analysis']['elapsed_seconds'],release_reasons=summary['release_reasons']))
            plan=piano_state['expression']
            generated={}
            for mode in ('articulation','sustain'):
                for section in plan['sections']:
                    section['voices']['piano']['mode']=mode
                request(piano_base+'/expression',dict(expression=plan))
                job=wait(request(piano_base+'/generate',dict(mode='audio',bpm=120,offset=0,division=4,
                    audio_shift_ms=0,timing_confirmed=True,settings={'keys':10,'density':12,'ln_ratio':1})))
                assert job['status']=='done',job
                piano_state=request(piano_base)
                assert piano_state['analysis']['candidates']==original
                assert piano_state['expression']['source']=='manual'
                assert all(s['voices']['piano']['mode']==mode for s in piano_state['expression']['sections'])
                exported=request(piano_base+'/export/osu')
                parsed=parse_osu(exported.decode('utf-8'));report=validate(parsed['notes'],10)
                assert report['valid'] and report['notes']>0,report
                assert grid_report(parsed['notes'],dict(bpm=120,offset=0,division=4))['off_grid']==0
                (work/f'piano-{mode}.osu').write_bytes(exported)
                generated[mode]=dict(notes=report['notes'],ln=report['ln'])
            record(dict(check='piano_manual_expression',modes=generated,evidence_preserved=True,
                timing='fixed synthetic grid for export verification, not musical timing acceptance'))
            saved_piano=piano_state['variants'][piano_state['active']]['notes']
            stop();start();restored=request(piano_base)
            assert restored['analysis']['candidates']==original
            assert restored['variants'][restored['active']]['notes']==saved_piano
            assert restored['expression']==piano_state['expression']
            job=wait(request(piano_base+'/analyze',analysis_payload),timeout=600)
            assert job['status']=='done',job
            warmed=request(piano_base)
            assert warmed['analysis']['voices'][0]['release_refinement']['cached']
            assert warmed['analysis']['candidates']==original
            record(dict(check='piano_restart_warm_cache',candidates=len(original),identical=True,
                seconds=warmed['analysis']['elapsed_seconds']))
        if instrument_fixtures is not None:
            from scripts.verify_portable_instruments import check_instruments
            first_specialist_check=len(checks)
            check_instruments(request,wait,start,stop,work,instrument_fixtures,checks)
            bind_new_specialist_evidence(checks,first_specialist_check,verification_identity)
        if guitar_probe is not None:
            from scripts.verify_portable_guitar import check_guitar
            first_specialist_check=len(checks)
            check_guitar(request,wait,start,stop,work,guitar_probe,checks)
            bind_new_specialist_evidence(checks,first_specialist_check,verification_identity)
        # Cancel a fresh separation, then confirm the same server remains usable.
        second=request('/api/import',upload=('files','取消测试.wav',source.read_bytes()+b'\x00\x00'))
        second_base='/api/projects/'+second['id'];job=request(second_base+'/analyze',dict(engine='deep',settings={}))
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            running=request('/api/jobs/'+job['id'])
            if running['status']=='running' and running['progress']>=2:break
            time.sleep(.1)
        request('/api/jobs/'+job['id']+'/cancel',{})
        cancelled=wait(job,30);assert cancelled['status']=='cancelled',cancelled
        assert not request(second_base).get('analysis')
        bad=request('/api/import',upload=('files','bad.wav',b'invalid audio'))
        failure=wait(request('/api/projects/'+bad['id']+'/analyze',dict(engine='basic',settings={})))
        assert failure['status']=='error'
        assert request('/api/status')['version']==__version__
        record(dict(check='cancel_and_failure',cancelled=cancelled['status'],bad_audio=failure['status'],service_alive=True))
    finally:
        stop()
        (work/'report.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('--work',type=Path,default=ROOT/'test-data/portable-alpha4')
    parser.add_argument('--piano-audio',type=Path,help='Short real piano WAV for packaged specialist cold/warm and expression checks')
    parser.add_argument('--instrument-fixtures',type=Path,help='Frozen real bass/other fixtures for packaged HTTP checks')
    parser.add_argument('--instruments-only',action='store_true',help='Only the packaged instrument checks, after component identity checks')
    parser.add_argument('--guitar-probe',type=Path,help='Completed fixed FL float probe, including both native stems')
    parser.add_argument('--guitar-only',action='store_true',help='Only guitar HTTP checks, after component identity checks')
    args=parser.parse_args()
    if args.guitar_only and args.instruments_only:
        parser.error('Choose at most one focused check')
    run(args.executable,args.work,args.piano_audio,args.instrument_fixtures,args.instruments_only,args.guitar_probe,args.guitar_only)
