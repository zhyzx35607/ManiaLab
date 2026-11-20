import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from manialab import __version__
from scripts import pytest_provenance
from scripts.collect_release_validation import collect
from scripts.package_assets import CORE_FILES, capabilities, verify_core_model_files

ROOT = Path(__file__).resolve().parents[1]


def write_pytest_evidence(path, junit_report):
    snapshot = pytest_provenance.source_snapshot(ROOT)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    evidence = {
        "schema": pytest_provenance.SOURCE_SCHEMA,
        "source_commit": commit,
        "argv": ["-m", "pytest", "-q", "--junitxml", str(junit_report)],
        "junit_sha256": hashlib.sha256(junit_report.read_bytes()).hexdigest(),
        "source_digest_before": snapshot["digest"],
        "source_digest_after": snapshot["digest"],
        "exit_code": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "file_count": snapshot["files"],
    }
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path, evidence


def make_instrument_worker_identity():
    from manialab.instrument_bundle import (
        MODELS, MODEL_REVISION, SOURCE_COMMIT, runtime_identity, source_manifest,
    )

    return {
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "source_files": source_manifest()["source_files"],
        "models": MODELS,
        "license": "MIT",
        "license_sha256": "f" * 64,
        "voices": ["bass", "other"],
        "prepared_only": False,
        "portable_runtime_verified": True,
        "worker_sources": runtime_identity(ROOT),
        "worker_sha256": "2" * 64,
        "worker_files": {"ManiaLabInstrument.exe": {"bytes": 1, "sha256": "2" * 64}},
        "verified_models": ["bass_v2", "other_v1_5"],
        "verification_sha256": "4" * 64,
    }


def make_release_fixture(root, profile='standard'):
    build_root = root / "build"
    executable = build_root / "dist" / "ManiaLab" / "ManiaLab.exe"
    package = build_root / "release"
    executable.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    executable.write_bytes(b"portable-executable")
    manifest = {
        "version": __version__,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "sources": {},
        "build_profile": profile,
        "capabilities": capabilities(profile),
        "core_models": verify_core_model_files(ROOT / 'assets'),
        "piano_model": {"sha256": "expected-model-sha", "bytes": 123} if profile != 'core' else None,
        "worker_sha256": "expected-worker-sha" if profile != 'core' else None,
        "worker_sources": {} if profile != 'core' else None,
        "worker_files": [] if profile != 'core' else None,
        "guitar_model": json.loads((ROOT / 'assets/guitar-model.json').read_text(encoding='utf-8'))
            if profile == 'extended' else None,
        "instrument_worker": make_instrument_worker_identity() if profile == 'extended' else None,
    }
    (executable.parent / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    from scripts.package_assets import archive_stem
    archive_path = package / f"{archive_stem(__version__, profile)}.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(executable, "ManiaLab/ManiaLab.exe")
        archive.writestr(
            "ManiaLab/release-manifest.json",
            json.dumps(manifest),
        )
        for name in CORE_FILES:
            archive.write(ROOT / 'assets' / name, 'ManiaLab/_internal/assets/' + name)
    reports = root / "reports"
    reports.mkdir()
    pytest_report = reports / "junit.xml"
    pytest_report.write_text(
        '<testsuites tests="3" failures="0" errors="0" skipped="1">'
        '<testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    pytest_evidence, _ = write_pytest_evidence(
        reports / (pytest_report.name + ".provenance.json"), pytest_report
    )
    portable_report = reports / "portable.json"
    verification_identity = {
        "build_profile": profile,
        "capabilities": manifest["capabilities"],
        "executable_sha256": manifest["executable_sha256"],
        "release_manifest_sha256": hashlib.sha256(
            (executable.parent / "release-manifest.json").read_bytes()).hexdigest(),
        "verification_run_id": "c" * 32,
    }
    portable_checks = [
        {"check": "release_manifest", **verification_identity},
        {"check": "startup"},
        {"check": "restart"},
    ]
    if profile == 'core':
        portable_checks += [
            {"check": "basic_analysis"},
            {"check": "provided_stem_analysis_export"},
            {"check": "bad_audio_recovery"},
        ]
    else:
        portable_checks += [
            {
                "check": "piano_onnx_runtime",
                "sha256": "expected-model-sha",
                "bytes": 123,
                "providers": ["CPUExecutionProvider"],
            },
            {"check": "deep_analysis"},
            {"check": "six_stem_analysis_export"},
            {"check": "cancel_and_failure"},
        ]
    if profile == 'extended':
        guitar_cases = (
            ('guitar-00_BN1-129-Eb_comp', (False, True)),
            ('sakura-nova-guitar', (False,)),
            ('youdiantian-htdemucs6-guitar', (False,)),
        )
        for case, modes in guitar_cases:
            portable_checks.append({
                "check": "guitar_real_audio_http",
                **verification_identity,
                "case": case,
                "fixture_pcm_sha256": "a" * 64,
                "modes": [
                    {
                        "allow_predicted_releases": allow,
                        "source_pcm_equal": True,
                        "raw_trace_equal": True,
                        "cache_reused_after_restart": True,
                    }
                    for allow in modes
                ],
            })
        portable_checks.append({
            "check": "guitar_cold_cancel",
            **verification_identity,
            "partial_analysis_saved": False,
            "service_alive": True,
        })
        for voice, model in (("bass", "bass_v2"), ("other", "other_v1_5")):
            portable_checks.append({
                "check": "instrument_portable_real_audio",
                **verification_identity,
                "voice": voice,
                "model": model,
                "fixture_sha256": "b" * 64,
                "modes": [
                    {
                        "allow_predicted_releases": allow,
                        "cache_reused_after_restart": True,
                        "source_preserved": True,
                    }
                    for allow in (False, True)
                ],
            })
    portable_report.write_text(json.dumps(portable_checks), encoding="utf-8")
    shutdown_report = reports / "shutdown.json"
    shutdown = {
        "passed": True,
        "version": manifest["version"],
        "build_profile": profile,
        "capabilities": manifest["capabilities"],
        "executable_sha256": manifest["executable_sha256"],
        "release_manifest_sha256": hashlib.sha256(
            (executable.parent / "release-manifest.json").read_bytes()).hexdigest(),
        "worker_expected": profile != 'core',
        "application_exited": True,
        "worker_stopped_on_parent_exit": profile != 'core',
    }
    if profile == 'core':
        shutdown.update(worker_processes_before_exit=[], worker_processes_after_exit=[])
    shutdown_report.write_text(
        json.dumps(shutdown), encoding="utf-8"
    )
    return build_root, pytest_report, portable_report, shutdown_report, archive_path, pytest_evidence


def test_collect_release_validation_writes_hashes_and_sanitized_summary(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(tmp_path)
    report, output, checksums = collect(
        build_root, pytest_report, portable_report, shutdown_report,
        pytest_evidence=pytest_evidence,
    )

    assert report["version"] == __version__
    assert report["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert report["verification"]["pytest"]["tests"] == 3
    assert report["verification"]["pytest"]["skipped"] == 1
    assert report["verification"]["shutdown"]["passed"] is True
    assert report["verification"]["piano_model"]["cpu_smoke_passed"] is True
    assert report["limitations"]["musical_acceptance"] is False
    assert "song.wav" not in output.read_text(encoding="utf-8")
    assert archive.name in checksums.read_text(encoding="utf-8")


def test_collector_discovers_the_adjacent_required_pytest_sidecar(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    report, _, _ = collect(build_root, pytest_report, portable_report, shutdown_report)
    assert pytest_evidence == pytest_report.with_name(pytest_report.name + ".provenance.json")
    assert report["verification"]["pytest"]["evidence_schema"] == pytest_provenance.SOURCE_SCHEMA


def test_collect_core_validation_requires_core_inventory_and_runtime_checks(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(
        tmp_path, profile='core'
    )
    report, output, checksums = collect(build_root, pytest_report, portable_report, shutdown_report,
        pytest_evidence=pytest_evidence)
    assert archive.name.endswith('-core.zip')
    assert report['build_profile'] == 'core'
    assert report['verification']['piano_model'] is None
    assert report['limitations']['separation_weights_bundled'] is False
    assert report['limitations']['musical_acceptance'] is False
    assert report['verification']['shutdown']['worker_expected'] is False
    assert report['verification']['shutdown']['application_exited'] is True
    assert output.name.endswith('-core-validation.json')
    assert archive.name in checksums.read_text(encoding='utf-8')


def test_collect_core_validation_rejects_declared_deep_components(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(
        tmp_path, profile='core'
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['worker_sha256'] = 'unexpected-worker'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    for report_path in (portable_report, shutdown_report):
        document = json.loads(report_path.read_text(encoding='utf-8'))
        identity = document[0] if isinstance(document, list) else document
        identity['release_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(document), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('ManiaLab/ManiaLab.exe', (build_root / 'dist/ManiaLab/ManiaLab.exe').read_bytes())
        bundle.writestr('ManiaLab/release-manifest.json', json.dumps(manifest))
        for name in CORE_FILES:
            bundle.write(ROOT / 'assets' / name, 'ManiaLab/_internal/assets/' + name)

    with pytest.raises(SystemExit, match='Core manifest declares'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_core_validation_rejects_changed_basic_pitch_bytes(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(
        tmp_path, profile='core'
    )
    manifest = json.loads(
        (build_root / 'dist' / 'ManiaLab' / 'release-manifest.json').read_text(encoding='utf-8')
    )
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.write(build_root / 'dist/ManiaLab/ManiaLab.exe', 'ManiaLab/ManiaLab.exe')
        bundle.writestr('ManiaLab/release-manifest.json', json.dumps(manifest))
        for name in CORE_FILES:
            asset = ROOT / 'assets' / name
            if name == 'basic-pitch.onnx':
                bundle.writestr('ManiaLab/_internal/assets/' + name, b'changed model')
            else:
                bundle.write(asset, 'ManiaLab/_internal/assets/' + name)

    with pytest.raises(SystemExit, match='Basic Pitch model checksum'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_extended_validation_requires_specialist_runtime_evidence(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    report, _, _ = collect(
        build_root, pytest_report, portable_report, shutdown_report,
        pytest_evidence=pytest_evidence,
    )
    assert report['build_profile'] == 'extended'
    assert report['capabilities']['guitar'] is True
    assert report['capabilities']['instruments'] is True
    manifest = json.loads((build_root / 'dist' / 'ManiaLab' / 'release-manifest.json').read_text(
        encoding='utf-8'
    ))
    assert manifest['guitar_model']['filename'] == 'guitar-fl.onnx'
    assert manifest['instrument_worker']['portable_runtime_verified'] is True

    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    checks = [item for item in checks if item.get('case') != 'sakura-nova-guitar']
    portable_report.write_text(json.dumps(checks), encoding='utf-8')
    with pytest.raises(SystemExit, match='guitar real-audio cases'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


SPECIALIST_REPORT_ROWS = (
    ('guitar_real_audio_http', 'case', 'sakura-nova-guitar'),
    ('guitar_cold_cancel', None, None),
    ('instrument_portable_real_audio', 'voice', 'other'),
)
VERIFICATION_IDENTITY_FIELDS = (
    'build_profile', 'capabilities', 'executable_sha256',
    'release_manifest_sha256', 'verification_run_id',
)


def _portable_row(checks, check_name, selector_field, selector_value):
    return next(item for item in checks if item.get('check') == check_name
        and (selector_field is None or item.get(selector_field) == selector_value))


@pytest.mark.parametrize(('check_name', 'selector_field', 'selector_value'), SPECIALIST_REPORT_ROWS)
@pytest.mark.parametrize('field', VERIFICATION_IDENTITY_FIELDS)
def test_collect_extended_validation_rejects_missing_specialist_identity_fields(
    tmp_path, check_name, selector_field, selector_value, field
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = _portable_row(checks, check_name, selector_field, selector_value)
    del record[field]
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='identity|verification_run_id'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(('check_name', 'selector_field', 'selector_value'), SPECIALIST_REPORT_ROWS)
@pytest.mark.parametrize('field', VERIFICATION_IDENTITY_FIELDS)
def test_collect_extended_validation_rejects_mismatched_specialist_identity_fields(
    tmp_path, check_name, selector_field, selector_value, field
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = _portable_row(checks, check_name, selector_field, selector_value)
    mismatches = {
        'build_profile': 'standard',
        'capabilities': capabilities('standard'),
        'executable_sha256': 'a' * 64,
        'release_manifest_sha256': 'b' * 64,
        'verification_run_id': 'd' * 32,
    }
    record[field] = mismatches[field]
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='identity|verification_run_id'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_extended_validation_rejects_current_manifest_with_stale_specialist_row(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    current_manifest = _portable_row(checks, 'release_manifest', None, None)
    stale_row = _portable_row(checks, 'guitar_real_audio_http', 'case', 'sakura-nova-guitar')
    assert current_manifest['verification_run_id'] == 'c' * 32
    stale_row['verification_run_id'] = 'd' * 32
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='specialist evidence identity differs'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize('run_id', ('invalid-run-id', 'C' * 32))
def test_collect_extended_validation_rejects_invalid_manifest_verification_run_id(tmp_path, run_id):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    manifest_row = _portable_row(checks, 'release_manifest', None, None)
    manifest_row['verification_run_id'] = run_id
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='invalid verification_run_id'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_extended_validation_rejects_missing_instrument_voice(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    checks = [item for item in checks if item.get('voice') != 'other']
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='instrument real-audio voices'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(
    ('field', 'identity', 'message'),
    [
        ('guitar_model', None, 'guitar_model identity'),
        ('guitar_model', {}, 'guitar_model identity'),
        ('guitar_model', {'filename': 'guitar-fl.onnx', 'sha256': 'invalid'}, 'guitar_model identity'),
        ('instrument_worker', None, 'instrument_worker identity'),
        ('instrument_worker', {}, 'instrument_worker identity'),
        ('instrument_worker', {'worker_sha256': 'invalid'}, 'instrument_worker identity'),
    ],
)
def test_collect_extended_validation_rejects_missing_or_invalid_specialist_identity(
    tmp_path, field, identity, message
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest[field] = identity
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(SystemExit, match=message):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize('field', ('guitar_model', 'instrument_worker'))
def test_collect_extended_validation_rejects_absent_specialist_identity_field(tmp_path, field):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    del manifest[field]
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(SystemExit, match='identity'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize('field', ('guitar_model', 'instrument_worker'))
def test_collect_profile_without_specialist_capability_rejects_component_identity(tmp_path, field):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='standard'
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest[field] = json.loads((ROOT / 'assets/guitar-model.json').read_text(encoding='utf-8')) \
        if field == 'guitar_model' else make_instrument_worker_identity()
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(SystemExit, match='without .* capability'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(
    ('check_name', 'identity_field', 'identity_value', 'message'),
    [
        ('guitar_real_audio_http', 'case', 'guitar-00_BN1-129-Eb_comp', 'guitar real-audio cases'),
        ('instrument_portable_real_audio', 'voice', 'bass', 'instrument real-audio voices'),
    ],
)
def test_collect_extended_validation_rejects_duplicate_specialist_records(
    tmp_path, check_name, identity_field, identity_value, message
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = next(item for item in checks if item.get('check') == check_name
        and item.get(identity_field) == identity_value)
    checks.append(record.copy())
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match=message):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(
    ('check_name', 'selector_field', 'selector_value', 'invalid_selector', 'message'),
    [
        ('guitar_real_audio_http', 'case', 'sakura-nova-guitar', [], 'invalid guitar real-audio case'),
        ('guitar_real_audio_http', 'case', 'sakura-nova-guitar', {}, 'invalid guitar real-audio case'),
        ('instrument_portable_real_audio', 'voice', 'bass', [], 'invalid instrument real-audio voice'),
        ('instrument_portable_real_audio', 'voice', 'bass', {}, 'invalid instrument real-audio voice'),
    ],
)
def test_collect_extended_validation_rejects_unhashable_specialist_selectors(
    tmp_path, check_name, selector_field, selector_value, invalid_selector, message
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = next(item for item in checks if item.get('check') == check_name
        and item.get(selector_field) == selector_value)
    record[selector_field] = invalid_selector
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match=message):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize('mode_flags', ([False], [False, False], [False, 'true']))
def test_collect_extended_validation_rejects_incomplete_or_invalid_modes(tmp_path, mode_flags):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = next(item for item in checks if item.get('check') == 'instrument_portable_real_audio'
        and item.get('voice') == 'bass')
    template = record['modes'][0]
    record['modes'] = [dict(template, allow_predicted_releases=flag) for flag in mode_flags]
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Instrument real-audio evidence is incomplete for voice bass'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(
    ('check_name', 'identity_field', 'identity_value', 'flag', 'message'),
    [
        ('guitar_real_audio_http', 'case', 'guitar-00_BN1-129-Eb_comp', 'source_pcm_equal',
            'Guitar real-audio evidence is incomplete'),
        ('guitar_real_audio_http', 'case', 'guitar-00_BN1-129-Eb_comp', 'raw_trace_equal',
            'Guitar real-audio evidence is incomplete'),
        ('guitar_real_audio_http', 'case', 'guitar-00_BN1-129-Eb_comp', 'cache_reused_after_restart',
            'Guitar real-audio evidence is incomplete'),
        ('instrument_portable_real_audio', 'voice', 'bass', 'cache_reused_after_restart',
            'Instrument real-audio evidence is incomplete'),
        ('instrument_portable_real_audio', 'voice', 'bass', 'source_preserved',
            'Instrument real-audio evidence is incomplete'),
    ],
)
def test_collect_extended_validation_rejects_failed_specialist_flags(
    tmp_path, check_name, identity_field, identity_value, flag, message
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = next(item for item in checks if item.get('check') == check_name
        and item.get(identity_field) == identity_value)
    record['modes'][0][flag] = False
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match=message):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(('field', 'value'), [
    ('partial_analysis_saved', True),
    ('service_alive', False),
])
def test_collect_extended_validation_rejects_failed_guitar_cancel_check(tmp_path, field, value):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    cancel = next(item for item in checks if item.get('check') == 'guitar_cold_cancel')
    cancel[field] = value
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='guitar cold-cancel check'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(
    ('check_name', 'identity_field', 'identity_value', 'hash_field'),
    [
        ('guitar_real_audio_http', 'case', 'guitar-00_BN1-129-Eb_comp', 'fixture_pcm_sha256'),
        ('instrument_portable_real_audio', 'voice', 'bass', 'fixture_sha256'),
    ],
)
def test_collect_extended_validation_rejects_invalid_fixture_sha256(
    tmp_path, check_name, identity_field, identity_value, hash_field
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='extended'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = next(item for item in checks if item.get('check') == check_name
        and item.get(identity_field) == identity_value)
    record[hash_field] = 'not-a-sha256'
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='evidence is incomplete'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize('profile', ('core', 'standard'))
def test_collect_profiles_without_specialist_capabilities_do_not_require_specialist_evidence(
    tmp_path, profile
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile=profile
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    assert not any(item.get('check') in {
        'guitar_real_audio_http', 'guitar_cold_cancel', 'instrument_portable_real_audio'
    } for item in checks)

    report, _, _ = collect(build_root, pytest_report, portable_report, shutdown_report,
        pytest_evidence=pytest_evidence)
    assert report['build_profile'] == profile


@pytest.mark.parametrize(('profile', 'check_name'), [
    ('core', 'guitar_real_audio_http'),
    ('core', 'guitar_cold_cancel'),
    ('core', 'instrument_portable_real_audio'),
    ('standard', 'guitar_real_audio_http'),
    ('standard', 'guitar_cold_cancel'),
    ('standard', 'instrument_portable_real_audio'),
])
def test_collect_profiles_without_specialist_capabilities_reject_specialist_rows(
        tmp_path, profile, check_name
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile=profile
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    checks.append({'check': check_name})
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='without the declared profile capability'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_missing_portable_checks(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    portable_report.write_text(json.dumps([{"check": "release_manifest"}]), encoding="utf-8")

    with pytest.raises(SystemExit, match="missing required checks"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_piano_smoke_for_other_model(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    checks = json.loads(portable_report.read_text(encoding="utf-8"))
    piano_check = next(item for item in checks if item["check"] == "piano_onnx_runtime")
    piano_check["sha256"] = "different-model-sha"
    portable_report.write_text(json.dumps(checks), encoding="utf-8")

    with pytest.raises(SystemExit, match="Piano runtime smoke does not match"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_missing_pytest_evidence(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    pytest_evidence.unlink()

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report)


def test_collect_release_validation_rejects_legacy_pytest_evidence_schema(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    sidecar = json.loads(pytest_evidence.read_text(encoding="utf-8"))
    sidecar["schema"] = "legacy-junit-without-source-identity"
    pytest_evidence.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_replaced_junit_bytes(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    pytest_report.write_text(
        '<testsuites tests="4" failures="0" errors="0" skipped="0">'
        '<testsuite tests="4" failures="0" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_junit_from_different_source_digest(tmp_path, monkeypatch):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    snapshot = pytest_provenance.source_snapshot(ROOT)
    monkeypatch.setattr(pytest_provenance, "source_snapshot", lambda root: {
        "digest": "f" * 64 if snapshot["digest"] != "f" * 64 else "e" * 64,
        "files": snapshot["files"],
    })

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_source_changes_during_pytest(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    sidecar = json.loads(pytest_evidence.read_text(encoding="utf-8"))
    sidecar["source_digest_after"] = "f" * 64 if sidecar["source_digest_before"] != "f" * 64 else "e" * 64
    pytest_evidence.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_failed_pytest_sidecar(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    sidecar = json.loads(pytest_evidence.read_text(encoding="utf-8"))
    sidecar["exit_code"] = 1
    pytest_evidence.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)
