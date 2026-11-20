"""Bind synthetic release reports to actual fixture bytes, without running an EXE."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from manialab import __version__
from scripts import collect_release_validation as collector
from scripts import package_assets, verify_portable, verify_shutdown


def _sha(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture(params=['core', 'standard'])
def release_fixture(tmp_path, monkeypatch, request):
    profile = request.param
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'launcher.py').write_bytes(b'synthetic source fixture')
    build = tmp_path / 'build'
    executable = build / 'dist/ManiaLab/ManiaLab.exe'
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b'synthetic executable fixture; never executed')
    model_bytes = b'synthetic model fixture; never loaded'
    model = dict(filename='basic-pitch.onnx', version='synthetic', url='synthetic-fixture',
        sha256=_sha(model_bytes))
    # Redirect only source/git identity and pinned model metadata to controlled fixtures.
    # Production ZIP, checksum, report, and synthetic model-byte validation still run.
    monkeypatch.setattr(collector, 'ROOT', source)
    monkeypatch.setattr(collector.subprocess, 'check_output', lambda *args, **kwargs: 'synthetic-commit')
    monkeypatch.setattr(collector, 'validate_evidence', lambda *args, **kwargs: {
        'schema': 'manialab-pytest-evidence-v1',
        'source_commit': 'synthetic-commit',
        'source_digest_before': 'a' * 64,
    })
    monkeypatch.setattr(package_assets, 'CORE_MODEL', model)
    declared = package_assets.capabilities(profile)
    identity = dict(build_profile=profile, capabilities=declared,
        executable_sha256=_sha(executable.read_bytes()))
    uv_runtime_bytes = b'synthetic uv executable fixture; never executed'
    uv_runtime_identity = dict(collector.uv_release_identity())
    uv_runtime_identity.update(bytes=len(uv_runtime_bytes), sha256=_sha(uv_runtime_bytes))
    monkeypatch.setattr(collector, 'uv_release_identity', lambda: uv_runtime_identity)
    runtime_license = b'synthetic packaged runtime license'
    runtime_inventory = b'[]'
    runtime_license_record = {'bytes': len(runtime_license), 'sha256': _sha(runtime_license)}
    runtime_inventory_record = {'bytes': len(runtime_inventory), 'sha256': _sha(runtime_inventory)}
    manifest = dict(identity, version=__version__,
        sources={'launcher.py': _sha((source / 'launcher.py').read_bytes())}, core_models=model,
        uv_runtime=uv_runtime_identity if declared['mapperatorinator_install'] else None,
        worker_sha256='synthetic-worker' if profile != 'core' else None,
        worker_sources={} if profile != 'core' else None, worker_files=[] if profile != 'core' else None,
        piano_model={'sha256': 'f' * 64, 'bytes': 1} if profile != 'core' else None,
        guitar_model=None, instrument_worker=None, instrument_runtime_license_files=None,
        runtime_license_files={'fixture/LICENSE': runtime_license_record},
        runtime_license_inventory=runtime_inventory_record,
        license_files={'runtime/fixture/LICENSE': runtime_license_record,
            'runtime-inventory.json': runtime_inventory_record})
    (executable.parent / 'release-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    identity = dict(identity, release_manifest_sha256=_sha(
        (executable.parent / 'release-manifest.json').read_bytes()))
    portable_identity = dict(identity, verification_run_id='c' * 32)
    package = build / 'release'
    package.mkdir()
    archive = package / (package_assets.archive_stem(__version__, profile) + '.zip')
    with zipfile.ZipFile(archive, 'w') as output:
        output.write(executable, 'ManiaLab/ManiaLab.exe')
        output.writestr('ManiaLab/release-manifest.json', json.dumps(manifest))
        output.writestr('ManiaLab/_internal/assets/basic-pitch.onnx', model_bytes)
        output.writestr('ManiaLab/_internal/assets/model-manifest.json', json.dumps({
            'basic_pitch': {key: value for key, value in model.items() if key != 'filename'}}))
        output.writestr('ManiaLab/_internal/assets/arrangement-profiles.json', '{}')
        if declared['mapperatorinator_install']:
            output.writestr('ManiaLab/_internal/assets/runtime/uv.exe', uv_runtime_bytes)
        for prefix in ('ManiaLab/licenses/', 'ManiaLab/_internal/licenses/'):
            output.writestr(prefix + 'runtime/fixture/LICENSE', runtime_license)
            output.writestr(prefix + 'runtime-inventory.json', runtime_inventory)
    junit = tmp_path / 'junit.xml'
    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>', encoding='utf-8')
    pytest_evidence = tmp_path / 'pytest-evidence.json'
    pytest_evidence.write_text('{}', encoding='utf-8')
    portable = tmp_path / 'portable.json'
    checks = [
        dict(portable_identity, check='release_manifest'),
        dict(check='startup', version=__version__, separator={
            'available': declared['separation_four'],
            'profiles': {
                'four': {'available': declared['separation_four']},
                'six': {'available': declared['separation_six']},
            },
        }, prior={'available': declared['pattern_prior']},
            engine={'install_available': declared['mapperatorinator_install']}),
        dict(check='restart', preserved=True, keys=10, unicode_audio='音频测试.wav'),
    ]
    checks += [dict(check='generate_export', keys=keys, notes=5, ln=5, off_grid=0)
        for keys in range(6, 11)]
    if profile == 'core':
        checks += [
            dict(check='basic_analysis', engine='basic', candidates=5, seconds=1.0),
            dict(check='provided_stem_analysis_export', voices=['other'], candidates=5,
                notes=5, ln=5, off_grid=0),
            dict(check='bad_audio_recovery', analysis='error', service_alive=True),
        ]
    else:
        checks += [dict(check='piano_onnx_runtime', sha256='f' * 64, bytes=1,
            providers=['CPUExecutionProvider'], output_shapes=[[1, 1001, 88]] * 4 + [[1, 1001, 1]] * 3,
            input='all-zero synthetic 10-second tensor; runtime smoke test only')]
        checks += [
            dict(check='deep_analysis', engine='deep', seconds=1.0, candidates=5),
            dict(check='six_stem_analysis_export', seconds=1.0,
                voices=['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'], notes=5, off_grid=0),
            dict(check='cancel_and_failure', cancelled='cancelled', bad_audio='error', service_alive=True),
        ]
    checks = [dict(row, **portable_identity) for row in checks]
    portable.write_text(json.dumps(checks), encoding='utf-8')
    shutdown = tmp_path / 'shutdown.json'
    shutdown_data = dict(identity, version=__version__, passed=True, worker_expected=profile != 'core',
        application_exited=True, worker_stopped_on_parent_exit=profile != 'core')
    if profile == 'core':
        shutdown_data.update(worker_processes_before_exit=[], worker_processes_after_exit=[])
    shutdown.write_text(json.dumps(shutdown_data), encoding='utf-8')
    return SimpleNamespace(build=build, executable=executable, manifest=manifest, identity=identity,
        junit=junit, pytest_evidence=pytest_evidence, portable=portable, shutdown=shutdown,
        checks=checks, shutdown_data=shutdown_data)


def _collect(fixture):
    return collector.collect(fixture.build, fixture.junit, fixture.portable, fixture.shutdown,
        pytest_evidence=fixture.pytest_evidence)


def test_new_evidence_binding_keeps_stale_rows_and_snapshots_the_current_identity():
    identity = dict(build_profile='core', capabilities={'piano': False},
        executable_sha256='a' * 64, release_manifest_sha256='b' * 64,
        verification_run_id='c' * 32)
    stale = dict(check='startup', verification_run_id='d' * 32)
    checks = [stale.copy()]
    row = dict(check='generate_export', keys=6, notes=5, ln=5, off_grid=0)

    verify_portable.append_verification_evidence(checks, identity, row)

    assert checks[0] == stale
    assert checks[1] == dict(row, **identity)
    assert row == dict(check='generate_export', keys=6, notes=5, ln=5, off_grid=0)
    identity['capabilities']['piano'] = True
    assert checks[1]['capabilities']['piano'] is False


@pytest.mark.parametrize('specialist', [False, True])
def test_current_evidence_binding_rejects_a_previously_bound_row(specialist):
    identity = {'verification_run_id': 'c' * 32}
    row = dict(check='guitar_real_audio_http' if specialist else 'startup',
        verification_run_id='d' * 32)
    if specialist:
        checks = [row.copy()]
        with pytest.raises(ValueError, match='another run'):
            verify_portable.bind_new_specialist_evidence(checks, 0, identity)
        assert checks == [row]
    else:
        checks = []
        with pytest.raises(ValueError, match='another run'):
            verify_portable.append_verification_evidence(checks, identity, row)
        assert checks == []


def test_matching_release_reports_are_accepted_for_the_same_fixture_bytes(release_fixture):
    report, output, _ = _collect(release_fixture)
    assert report['executable_sha256'] == _sha(release_fixture.executable.read_bytes())
    assert report['verification']['portable']['passed'] is True
    assert report['verification']['shutdown']['passed'] is True
    assert report['verification']['piano_audio']['status'] == 'not_run'
    assert report['verification']['piano_audio']['musical_acceptance'] is False
    if release_fixture.identity['build_profile'] == 'standard':
        assert report['verification']['piano_model']['cpu_smoke_passed'] is True
        assert report['verification']['piano_audio']['reason'] == 'optional_audio_fixture_not_supplied'
    else:
        assert report['verification']['piano_model'] is None
        assert report['verification']['piano_audio']['reason'] == 'profile_has_no_piano_audio_capability'
    assert output.is_file()


@pytest.mark.parametrize('document', [{}, None, 'not-an-array'])
def test_portable_report_root_requires_an_array(release_fixture, document):
    release_fixture.portable.write_text(json.dumps(document), encoding='utf-8')

    with pytest.raises(SystemExit, match='unexpected format'):
        _collect(release_fixture)


@pytest.mark.parametrize('row', [
    None, [], 'not-a-row', {},
    {'check': []}, {'check': {}}, {'check': None},
    {'check': ''}, {'check': '   '},
])
def test_portable_report_rejects_malformed_check_rows(release_fixture, row):
    checks = json.loads(release_fixture.portable.read_text(encoding='utf-8'))
    checks.append(row)
    release_fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='unexpected format'):
        _collect(release_fixture)


def test_portable_report_rejects_unknown_check_names(release_fixture):
    checks = json.loads(release_fixture.portable.read_text(encoding='utf-8'))
    checks.append(dict(
        release_fixture.identity,
        verification_run_id='c' * 32,
        check='unverified_extension_check',
    ))
    release_fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='unsupported check name'):
        _collect(release_fixture)


PIANO_AUDIO_CHECKS = (
    'piano_cold_inference',
    'piano_manual_expression',
    'piano_restart_warm_cache',
)


def _add_piano_audio_checks(fixture, names=PIANO_AUDIO_CHECKS):
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    rows = {
        'piano_cold_inference': dict(
            input_sha256='a' * 64, model_sha256='f' * 64, candidates=5,
            seconds=1.25, release_reasons={'frame_decay': 5},
        ),
        'piano_manual_expression': dict(
            modes={
                'articulation': {'notes': 5, 'ln': 0},
                'sustain': {'notes': 5, 'ln': 3},
            },
            evidence_preserved=True,
            timing='fixed synthetic grid for export verification, not musical timing acceptance',
        ),
        'piano_restart_warm_cache': dict(candidates=5, identical=True, seconds=0.75),
    }
    identity = dict(fixture.identity, verification_run_id='c' * 32)
    checks.extend(dict(check=name, **rows[name], **identity) for name in names)
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')
    return checks


def test_complete_optional_piano_audio_evidence_is_classified_without_music_acceptance(release_fixture):
    if release_fixture.identity['build_profile'] != 'standard':
        pytest.skip('piano audio checks require a non-Core profile')
    _add_piano_audio_checks(release_fixture)

    report, _, _ = _collect(release_fixture)

    piano_audio = report['verification']['piano_audio']
    assert piano_audio['status'] == 'passed'
    assert piano_audio['checks'] == list(PIANO_AUDIO_CHECKS)
    assert piano_audio['input_sha256'] == 'a' * 64
    assert piano_audio['model_sha256'] == 'f' * 64
    assert piano_audio['musical_acceptance'] is False
    assert report['limitations']['musical_acceptance'] is False


@pytest.mark.parametrize('present', [
    ('piano_cold_inference',),
    ('piano_manual_expression',),
    ('piano_restart_warm_cache',),
    ('piano_cold_inference', 'piano_manual_expression'),
    ('piano_cold_inference', 'piano_restart_warm_cache'),
    ('piano_manual_expression', 'piano_restart_warm_cache'),
])
def test_partial_optional_piano_audio_evidence_is_rejected(release_fixture, present):
    if release_fixture.identity['build_profile'] != 'standard':
        pytest.skip('piano audio checks require a non-Core profile')
    _add_piano_audio_checks(release_fixture, present)

    with pytest.raises(SystemExit, match='Portable check payload'):
        _collect(release_fixture)


@pytest.mark.parametrize('check_name', PIANO_AUDIO_CHECKS)
@pytest.mark.parametrize('field', [
    'build_profile', 'capabilities', 'executable_sha256',
    'release_manifest_sha256', 'verification_run_id',
])
def test_optional_piano_audio_rows_require_current_run_identity(release_fixture, check_name, field):
    if release_fixture.identity['build_profile'] != 'standard':
        pytest.skip('piano audio checks require a non-Core profile')
    checks = _add_piano_audio_checks(release_fixture)
    row = next(item for item in checks if item.get('check') == check_name)
    row[field] = {
        'build_profile': 'extended',
        'capabilities': {'piano': False},
        'executable_sha256': _sha(b'other executable'),
        'release_manifest_sha256': _sha(b'other manifest'),
        'verification_run_id': 'd' * 32,
    }[field]
    release_fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    expected_error = (
        'Portable verification identity capabilities'
        if field == 'capabilities' else 'Portable check identity'
    )
    with pytest.raises(SystemExit, match=expected_error):
        _collect(release_fixture)


def test_duplicate_optional_piano_audio_check_is_rejected(release_fixture):
    if release_fixture.identity['build_profile'] != 'standard':
        pytest.skip('piano audio checks require a non-Core profile')
    checks = _add_piano_audio_checks(release_fixture)
    cold = next(item for item in checks if item.get('check') == 'piano_cold_inference')
    checks.append(cold.copy())
    release_fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Portable check payload'):
        _collect(release_fixture)


def test_core_rejects_optional_piano_audio_rows(release_fixture):
    if release_fixture.identity['build_profile'] != 'core':
        pytest.skip('Core profile only')
    _add_piano_audio_checks(release_fixture)

    with pytest.raises(SystemExit, match='not applicable to the declared profile or capabilities'):
        _collect(release_fixture)


@pytest.mark.parametrize(('check_name', 'field', 'value'), [
    ('piano_cold_inference', 'input_sha256', 'bad'),
    ('piano_cold_inference', 'model_sha256', '0' * 64),
    ('piano_cold_inference', 'candidates', 0),
    ('piano_cold_inference', 'seconds', float('inf')),
    ('piano_cold_inference', 'release_reasons', {'frame_decay': 4}),
    ('piano_cold_inference', 'release_reasons', {'frame_decay': 6}),
    ('piano_manual_expression', 'modes', {'articulation': {'notes': 5, 'ln': 0}}),
    ('piano_manual_expression', 'evidence_preserved', False),
    ('piano_restart_warm_cache', 'identical', False),
    ('piano_restart_warm_cache', 'candidates', 4),
    ('piano_restart_warm_cache', 'candidates', 5.0),
])
def test_optional_piano_audio_rows_reject_invalid_payloads(release_fixture, check_name, field, value):
    if release_fixture.identity['build_profile'] != 'standard':
        pytest.skip('piano audio checks require a non-Core profile')
    checks = _add_piano_audio_checks(release_fixture)
    row = next(item for item in checks if item.get('check') == check_name)
    row[field] = value
    release_fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Portable check payload'):
        _collect(release_fixture)


@pytest.mark.parametrize('report_kind', ['portable', 'shutdown'])
@pytest.mark.parametrize('field', ['executable_sha256', 'build_profile', 'capabilities',
    'release_manifest_sha256'])
def test_reports_for_other_bytes_or_capabilities_are_rejected(release_fixture, report_kind, field):
    fixture = release_fixture
    document = json.loads(getattr(fixture, report_kind).read_text(encoding='utf-8'))
    identity = document[0] if report_kind == 'portable' else document
    identity[field] = {'executable_sha256': _sha(b'another executable'),
        'build_profile': 'extended', 'capabilities': {'audio': False},
        'release_manifest_sha256': _sha(b'another release manifest')}[field]
    getattr(fixture, report_kind).write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(SystemExit, match='identity'):
        _collect(fixture)


@pytest.mark.parametrize('report_kind', ['portable', 'shutdown'])
@pytest.mark.parametrize('field', ['executable_sha256', 'release_manifest_sha256'])
def test_legacy_reports_without_build_identity_are_rejected(release_fixture, report_kind, field):
    fixture = release_fixture
    document = json.loads(getattr(fixture, report_kind).read_text(encoding='utf-8'))
    identity = document[0] if report_kind == 'portable' else document
    del identity[field]
    getattr(fixture, report_kind).write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(SystemExit, match='identity'):
        _collect(fixture)


@pytest.mark.parametrize(('check_name', 'failed_payload'), [
    ('deep_analysis', {'engine': 'old-exe', 'candidates': 0, 'seconds': -1}),
    ('six_stem_analysis_export', {'voices': [], 'notes': 0, 'off_grid': 9}),
    ('cancel_and_failure', {'cancelled': 'done', 'bad_audio': 'done', 'service_alive': False}),
])
def test_standard_collector_rejects_failed_required_portable_payloads(
        release_fixture, check_name, failed_payload):
    fixture = release_fixture
    if fixture.identity['build_profile'] != 'standard':
        pytest.skip('standard profile only')
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    row = next(item for item in checks if item.get('check') == check_name)
    row.update(failed_payload)
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Portable check payload'):
        _collect(fixture)


def test_duplicate_portable_identity_checks_are_not_ambiguously_accepted(release_fixture):
    fixture = release_fixture
    fixture.portable.write_text(json.dumps(fixture.checks + [fixture.checks[0]]), encoding='utf-8')
    with pytest.raises(SystemExit, match='exactly one'):
        _collect(fixture)


@pytest.mark.parametrize('field', [
    'build_profile', 'capabilities', 'executable_sha256',
    'release_manifest_sha256', 'verification_run_id',
])
def test_required_portable_row_identity_claims_must_match_current_run(release_fixture, field):
    fixture = release_fixture
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    row = next(item for item in checks if item.get('check') == 'generate_export')
    row.update({
        'build_profile': fixture.identity['build_profile'],
        'capabilities': fixture.identity['capabilities'],
        'executable_sha256': fixture.identity['executable_sha256'],
        'release_manifest_sha256': fixture.identity['release_manifest_sha256'],
        'verification_run_id': 'c' * 32,
    })
    row[field] = {
        'build_profile': 'extended',
        'capabilities': {'audio': False},
        'executable_sha256': _sha(b'other executable'),
        'release_manifest_sha256': _sha(b'other manifest'),
        'verification_run_id': 'd' * 32,
    }[field]
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')
    expected_error = (
        'Portable verification identity capabilities'
        if field == 'capabilities' else 'Portable check identity'
    )
    with pytest.raises(SystemExit, match=expected_error):
        _collect(fixture)


PORTABLE_REQUIRED_CHECKS = [
    'startup', 'restart', 'generate_export', 'basic_analysis',
    'provided_stem_analysis_export', 'bad_audio_recovery', 'piano_onnx_runtime',
    'deep_analysis', 'six_stem_analysis_export', 'cancel_and_failure',
]
PORTABLE_IDENTITY_FIELDS = [
    'build_profile', 'capabilities', 'executable_sha256',
    'release_manifest_sha256', 'verification_run_id',
]


@pytest.mark.parametrize('check_name', PORTABLE_REQUIRED_CHECKS)
@pytest.mark.parametrize('stale_kind', ['missing_identity', 'old_run'])
def test_current_report_rejects_a_valid_payload_row_from_an_unbound_or_old_run(
        release_fixture, check_name, stale_kind):
    fixture = release_fixture
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    row = next((item for item in checks if item['check'] == check_name), None)
    if row is None:
        pytest.skip('check belongs to another profile')
    if stale_kind == 'missing_identity':
        for field in PORTABLE_IDENTITY_FIELDS:
            del row[field]
    else:
        row['verification_run_id'] = 'd' * 32
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Portable check identity'):
        _collect(fixture)


@pytest.mark.parametrize('field', PORTABLE_IDENTITY_FIELDS)
def test_required_portable_row_cannot_omit_an_identity_field(release_fixture, field):
    fixture = release_fixture
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    row = next(item for item in checks if item['check'] == 'generate_export')
    del row[field]
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Portable check identity'):
        _collect(fixture)


@pytest.mark.parametrize('check_name', PORTABLE_REQUIRED_CHECKS)
def test_required_portable_rows_must_not_be_ambiguous(release_fixture, check_name):
    fixture = release_fixture
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    row = next((item for item in checks if item['check'] == check_name), None)
    if row is None:
        pytest.skip('check belongs to another profile')
    checks.append(row.copy())
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Portable check payload'):
        _collect(fixture)


def test_generate_export_requires_every_key_count(release_fixture):
    fixture = release_fixture
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    checks = [row for row in checks if not (row['check'] == 'generate_export' and row['keys'] == 7)]
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')
    with pytest.raises(SystemExit, match='6K–10K'):
        _collect(fixture)


@pytest.mark.parametrize(('check_name', 'invalid_payload'), [
    ('startup', {'separator': None}),
    ('restart', {'preserved': False}),
    ('generate_export', {'notes': True}),
    ('generate_export', {'ln': 6}),
    ('generate_export', {'off_grid': False}),
    ('basic_analysis', {'seconds': float('nan')}),
    ('provided_stem_analysis_export', {'candidates': 0}),
    ('bad_audio_recovery', {'service_alive': False}),
    ('piano_onnx_runtime', {'providers': None}),
    ('piano_onnx_runtime', {'output_shapes': []}),
    ('deep_analysis', {'seconds': float('inf')}),
    ('six_stem_analysis_export', {'voices': [{}]}),
    ('cancel_and_failure', {'service_alive': 1}),
])
def test_required_portable_results_reject_invalid_types_and_values(
        release_fixture, check_name, invalid_payload):
    fixture = release_fixture
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    row = next((item for item in checks if item['check'] == check_name), None)
    if row is None:
        pytest.skip('check belongs to another profile')
    row.update(invalid_payload)
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')
    with pytest.raises(SystemExit, match='Portable check payload'):
        _collect(fixture)


def test_success_payload_does_not_depend_on_a_specific_description_or_unicode_filename(release_fixture):
    fixture = release_fixture
    checks = json.loads(fixture.portable.read_text(encoding='utf-8'))
    next(row for row in checks if row['check'] == 'restart')['unicode_audio'] = '合成演奏.wav'
    for row in checks:
        if row['check'] == 'piano_onnx_runtime':
            row['input'] = 'Synthetic runtime smoke; no music quality claim'
    fixture.portable.write_text(json.dumps(checks), encoding='utf-8')
    assert _collect(fixture)[0]['verification']['portable']['passed'] is True


def test_shutdown_report_version_must_match_the_verified_manifest(release_fixture):
    fixture = release_fixture
    fixture.shutdown_data['version'] = 'a different release'
    fixture.shutdown.write_text(json.dumps(fixture.shutdown_data), encoding='utf-8')
    with pytest.raises(SystemExit, match='identity'):
        _collect(fixture)


def test_shutdown_report_identity_uses_file_bytes_and_rejects_changed_files(release_fixture):
    fixture = release_fixture
    identity = verify_shutdown.shutdown_report_identity(fixture.executable, fixture.manifest)
    assert identity == dict(fixture.identity, version=__version__)
    fixture.executable.write_bytes(b'changed executable fixture')
    with pytest.raises(ValueError, match='identity'):
        verify_shutdown.shutdown_report_identity(fixture.executable, fixture.manifest)


def test_same_executable_with_changed_source_manifest_rejects_old_reports(release_fixture):
    fixture = release_fixture
    source = collector.ROOT / 'static/app.js'
    source.parent.mkdir()
    source.write_bytes(b'updated synthetic static behavior')
    fixture.manifest['sources']['static/app.js'] = _sha(source.read_bytes())
    manifest_path = fixture.executable.parent / 'release-manifest.json'
    manifest_path.write_text(json.dumps(fixture.manifest, indent=2), encoding='utf-8')
    archive = next((fixture.build / 'release').glob('*.zip'))
    with zipfile.ZipFile(archive) as package:
        files = {name: package.read(name) for name in package.namelist()}
    files['ManiaLab/release-manifest.json'] = manifest_path.read_bytes()
    files['ManiaLab/_internal/static/app.js'] = source.read_bytes()
    with zipfile.ZipFile(archive, 'w') as package:
        for name, content in files.items():
            package.writestr(name, content)
    assert _sha(fixture.executable.read_bytes()) == fixture.identity['executable_sha256']
    with pytest.raises(SystemExit, match='Portable verification identity'):
        _collect(fixture)

    # Updating only the portable report must still reject the old shutdown evidence.
    for row in fixture.checks:
        row['release_manifest_sha256'] = _sha(manifest_path.read_bytes())
    fixture.portable.write_text(json.dumps(fixture.checks), encoding='utf-8')
    with pytest.raises(SystemExit, match='Shutdown verification identity'):
        _collect(fixture)


def test_zip_manifest_bytes_must_match_the_verified_manifest(release_fixture):
    fixture = release_fixture
    archive = next((fixture.build / 'release').glob('*.zip'))
    with zipfile.ZipFile(archive) as package:
        files = {name: package.read(name) for name in package.namelist()}
    manifest_bytes = files['ManiaLab/release-manifest.json']
    files['ManiaLab/release-manifest.json'] = b'\n' + manifest_bytes
    with zipfile.ZipFile(archive, 'w') as package:
        for name, content in files.items():
            package.writestr(name, content)
    with pytest.raises(SystemExit, match='manifest bytes'):
        _collect(fixture)
