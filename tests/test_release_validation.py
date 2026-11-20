import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from manialab import __version__
from scripts import pytest_provenance
from scripts.build_provenance import uv_release_identity
from scripts import collect_release_validation
from scripts.collect_release_validation import collect as _collect
from scripts.package_assets import CORE_FILES, capabilities, verify_core_model_files
from scripts.worker_inventory import worker_inventory

ROOT = Path(__file__).resolve().parents[1]
TEST_GUITAR_MODEL = b"synthetic guitar model"
TEST_GUITAR_LICENSE = b"synthetic guitar license"
TEST_GUITAR_SOURCE_NOTICE = b"synthetic guitar source notice"
TEST_GUITAR_NOTICE = b"synthetic guitar notice"
TEST_INSTRUMENT_WORKER = b"synthetic instrument worker"
TEST_INSTRUMENT_LICENSE = b"synthetic instrument license"
TEST_INSTRUMENT_VERIFICATION = b"synthetic instrument verification"
TEST_INSTRUMENT_SOURCE = b"synthetic instrument source"
TEST_RUNTIME_LICENSE = b"synthetic runtime dependency license"
TEST_RUNTIME_LICENSE_INVENTORY = b"[]"
TEST_PACKAGED_RUNTIME_LICENSE = b"synthetic packaged runtime license"
TEST_PACKAGED_RUNTIME_INVENTORY = b'[{"name":"fixture","version":"1","license":"MIT"}]'
TEST_UV_LICENSE = b"synthetic uv license"
TEST_UV_RUNTIME = b"synthetic uv executable"
TEST_FFMPEG_LICENSE = b"synthetic FFmpeg license"
TEST_FFMPEG_BUILD_INFO = b"synthetic FFmpeg build info"
TEST_BASS_MODEL = b"synthetic bass model"
TEST_OTHER_MODEL = b"synthetic other model"


def _license_staging_helpers():
    source = ast.parse((ROOT / 'scripts/build_windows.py').read_text(encoding='utf-8'))
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
        and node.name in {'stage_license_inputs', 'stage_runtime_licenses'}]
    namespace = dict(
        Path=Path,
        shutil=shutil,
        json=json,
        hashlib=hashlib,
        worker_inventory=worker_inventory,
    )
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'license-staging-only', 'exec'), namespace)
    return namespace['stage_license_inputs'], namespace['stage_runtime_licenses']


def test_build_stages_only_current_environment_runtime_licenses(tmp_path):
    stage_license_inputs, stage_runtime_licenses = _license_staging_helpers()
    source = tmp_path / 'source-licenses'
    (source / 'runtime' / 'old-package').mkdir(parents=True)
    (source / 'runtime' / 'old-package' / 'LICENSE').write_bytes(b'stale ignored license')
    (source / 'runtime-inventory.json').write_text('[{"name":"old-package"}]', encoding='utf-8')
    (source / 'uv-MIT.txt').write_bytes(TEST_UV_LICENSE)
    environment = tmp_path / 'environment'
    license_entry = Path('fixture-2.0.dist-info/LICENSE')
    license_source = environment / license_entry
    license_source.parent.mkdir(parents=True)
    license_source.write_bytes(b'current environment license')

    class Distribution:
        metadata = {'Name': 'fixture', 'License-Expression': 'MIT'}
        version = '2.0'
        files = (license_entry,)

        @staticmethod
        def locate_file(entry):
            return environment / entry

    stage = tmp_path / 'build' / 'licenses'
    stage_license_inputs(source, stage)
    assert not (stage / 'runtime').exists()
    assert not (stage / 'runtime-inventory.json').exists()
    assert (stage / 'uv-MIT.txt').read_bytes() == TEST_UV_LICENSE

    files, inventory = stage_runtime_licenses(stage, [Distribution()])
    assert files == {
        'fixture/fixture-2.0.dist-info/LICENSE': payload_record(b'current environment license'),
    }
    assert json.loads((stage / 'runtime-inventory.json').read_text(encoding='utf-8')) == [{
        'name': 'fixture', 'version': '2.0', 'license': 'MIT',
    }]
    inventory_bytes = (stage / 'runtime-inventory.json').read_bytes()
    assert inventory == payload_record(inventory_bytes)
    assert (source / 'runtime' / 'old-package' / 'LICENSE').read_bytes() == b'stale ignored license'


def test_build_source_snapshot_does_not_read_old_runtime_inventory(tmp_path, monkeypatch):
    source = ast.parse((ROOT / 'scripts/build_windows.py').read_text(encoding='utf-8'))
    function = next(node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == 'source_snapshot')
    namespace = dict(ROOT=tmp_path, hashlib=hashlib, enabled=capabilities('core'))
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'source-snapshot-only', 'exec'), namespace)
    licenses = tmp_path / 'licenses'
    licenses.mkdir()
    (licenses / 'runtime-inventory.json').write_bytes(b'stale ignored inventory')
    (licenses / 'uv-MIT.txt').write_bytes(TEST_UV_LICENSE)

    def synthetic_source_bytes(path):
        if path == licenses / 'runtime-inventory.json':
            pytest.fail('The build source snapshot read the old ignored inventory')
        return TEST_UV_LICENSE

    # All other declared source files use synthetic bytes; no builder import or EXE runs.
    monkeypatch.setattr(Path, 'read_bytes', synthetic_source_bytes)
    snapshot = namespace['source_snapshot']()
    assert 'licenses/runtime-inventory.json' not in snapshot
    assert snapshot['licenses/uv-MIT.txt'] == hashlib.sha256(TEST_UV_LICENSE).hexdigest()


def payload_identity(value):
    return hashlib.sha256(value).hexdigest()


def payload_record(value):
    return {"bytes": len(value), "sha256": payload_identity(value)}


def packaged_license_payloads(profile):
    files = {
        'runtime/fixture/LICENSE': TEST_PACKAGED_RUNTIME_LICENSE,
        'runtime-inventory.json': TEST_PACKAGED_RUNTIME_INVENTORY,
        'uv-MIT.txt': TEST_UV_LICENSE,
        'FFmpeg-GPLv3.txt': TEST_FFMPEG_LICENSE,
        'FFmpeg-build-info.txt': TEST_FFMPEG_BUILD_INFO,
    }
    if profile == 'extended':
        files.update({
            'HFGuitar-Model-Card.md': TEST_GUITAR_LICENSE,
            'HFGuitar-Source-Setup.py.txt': TEST_GUITAR_SOURCE_NOTICE,
            'HFGuitar-NOTICE.txt': TEST_GUITAR_NOTICE,
            'instrument-runtime/inventory.json': TEST_RUNTIME_LICENSE_INVENTORY,
            'instrument-runtime/torch/LICENSE': TEST_RUNTIME_LICENSE,
        })
    return files


def write_license_payloads(archive, files):
    for prefix in ('ManiaLab/licenses/', 'ManiaLab/_internal/licenses/'):
        for name, content in files.items():
            archive.writestr(prefix + name, content)


def make_guitar_model_identity():
    identity = json.loads((ROOT / 'assets/guitar-model.json').read_text(encoding='utf-8'))
    identity['sha256'] = payload_identity(TEST_GUITAR_MODEL)
    identity['license_sha256'] = payload_identity(TEST_GUITAR_LICENSE)
    identity['source_license_declaration_sha256'] = payload_identity(TEST_GUITAR_SOURCE_NOTICE)
    identity['notice_sha256'] = payload_identity(TEST_GUITAR_NOTICE)
    return identity


TEST_GUITAR_IDENTITY = make_guitar_model_identity()


def collect(*args, **kwargs):
    original_guitar_validator = collect_release_validation._is_guitar_model_identity
    original_instrument_validator = collect_release_validation._is_instrument_worker_identity
    original_uv_release_identity = collect_release_validation.uv_release_identity
    instrument_identity = make_instrument_worker_identity()
    build_root = kwargs.get("build_root", args[0] if args else None)
    manifest_path = Path(build_root) / "dist" / "ManiaLab" / "release-manifest.json"
    fixture_uv_identity = json.loads(manifest_path.read_text(encoding="utf-8")).get("uv_runtime")
    collect_release_validation._is_guitar_model_identity = (
        lambda value: value == TEST_GUITAR_IDENTITY or original_guitar_validator(value)
    )
    collect_release_validation._is_instrument_worker_identity = (
        lambda value: value == instrument_identity or original_instrument_validator(value)
    )
    collect_release_validation.uv_release_identity = lambda: fixture_uv_identity
    try:
        return _collect(*args, **kwargs)
    finally:
        collect_release_validation._is_guitar_model_identity = original_guitar_validator
        collect_release_validation._is_instrument_worker_identity = original_instrument_validator
        collect_release_validation.uv_release_identity = original_uv_release_identity


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
        MODEL_REVISION, SOURCE_COMMIT, runtime_identity,
    )

    worker_sha = payload_identity(TEST_INSTRUMENT_WORKER)
    models = {
        "bass_v2": payload_identity(TEST_BASS_MODEL),
        "other_v1_5": payload_identity(TEST_OTHER_MODEL),
    }
    return {
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "source_files": {"fixture.py": payload_identity(TEST_INSTRUMENT_SOURCE)},
        "models": models,
        "license": "MIT",
        "license_sha256": payload_identity(TEST_INSTRUMENT_LICENSE),
        "voices": ["bass", "other"],
        "prepared_only": False,
        "portable_runtime_verified": True,
        "worker_sources": runtime_identity(ROOT),
        "worker_sha256": worker_sha,
        "worker_files": {
            "ManiaLabInstrument.exe": {
                "bytes": len(TEST_INSTRUMENT_WORKER),
                "sha256": worker_sha,
            }
        },
        "verified_models": list(models),
        "verification_sha256": payload_identity(TEST_INSTRUMENT_VERIFICATION),
    }


def make_release_fixture(root, profile='standard'):
    build_root = root / "build"
    executable = build_root / "dist" / "ManiaLab" / "ManiaLab.exe"
    package = build_root / "release"
    executable.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    executable.write_bytes(b"portable-executable")
    license_payloads = packaged_license_payloads(profile)
    uv_identity = uv_release_identity()
    uv_identity.update(bytes=len(TEST_UV_RUNTIME), sha256=payload_identity(TEST_UV_RUNTIME))
    manifest = {
        "version": __version__,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "sources": {},
        "build_profile": profile,
        "capabilities": capabilities(profile),
        "core_models": verify_core_model_files(ROOT / 'assets'),
        "uv_runtime": uv_identity if profile != 'core' else None,
        "piano_model": {"sha256": "f" * 64, "bytes": 123} if profile != 'core' else None,
        "worker_sha256": "expected-worker-sha" if profile != 'core' else None,
        "worker_sources": {} if profile != 'core' else None,
        "worker_files": [] if profile != 'core' else None,
        "guitar_model": TEST_GUITAR_IDENTITY if profile == 'extended' else None,
        "instrument_worker": make_instrument_worker_identity() if profile == 'extended' else None,
        "instrument_runtime_license_files": (
            {
                "inventory.json": payload_record(TEST_RUNTIME_LICENSE_INVENTORY),
                "torch/LICENSE": payload_record(TEST_RUNTIME_LICENSE),
            }
            if profile == 'extended' else None
        ),
        "runtime_license_files": {
            "fixture/LICENSE": payload_record(TEST_PACKAGED_RUNTIME_LICENSE),
        },
        "runtime_license_inventory": payload_record(TEST_PACKAGED_RUNTIME_INVENTORY),
        "license_files": {
            name: payload_record(content) for name, content in license_payloads.items()
        },
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
        if profile != 'core':
            archive.writestr('ManiaLab/_internal/assets/runtime/uv.exe', TEST_UV_RUNTIME)
        write_license_payloads(archive, license_payloads)
        if profile == 'extended':
            archive.writestr(
                'ManiaLab/_internal/assets/guitar-model.json',
                json.dumps(manifest['guitar_model']),
            )
            archive.writestr(
                'ManiaLab/_internal/assets/deep/guitar-fl.onnx', TEST_GUITAR_MODEL
            )
            instrument_root = 'ManiaLab/_internal/assets/instruments/'
            identity = manifest['instrument_worker']
            archive.writestr(instrument_root + 'manifest.json', json.dumps(identity))
            archive.writestr(instrument_root + 'verification.json', TEST_INSTRUMENT_VERIFICATION)
            archive.writestr(instrument_root + 'LICENSE-tsumugi', TEST_INSTRUMENT_LICENSE)
            archive.writestr(
                instrument_root + 'checkpoints/best_model_bass_v2.pth', TEST_BASS_MODEL
            )
            archive.writestr(
                instrument_root + 'checkpoints/best_model_other_v1_5.pth', TEST_OTHER_MODEL
            )
            archive.writestr(instrument_root + 'source/fixture.py', TEST_INSTRUMENT_SOURCE)
            archive.writestr(
                instrument_root + 'ManiaLabInstrument/ManiaLabInstrument.exe',
                TEST_INSTRUMENT_WORKER,
            )
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
        {
            "check": "startup",
            "version": manifest["version"],
            "separator": {
                "available": manifest["capabilities"]["separation_four"],
                "profiles": {
                    "four": {"available": manifest["capabilities"]["separation_four"]},
                    "six": {"available": manifest["capabilities"]["separation_six"]},
                },
            },
            "prior": {"available": manifest["capabilities"]["pattern_prior"]},
            "engine": {"install_available": manifest["capabilities"]["mapperatorinator_install"]},
        },
        {"check": "restart", "preserved": True, "keys": 10, "unicode_audio": "音频测试.wav"},
    ]
    portable_checks += [
        {"check": "generate_export", "keys": keys, "notes": 5, "ln": 5, "off_grid": 0}
        for keys in range(6, 11)
    ]
    if profile == 'core':
        portable_checks += [
            {"check": "basic_analysis", "engine": "basic", "candidates": 5, "seconds": 1.0},
            {"check": "provided_stem_analysis_export", "voices": ["other"], "candidates": 5,
             "notes": 5, "ln": 5, "off_grid": 0},
            {"check": "bad_audio_recovery", "analysis": "error", "service_alive": True},
        ]
    else:
        portable_checks += [
            {
                "check": "piano_onnx_runtime",
                "sha256": "f" * 64,
                "bytes": 123,
                "providers": ["CPUExecutionProvider"],
                "output_shapes": [[1, 1001, 88]] * 4 + [[1, 1001, 1]] * 3,
                "input": "all-zero synthetic 10-second tensor; runtime smoke test only",
            },
            {"check": "deep_analysis", "engine": "deep", "seconds": 1.0, "candidates": 5},
            {"check": "six_stem_analysis_export", "seconds": 1.0,
             "voices": ["drums", "bass", "vocals", "other", "guitar", "piano"],
             "notes": 5, "off_grid": 0},
            {"check": "cancel_and_failure", "cancelled": "cancelled",
             "bad_audio": "error", "service_alive": True},
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
    portable_checks = [dict(row, **verification_identity) for row in portable_checks]
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


def rewrite_archive(archive_path, *, omit=(), replacements=None, additions=None):
    with zipfile.ZipFile(archive_path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    for name in omit:
        entries.pop(name, None)
    entries.update(replacements or {})
    entries.update(additions or {})
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


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


def test_release_collector_requires_the_pinned_uv_release_identity(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = (
        make_release_fixture(tmp_path)
    )

    with pytest.raises(SystemExit, match='invalid uv_runtime identity'):
        _collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize(
    "invalid_identity",
    [pytest.param(None, id="missing"), pytest.param({"unexpected": "identity"}, id="non-null")],
)
def test_release_collector_requires_null_uv_identity_without_installer_capability(
    tmp_path, invalid_identity
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = (
        make_release_fixture(tmp_path, profile="core")
    )
    manifest_path = build_root / "dist" / "ManiaLab" / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if invalid_identity is None:
        manifest.pop("uv_runtime")
    else:
        manifest["uv_runtime"] = invalid_identity
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        SystemExit, match="uv_runtime=null without the installer capability"
    ):
        _collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


def test_release_collector_hashes_uv_inside_the_actual_zip(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path)
    )
    rewrite_archive(
        archive,
        replacements={'ManiaLab/_internal/assets/runtime/uv.exe': b'changed uv executable'},
    )

    with pytest.raises(SystemExit, match='differs from its manifest identity'):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


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
        if isinstance(document, list):
            for row in document:
                if isinstance(row, dict) and 'release_manifest_sha256' in row:
                    row['release_manifest_sha256'] = hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest()
        else:
            document['release_manifest_sha256'] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        report_path.write_text(json.dumps(document), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('ManiaLab/ManiaLab.exe', (build_root / 'dist/ManiaLab/ManiaLab.exe').read_bytes())
        bundle.writestr('ManiaLab/release-manifest.json', json.dumps(manifest))
        for name in CORE_FILES:
            bundle.write(ROOT / 'assets' / name, 'ManiaLab/_internal/assets/' + name)
        write_license_payloads(bundle, packaged_license_payloads('core'))

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
        write_license_payloads(bundle, packaged_license_payloads('core'))

    with pytest.raises(SystemExit, match='Basic Pitch model checksum'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


@pytest.mark.parametrize(('prefix', 'change'), (
    ('ManiaLab/licenses/', 'missing'),
    ('ManiaLab/licenses/', 'changed'),
    ('ManiaLab/licenses/', 'extra'),
    ('ManiaLab/_internal/licenses/', 'missing'),
    ('ManiaLab/_internal/licenses/', 'changed'),
    ('ManiaLab/_internal/licenses/', 'extra'),
))
def test_collect_core_validation_checks_runtime_license_tree(tmp_path, prefix, change):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='core')
    )
    license_path = prefix + 'runtime/fixture/LICENSE'
    if change == 'missing':
        rewrite_archive(archive, omit=(license_path,))
    elif change == 'changed':
        rewrite_archive(archive, replacements={license_path: b'changed license'})
    else:
        rewrite_archive(archive, additions={
            prefix + 'runtime/unlisted/COPYING': b'unlisted license',
        })

    message = 'differs from its manifest identity' if change == 'changed' else 'inventory differs'
    with pytest.raises(SystemExit, match=message):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize(('prefix', 'change'), (
    ('ManiaLab/licenses/', 'missing'),
    ('ManiaLab/licenses/', 'changed'),
    ('ManiaLab/_internal/licenses/', 'missing'),
    ('ManiaLab/_internal/licenses/', 'changed'),
))
def test_collect_core_validation_checks_runtime_license_inventory_file(
    tmp_path, prefix, change
):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='core')
    )
    inventory_path = prefix + 'runtime-inventory.json'
    if change == 'missing':
        rewrite_archive(archive, omit=(inventory_path,))
        message = 'missing required component'
    else:
        rewrite_archive(archive, replacements={inventory_path: b'[]'})
        message = 'differs from its manifest identity'

    with pytest.raises(SystemExit, match=message):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize(('prefix', 'name', 'change'), (
    ('ManiaLab/licenses/', 'uv-MIT.txt', 'changed'),
    ('ManiaLab/licenses/', 'FFmpeg-GPLv3.txt', 'missing'),
    ('ManiaLab/_internal/licenses/', 'FFmpeg-build-info.txt', 'changed'),
    ('ManiaLab/_internal/licenses/', 'unlisted-license.txt', 'extra'),
))
def test_collect_core_validation_checks_complete_license_tree(tmp_path, prefix, name, change):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='core')
    )
    license_path = prefix + name
    if change == 'missing':
        rewrite_archive(archive, omit=(license_path,))
        message = 'inventory differs'
    elif change == 'changed':
        rewrite_archive(archive, replacements={license_path: b'changed license payload'})
        message = 'differs from its manifest identity'
    else:
        rewrite_archive(archive, additions={license_path: b'unlisted license payload'})
        message = 'inventory differs'

    with pytest.raises(SystemExit, match=message):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize(('profile', 'field', 'name', 'prefix'), (
    ('core', 'license_files', 'uv-MIT.txt', 'ManiaLab/licenses/'),
    ('core', 'runtime_license_files', 'fixture/LICENSE', 'ManiaLab/licenses/runtime/'),
    ('extended', 'instrument_runtime_license_files', 'inventory.json',
        'ManiaLab/licenses/instrument-runtime/'),
))
def test_collect_rejects_null_manifest_license_identities(tmp_path, profile, field, name, prefix):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile=profile)
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest[field][name] = None
    manifest_bytes = json.dumps(manifest).encode('utf-8')
    manifest_path.write_bytes(manifest_bytes)
    rewrite_archive(archive, replacements={'ManiaLab/release-manifest.json': manifest_bytes})

    with pytest.raises(SystemExit, match=f'invalid component identity under {prefix}'):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize(('left', 'right'), (
    ('ManiaLab/docs', 'ManiaLab/DOCS/'),
    ('ManiaLab/docs', 'ManiaLab/docs/file.txt'),
))
def test_collect_release_validation_rejects_component_case_or_file_directory_collision(
    tmp_path, left, right
):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='core')
    )
    rewrite_archive(archive, additions={left: b'file', right: b''})

    with pytest.raises(SystemExit, match='case-insensitive path collisions'):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize('name', (
    'ManiaLab/stream.txt:payload',
    'ManiaLab/bad<name.txt',
    'ManiaLab/bad>name.txt',
    'ManiaLab/bad"name.txt',
    'ManiaLab/bad|name.txt',
    'ManiaLab/bad?name.txt',
    'ManiaLab/bad*name.txt',
    'ManiaLab/bad\x1fname.txt',
    'ManiaLab/ leading-space.txt',
    'ManiaLab/trailing-dot.',
    'ManiaLab/trailing-space ',
    'ManiaLab/CON.txt',
    'ManiaLab/CONIN$',
    'ManiaLab/CONOUT$',
    'ManiaLab/subdir/aux/README.txt',
    'ManiaLab/LPT9.log',
    'ManiaLab/COM¹.bin',
))
def test_collect_release_validation_rejects_windows_unsafe_path_components(tmp_path, name):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='core')
    )
    rewrite_archive(archive, additions={name: b'unsafe path'})

    with pytest.raises(SystemExit, match='unsafe entry path'):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


def test_windows_archive_component_key_normalizes_case_and_trailing_ascii_dot_space():
    key = collect_release_validation._windows_archive_component_key

    assert key('docs') == key('DOCS') == key('docs.') == key('docs ')


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


@pytest.mark.parametrize(('name', 'content'), (
    ('ManiaLab/_internal/assets/deep/guitar-fl.onnx', b'changed guitar model'),
    ('ManiaLab/_internal/assets/guitar-model.json', b'{}'),
    ('ManiaLab/licenses/HFGuitar-Model-Card.md', b'changed guitar license'),
    ('ManiaLab/_internal/assets/instruments/ManiaLabInstrument/ManiaLabInstrument.exe',
        b'changed instrument worker'),
    ('ManiaLab/_internal/assets/instruments/checkpoints/best_model_bass_v2.pth',
        b'changed bass model'),
    ('ManiaLab/_internal/assets/instruments/source/fixture.py', b'changed instrument source'),
    ('ManiaLab/_internal/assets/instruments/LICENSE-tsumugi', b'changed instrument license'),
    ('ManiaLab/_internal/assets/instruments/verification.json', b'changed verification'),
))
def test_collect_extended_validation_rejects_changed_specialist_zip_payloads(
    tmp_path, name, content
):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='extended')
    )
    rewrite_archive(archive, replacements={name: content})

    with pytest.raises(SystemExit, match='differs from its manifest identity'):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize(('name', 'message'), (
    ('ManiaLab/_internal/assets/deep/guitar-fl.onnx', 'missing required component'),
    ('ManiaLab/_internal/assets/instruments/ManiaLabInstrument/ManiaLabInstrument.exe',
        'component inventory differs'),
))
def test_collect_extended_validation_rejects_missing_specialist_zip_payloads(
    tmp_path, name, message
):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='extended')
    )
    rewrite_archive(archive, omit=(name,))

    with pytest.raises(SystemExit, match=message):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


def test_collect_extended_validation_rejects_unlisted_instrument_zip_payload(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='extended')
    )
    rewrite_archive(archive, additions={
        'ManiaLab/_internal/assets/instruments/ManiaLabInstrument/injected.dll': b'extra',
    })

    with pytest.raises(SystemExit, match='component inventory differs'):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


@pytest.mark.parametrize(('change', 'expected'), (
    ('changed', 'differs from its manifest identity'),
    ('missing', 'component inventory differs'),
    ('extra', 'component inventory differs'),
))
def test_collect_extended_validation_checks_runtime_license_inventory(tmp_path, change, expected):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='extended')
    )
    license_path = 'ManiaLab/licenses/instrument-runtime/torch/LICENSE'
    if change == 'changed':
        rewrite_archive(archive, replacements={license_path: b'changed runtime license'})
    elif change == 'missing':
        rewrite_archive(archive, omit=(license_path,))
    else:
        rewrite_archive(archive, additions={
            'ManiaLab/licenses/instrument-runtime/unlisted/COPYING': b'unlisted',
        })

    with pytest.raises(SystemExit, match=expected):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


def test_archive_inventory_can_require_a_byte_count(tmp_path):
    archive_path = tmp_path / 'inventory.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('licenses/LICENSE', TEST_RUNTIME_LICENSE)

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(SystemExit, match='invalid component identity'):
            collect_release_validation._require_archive_inventory(
                archive,
                'licenses/',
                {'LICENSE': {'sha256': payload_identity(TEST_RUNTIME_LICENSE)}},
                require_bytes=True,
            )


def test_collect_extended_validation_rejects_case_insensitive_specialist_path_collision(
    tmp_path,
):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = (
        make_release_fixture(tmp_path, profile='extended')
    )
    rewrite_archive(archive, additions={
        'ManiaLab/_internal/assets/INSTRUMENTS/ManiaLabInstrument/ManiaLabInstrument.exe':
            TEST_INSTRUMENT_WORKER,
    })

    with pytest.raises(SystemExit, match='case-insensitive path collisions'):
        collect(
            build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence,
        )


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


@pytest.mark.parametrize(
    ('check_name', 'field', 'value'),
    [
        ('deep_analysis', 'candidates', 5.0),
        ('six_stem_analysis_export', 'notes', 5.0),
        ('piano_onnx_runtime', 'output_shapes', [[True, 1001, 88]] * 4 + [[1, 1001, 1]] * 3),
        ('piano_onnx_runtime', 'output_shapes', [[1, 1001.0, 88]] * 4 + [[1, 1001, 1]] * 3),
    ],
)
def test_collect_standard_validation_rejects_numeric_type_substitution(
    tmp_path, check_name, field, value
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='standard'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    record = next(item for item in checks if item['check'] == check_name)
    record[field] = value
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match=f'Portable check payload is invalid: {check_name}'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def _append_optional_piano_audio_evidence(checks, release_reasons):
    identity_fields = (
        'build_profile', 'capabilities', 'executable_sha256',
        'release_manifest_sha256', 'verification_run_id',
    )
    identity = {field: checks[0][field] for field in identity_fields}
    candidate_count = sum(release_reasons.values())
    checks.extend([
        {
            'check': 'piano_cold_inference',
            **identity,
            'input_sha256': 'a' * 64,
            'model_sha256': 'f' * 64,
            'candidates': candidate_count,
            'seconds': 1.0,
            'release_reasons': release_reasons,
        },
        {
            'check': 'piano_manual_expression',
            **identity,
            'modes': {
                'articulation': {'notes': 1, 'ln': 0},
                'sustain': {'notes': 1, 'ln': 1},
            },
            'evidence_preserved': True,
            'timing': 'manual fixture; not musical timing acceptance',
        },
        {
            'check': 'piano_restart_warm_cache',
            **identity,
            'candidates': candidate_count,
            'identical': True,
            'seconds': 0.5,
        },
    ])


def test_collect_standard_validation_accepts_piano_release_reasons_from_producer(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='standard'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    release_reasons = {
        'release_regression': 1,
        'frame_decay': 1,
        'frame_inactive': 1,
        'reattack': 1,
        'duration_limit': 1,
        'input_boundary': 1,
    }
    _append_optional_piano_audio_evidence(checks, release_reasons)
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    report, _, _ = collect(build_root, pytest_report, portable_report, shutdown_report,
        pytest_evidence=pytest_evidence)

    assert report['verification']['piano_audio']['status'] == 'passed'
    assert report['verification']['piano_audio']['musical_acceptance'] is False


def test_collect_standard_validation_rejects_unknown_piano_release_reason(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile='standard'
    )
    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    _append_optional_piano_audio_evidence(checks, {'frame_decay': 1, 'invented': 1})
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Portable check payload is invalid: piano_cold_inference'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def _invalid_capability_mapping(value, invalid_kind):
    malformed = dict(value)
    if invalid_kind == 'numeric':
        return {key: int(flag) for key, flag in malformed.items()}
    if invalid_kind == 'extra_key':
        malformed['unexpected'] = False
        return malformed
    if invalid_kind == 'missing_key':
        malformed.pop(next(iter(malformed)))
        return malformed
    raise AssertionError(f'Unknown invalid capability mapping kind: {invalid_kind}')


@pytest.mark.parametrize(
    ('source', 'profile', 'invalid_kind'),
    [
        ('manifest', 'standard', 'numeric'),
        ('manifest', 'standard', 'extra_key'),
        ('portable', 'standard', 'numeric'),
        ('portable', 'standard', 'missing_key'),
        ('specialist', 'extended', 'numeric'),
        ('specialist', 'extended', 'extra_key'),
        ('shutdown', 'standard', 'numeric'),
        ('shutdown', 'standard', 'missing_key'),
    ],
)
def test_collect_rejects_noncanonical_capability_identity_mapping(
    tmp_path, source, profile, invalid_kind
):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(
        tmp_path, profile=profile
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'

    if source == 'manifest':
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['capabilities'] = _invalid_capability_mapping(
            manifest['capabilities'], invalid_kind
        )
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    elif source in {'portable', 'specialist'}:
        checks = json.loads(portable_report.read_text(encoding='utf-8'))
        target_check = 'release_manifest' if source == 'portable' else 'guitar_real_audio_http'
        row = next(item for item in checks if item['check'] == target_check)
        row['capabilities'] = _invalid_capability_mapping(row['capabilities'], invalid_kind)
        portable_report.write_text(json.dumps(checks), encoding='utf-8')
    else:
        shutdown = json.loads(shutdown_report.read_text(encoding='utf-8'))
        shutdown['capabilities'] = _invalid_capability_mapping(
            shutdown['capabilities'], invalid_kind
        )
        shutdown_report.write_text(json.dumps(shutdown), encoding='utf-8')

    with pytest.raises(SystemExit, match='capabilities|identity differs'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def _rewrite_fixture_manifest_identity(
    build_root, archive_path, portable_report, shutdown_report, manifest
):
    manifest_bytes = json.dumps(manifest).encode('utf-8')
    (build_root / 'dist' / 'ManiaLab' / 'release-manifest.json').write_bytes(manifest_bytes)
    rewrite_archive(archive_path, replacements={'ManiaLab/release-manifest.json': manifest_bytes})
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    for row in checks:
        if 'capabilities' in row:
            row['capabilities'] = manifest['capabilities']
        if 'release_manifest_sha256' in row:
            row['release_manifest_sha256'] = manifest_sha256
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    shutdown = json.loads(shutdown_report.read_text(encoding='utf-8'))
    shutdown['capabilities'] = manifest['capabilities']
    shutdown['release_manifest_sha256'] = manifest_sha256
    shutdown_report.write_text(json.dumps(shutdown), encoding='utf-8')


@pytest.mark.parametrize(
    ('source', 'manifest_bytes', 'portable_bytes'),
    [
        ('manifest_float', 123.0, 123),
        ('manifest_bool', True, True),
        ('manifest_negative', -1, -1),
        ('portable_float', 123, 123.0),
        ('portable_bool', 1, True),
    ],
)
def test_collect_rejects_noninteger_or_negative_piano_model_bytes(
    tmp_path, source, manifest_bytes, portable_bytes
):
    build_root, pytest_report, portable_report, shutdown_report, archive_path, pytest_evidence = (
        make_release_fixture(tmp_path, profile='standard')
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['piano_model']['bytes'] = manifest_bytes
    _rewrite_fixture_manifest_identity(
        build_root, archive_path, portable_report, shutdown_report, manifest
    )

    checks = json.loads(portable_report.read_text(encoding='utf-8'))
    runtime = next(item for item in checks if item['check'] == 'piano_onnx_runtime')
    runtime['bytes'] = portable_bytes
    portable_report.write_text(json.dumps(checks), encoding='utf-8')

    with pytest.raises(SystemExit, match='Piano runtime smoke does not match'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)
