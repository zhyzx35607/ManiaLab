import ast
import argparse
from pathlib import Path
import json
import re
import shutil

import pytest

from scripts.package_assets import (
    BASE_FILES,
    CORE_FILES,
    MAPPER_INSTALLER_FILES,
    PUBLIC_PROFILE,
    add_profile_argument,
    capabilities,
    checksum_filename,
    lora_adapter_inventory,
    stage_assets,
    start_guide,
    verify_core_model_files,
    verify_core_payload,
)
from scripts.worker_inventory import worker_inventory

ROOT = Path(__file__).resolve().parents[1]
PORTABLE_PR_PROFILE = (
    "${{ (startsWith(github.ref, 'refs/tags/v0.2.') || "
    "(github.event_name == 'pull_request' && "
    "contains(github.event.pull_request.title, '[portable]'))) && "
    "'workbench' || 'core' }}"
)


def _workflow_uses_public_profile(workflow):
    lines = workflow.splitlines()

    def block_end(start, indent):
        for index in range(start + 1, len(lines)):
            stripped = lines[index].lstrip(" ")
            if stripped and not stripped.startswith("#"):
                if len(lines[index]) - len(stripped) <= indent:
                    return index
        return len(lines)

    def child_block(parent_start, parent_end, parent_indent, key):
        indent = parent_indent + 2
        candidates = []
        for index in range(parent_start + 1, parent_end):
            line = lines[index]
            if line.startswith(" " * indent):
                rest = line[indent:]
                if re.match(rf"(?:{re.escape(key)}|['\"]{re.escape(key)}['\"])[ \t]*:", rest) or \
                        re.fullmatch(rf"\?[ \t]*(?:{re.escape(key)}|['\"]{re.escape(key)}['\"])[ \t]*", rest):
                    candidates.append(index)

        if len(candidates) != 1:
            return None
        index = candidates[0]
        if not re.fullmatch(rf"{' ' * indent}{re.escape(key)}:[ \t]*(?:#.*)?", lines[index]):
            return None
        return index

    jobs = [index for index, line in enumerate(lines) if re.fullmatch(r"jobs:[ \t]*(?:#.*)?", line)]
    if len(jobs) != 1:
        return False
    jobs_start = jobs[0]
    verify = child_block(jobs_start, block_end(jobs_start, 0), 0, "verify")
    if verify is None:
        return False
    env = child_block(verify, block_end(verify, 2), 2, "env")
    if env is None:
        return False

    env_end = block_end(env, 4)
    profile_count = 0
    for line in lines[env + 1:env_end]:
        if not line.strip() or line.lstrip(" ").startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent] or indent != 6:
            return False
        if "PORTABLE_PROFILE" in line:
            match = re.fullmatch(r" {6}PORTABLE_PROFILE:[ \t]*(.*)", line)
            if match is None:
                return False
            profile_count += 1
            if re.fullmatch(
                re.escape(PORTABLE_PR_PROFILE) + r"(?:[ \t]+#.*)?[ \t]*",
                match.group(1),
            ) is None:
                return False

    return profile_count == 1


def test_portable_pr_builds_workbench_while_other_events_keep_core():
    workflow = (ROOT / '.github/workflows/windows.yml').read_text(encoding='utf-8')
    assert _workflow_uses_public_profile(workflow)
    assert (
        "contains(github.event.pull_request.title, '[portable]')"
        ")) && 'workbench' || 'core'"
    ) in workflow


def test_v020_version_tags_select_the_workbench_profile():
    workflow = (ROOT / '.github/workflows/windows.yml').read_text(encoding='utf-8')
    assert _workflow_uses_public_profile(workflow)
    assert "startsWith(github.ref, 'refs/tags/v0.2.')" in workflow


def test_marked_portable_pr_runs_workbench_verifiers_without_heavy_assets_or_legacy_collector():
    workflow = (ROOT / '.github/workflows/windows.yml').read_text(encoding='utf-8')
    light_profiles = "contains(fromJSON('[\"standard\",\"extended\"]'), env.PORTABLE_PROFILE)"
    heavy_steps = (
        'Fetch verified audio models',
        'Install isolated audio environment',
        'Build and verify the fixed piano model',
        'Build portable audio worker',
    )
    for name in heavy_steps:
        start = workflow.index(f'- name: {name}')
        next_step = workflow.find('\n      - name:', start + 1)
        block = workflow[start: next_step if next_step >= 0 else len(workflow)]
        assert light_profiles in block

    assert '- name: Build declared Windows profile' in workflow
    assert '- name: Verify actual EXE offline with clean PATH and Unicode data paths' in workflow
    assert '- name: Verify portable process and worker shutdown behavior' in workflow
    collector_start = workflow.index('- name: Collect release verification evidence')
    collector_end = workflow.find('\n      - name:', collector_start + 1)
    collector = workflow[collector_start:collector_end]
    assert "contains(fromJSON('[\"core\",\"standard\",\"extended\"]'), env.PORTABLE_PROFILE)" in collector
    assert 'env.PORTABLE_PROFILE != \'workbench\'' not in collector


@pytest.mark.parametrize(('workflow', 'valid'), [
    (f'PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}\n', True),
    (f'PORTABLE_PROFILE: {PORTABLE_PR_PROFILE} # profile policy\n', True),
    ('PORTABLE_PROFILE: core\n', False),
    ('PORTABLE_PROFILE: workbench\n', False),
    (f'PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}#suffix\n', False),
    (f'PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}\n  workbench\n', False),
    (f'PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}\u00a0\n', False),
])
def test_workflow_profile_must_be_one_exact_policy_scalar(workflow, valid):
    wrapped = f'jobs:\n  verify:\n    env:\n      {workflow.rstrip(chr(10)).replace(chr(10), chr(10) + "      ")}\n'
    assert _workflow_uses_public_profile(wrapped) is valid


@pytest.mark.parametrize('env_block', [
    f'      PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}\n      PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}\n',
    f'      PORTABLE_PROFILE : {PUBLIC_PROFILE}\n',
    f"      'PORTABLE_PROFILE': {PUBLIC_PROFILE}\n",
    f'      "PORTABLE_PROFILE" : {PUBLIC_PROFILE}\n',
    f'      ? PORTABLE_PROFILE\n      : {PUBLIC_PROFILE}\n',
    f'      OTHER: {{PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}}}\n',
    f'      PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}\n        workbench\n',
])
def test_workflow_profile_guard_rejects_ambiguous_yaml_declarations(env_block):
    workflow = f'jobs:\n  verify:\n    env:\n{env_block}'
    assert not _workflow_uses_public_profile(workflow)


def test_workflow_profile_guard_rejects_flow_env_mapping():
    workflow = f'jobs:\n  verify:\n    env: {{PORTABLE_PROFILE: {PORTABLE_PR_PROFILE}}}\n'
    assert not _workflow_uses_public_profile(workflow)


def _stage_license_inputs_function():
    source = ast.parse((ROOT / 'scripts/build_windows.py').read_text(encoding='utf-8'))
    function = next(node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == 'stage_license_inputs')
    namespace = {'Path': Path, 'shutil': shutil}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'license-staging-only', 'exec'), namespace)
    return namespace['stage_license_inputs']


def assets(root):
    for name in BASE_FILES:
        target = root / 'assets' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in ('basic-pitch.onnx', 'model-manifest.json'):
            target.write_bytes((ROOT / 'assets' / name).read_bytes())
        else:
            target.write_bytes(name.encode())
    (root / 'assets/deep/ManiaLabAudio').mkdir()
    (root / 'assets/deep/ManiaLabAudio/ManiaLabAudio.exe').write_bytes(b'worker')
    (root / 'assets/deep/manifest.json').write_text(json.dumps(dict(
        profiles=['four','six','four-piano'],
        worker_files=worker_inventory(root / 'assets/deep/ManiaLabAudio'))), encoding='utf-8')
    return root / 'assets'


def test_standard_profile_excludes_undeclared_local_assets(tmp_path):
    source = assets(tmp_path)
    (source / 'deep/guitar-fl.onnx').write_bytes(b'guitar')
    (source / 'instruments').mkdir()
    (source / 'instruments/private.wav').write_bytes(b'user audio')
    (source / 'deep/piano-regression-historical.onnx').write_bytes(b'old graph')
    (source / 'unrelated.zip').write_bytes(b'research archive')
    destination = tmp_path / 'staged'
    enabled = stage_assets(tmp_path, destination, 'standard')
    assert enabled == capabilities('standard')
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(BASE_FILES) | {'deep/ManiaLabAudio/ManiaLabAudio.exe'}
    guide = start_guide('test', enabled)
    assert '本包未包含：吉他 FL 专用模型、Tsumugi' in guide
    assert '可勾选贝斯' not in guide


def test_audio_worker_manifest_must_declare_piano_focused_profile(tmp_path):
    source = assets(tmp_path)
    manifest_path = source / 'deep/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['profiles'] = ['four', 'six']
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(ValueError, match='missing a supported profile'):
        stage_assets(tmp_path, tmp_path / 'staged', 'standard')
    assert not (tmp_path / 'staged').exists()


def test_core_profile_excludes_weights_and_workers(tmp_path):
    assets(tmp_path)
    destination = tmp_path / 'staged-core'
    enabled = stage_assets(tmp_path, destination, 'core')
    assert enabled == capabilities('core')
    assert enabled == dict(audio=True, separation_four=False, separation_six=False,
                           piano=False, guitar=False, instruments=False,
                           pattern_prior=False, mapperatorinator_install=False,
                           lora_support=False)
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(CORE_FILES)
    guide = start_guide('test', enabled)
    assert '本包未包含：内置四／六声部分离与节拍模型' in guide
    assert '钢琴专用模型' in guide


def test_core_payload_rejects_models_and_accepts_exact_inventory():
    names = ['ManiaLab/_internal/assets/' + name for name in CORE_FILES]
    verify_core_payload(names)
    with pytest.raises(ValueError, match='model checkpoint'):
        verify_core_payload(names + ['ManiaLab/_internal/assets/deep/htdemucs_6s.th'])
    with pytest.raises(ValueError, match='inventory mismatch'):
        verify_core_payload(names + ['ManiaLab/_internal/assets/unlisted-private.wav'])
    with pytest.raises(ValueError, match='installer or runtime tool'):
        verify_core_payload(names + ['ManiaLab/_internal/uv.exe'])


def test_core_model_must_match_pinned_provenance_and_hash(tmp_path):
    source = assets(tmp_path)
    assert verify_core_model_files(source)['sha256'] == \
        '2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec'
    (source / 'basic-pitch.onnx').write_bytes(b'unexpected model')
    destination = tmp_path / 'staged-core'
    with pytest.raises(ValueError, match='checksum'):
        stage_assets(tmp_path, destination, 'core')
    assert not destination.exists()


def test_extended_profile_requires_components_before_staging(tmp_path):
    source = assets(tmp_path)
    destination = tmp_path / 'staged'
    with pytest.raises(ValueError, match='guitar'):
        stage_assets(tmp_path, destination, 'extended')
    assert not destination.exists()
    (source / 'deep/guitar-fl.onnx').write_bytes(b'guitar')
    with pytest.raises(ValueError, match='instruments'):
        stage_assets(tmp_path, destination, 'extended')
    assert not destination.exists()
    (source / 'instruments').mkdir()
    from manialab.instrument_bundle import MODELS, source_manifest
    instrument_files = ['manifest.json', 'verification.json', 'LICENSE-tsumugi']
    instrument_files += ['checkpoints/best_model_' + name + '.pth' for name in MODELS]
    instrument_files += ['source/' + name for name in source_manifest()['source_files']]
    for name in instrument_files:
        target = source / 'instruments' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'fixture')
    worker_root = source / 'instruments/ManiaLabInstrument'
    worker_files = ('ManiaLabInstrument.exe', 'runtime/worker-data.bin')
    for name in worker_files:
        target = worker_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'worker fixture')
    frozen_worker_inventory = worker_inventory(worker_root)
    (source / 'instruments/manifest.json').write_text(json.dumps(dict(
        worker_files=frozen_worker_inventory)), encoding='utf-8')

    (source / 'deep/unlisted-specialist.onnx').write_bytes(b'unlisted model')
    (source / 'instruments/private.wav').write_bytes(b'user audio')
    (source / 'instruments/source/unlisted.py').write_bytes(b'unlisted source')
    (source / 'unrelated.zip').write_bytes(b'research archive')

    enabled = stage_assets(tmp_path, destination, 'extended')
    staged_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob('*')
        if path.is_file()
    }
    expected_files = (
        set(BASE_FILES)
        | {'deep/ManiaLabAudio/ManiaLabAudio.exe'}
        | {'deep/guitar-fl.onnx'}
        | {'instruments/' + name for name in instrument_files}
        | {
            'instruments/ManiaLabInstrument/' + name
            for name in frozen_worker_inventory
        }
    )
    assert staged_files == expected_files
    assert '本包未包含' not in start_guide('test', enabled)


def test_workbench_profile_is_lightweight_but_includes_v32_installer_and_lora_support(tmp_path):
    source = assets(tmp_path)
    lora = source / 'lora' / 'ln-focus'
    lora.mkdir(parents=True)
    (lora / 'adapter_config.json').write_text(
        json.dumps({'r': 16, 'lora_alpha': 32}), encoding='utf-8',
    )
    (lora / 'adapter_model.safetensors').write_bytes(b'portable lora weights')
    (lora / 'lora-manifest.json').write_text(
        json.dumps({'base_model': {'variant': 'v32-mini'}}), encoding='utf-8',
    )

    destination = tmp_path / 'staged-workbench'
    enabled = stage_assets(tmp_path, destination, 'workbench')
    staged_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob('*')
        if path.is_file()
    }

    assert enabled == capabilities('workbench')
    assert enabled['mapperatorinator_install'] is True
    assert enabled['lora_support'] is True
    assert enabled['separation_four'] is False
    assert enabled['piano'] is False
    assert staged_files == (
        set(CORE_FILES)
        | set(MAPPER_INSTALLER_FILES)
        | {
            'lora/ln-focus/adapter_config.json',
            'lora/ln-focus/adapter_model.safetensors',
            'lora/ln-focus/lora-manifest.json',
        }
    )
    assert lora_adapter_inventory(destination / 'lora') == lora_adapter_inventory(source / 'lora')
    guide = start_guide('test', enabled)
    for required in ('最多 20 首', 'rice_stream', 'rice_jack', 'hybrid_adaptive',
                     'ln_focus', 'torosu_remap', 'adapter_model.safetensors'):
        assert required in guide


def test_workbench_profile_creates_empty_scan_directory_without_local_loras(tmp_path):
    source = assets(tmp_path)

    destination = tmp_path / 'staged-workbench'
    stage_assets(tmp_path, destination, 'workbench')

    assert (destination / 'lora' / '.keep').is_file()
    assert lora_adapter_inventory(destination / 'lora') == {}
    assert not (destination / 'deep').exists()
    assert not (destination / 'instruments').exists()


def test_workbench_checksum_has_stable_name_without_changing_other_profiles():
    assert checksum_filename('0.2.0-beta.1', 'workbench') == 'SHA256SUMS.txt'
    assert checksum_filename('0.2.0-beta.1', 'core') == \
        'ManiaLab-0.2.0-beta.1-windows-x64-core-SHA256SUMS.txt'
    assert checksum_filename('0.2.0-beta.1', 'standard') == \
        'ManiaLab-0.2.0-beta.1-windows-x64-SHA256SUMS.txt'


def test_workbench_profile_rejects_incomplete_local_lora_before_staging(tmp_path):
    source = assets(tmp_path)
    adapter = source / 'lora' / 'broken'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    destination = tmp_path / 'staged-workbench'

    with pytest.raises(ValueError, match='adapter_model.safetensors'):
        stage_assets(tmp_path, destination, 'workbench')
    assert not destination.exists()


def test_existing_profiles_do_not_sweep_local_lora_weights(tmp_path):
    source = assets(tmp_path)
    adapter = source / 'lora' / 'private'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (adapter / 'adapter_model.safetensors').write_bytes(b'private weights')

    destination = tmp_path / 'staged-standard'
    stage_assets(tmp_path, destination, 'standard')

    assert not (destination / 'lora' / 'private').exists()


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match='profile'):
        capabilities('automatic')


@pytest.mark.parametrize('profile', ('standard', 'extended', 'workbench'))
def test_build_profile_argument_defaults_public_and_keeps_explicit_internal_profiles(profile):
    parser = argparse.ArgumentParser()
    add_profile_argument(parser)

    assert parser.parse_args([]).profile == PUBLIC_PROFILE == 'core'
    assert parser.parse_args(['--profile', profile]).profile == profile


def test_license_staging_includes_only_runtime_trees_for_enabled_components(tmp_path):
    stage_license_inputs = _stage_license_inputs_function()
    source = tmp_path / 'licenses'
    source.mkdir()
    for name in ('uv-MIT.txt', 'runtime', 'deep-runtime', 'instrument-runtime'):
        path = source / name
        if name.endswith('.txt'):
            path.write_text('tracked license', encoding='utf-8')
        else:
            path.mkdir()
            (path / 'inventory.json').write_text('[]', encoding='utf-8')
    (source / 'runtime-inventory.json').write_text('stale', encoding='utf-8')

    for profile, included in (
        ('core', set()),
        ('standard', {'deep-runtime'}),
        ('extended', {'deep-runtime', 'instrument-runtime'}),
        ('workbench', set()),
    ):
        destination = tmp_path / f'staged-{profile}'
        stage_license_inputs(source, destination, capabilities(profile))
        staged = {path.name for path in destination.iterdir()}
        assert staged == {'uv-MIT.txt'} | included
        assert not (destination / 'runtime-inventory.json').exists()


def test_worker_private_file_rejected_before_staging(tmp_path):
    source = assets(tmp_path)
    (source / 'deep/ManiaLabAudio/private.wav').write_bytes(b'private recording')
    destination = tmp_path / 'staged'
    with pytest.raises(ValueError, match='extra=.*private.wav'):
        stage_assets(tmp_path, destination, 'standard')
    assert not destination.exists()


def test_worker_added_after_inventory_is_not_copied(tmp_path, monkeypatch):
    source = assets(tmp_path)
    import scripts.package_assets as package
    copy = package.shutil.copy2

    def copy_with_new_file(src, dst):
        (source / 'deep/ManiaLabAudio/private.wav').write_bytes(b'late recording')
        return copy(src, dst)

    monkeypatch.setattr(package.shutil, 'copy2', copy_with_new_file)
    destination = tmp_path / 'staged'
    stage_assets(tmp_path, destination, 'standard')
    assert not (destination / 'deep/ManiaLabAudio/private.wav').exists()
