"""Explicit portable profiles; only declared assets enter the application."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
from scripts.worker_inventory import verify_worker_inventory

CORE_FILES = (
    'basic-pitch.onnx', 'arrangement-profiles.json', 'model-manifest.json',
)
CORE_MODEL = {
    'filename': 'basic-pitch.onnx',
    'version': 'v0.4.0',
    'url': 'https://raw.githubusercontent.com/spotify/basic-pitch/v0.4.0/basic_pitch/saved_models/icassp_2022/nmp.onnx',
    'sha256': '2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec',
}
MODEL_METADATA_FILES = (
    'pattern-prior.json', 'deep-models.json', 'piano-model.json',
    'guitar-model.json',
)
MAPPER_INSTALLER_FILES = (
    'download_mapper.py', 'run_mapper.py', 'mapper_sketch_context.py',
    'mania_event_constraints.py', 'mapperatorinator-requirements.txt',
    'mapperatorinator-source.zip', 'runtime/uv.exe',
)
NON_CORE_PAYLOAD_NAMES = {
    'uv.exe',
    'download_mapper.py',
    'run_mapper.py',
    'mapper_sketch_context.py',
    'mania_event_constraints.py',
    'mapperatorinator-requirements.txt',
    'mapperatorinator-source.zip',
}
DEEP_FILES = (
    'deep/manifest.json', 'deep/htdemucs_6s.th',
    'deep/hdemucs_high_trained.pt', 'deep/beat-this-final0.ckpt', 'deep/piano-regression.onnx',
)
BASE_FILES = CORE_FILES + MODEL_METADATA_FILES + MAPPER_INSTALLER_FILES + DEEP_FILES
PUBLIC_PROFILE = 'core'
PROFILES = {
    PUBLIC_PROFILE: dict(separation_four=False, separation_six=False, piano=False,
                         guitar=False, instruments=False, pattern_prior=False,
                         mapperatorinator_install=False),
    'standard': dict(guitar=False, instruments=False),
    'extended': dict(guitar=True, instruments=True),
}


def add_profile_argument(parser):
    parser.add_argument('--profile', choices=sorted(PROFILES), default=PUBLIC_PROFILE)
    return parser


def capabilities(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown portable profile: ' + profile)
    defaults = dict(audio=True, separation_four=True, separation_six=True, piano=True,
                    pattern_prior=True, mapperatorinator_install=True)
    return defaults | PROFILES[profile]


def verify_core_model_payload(manifest, model_stream):
    """Verify the pinned Basic Pitch identity and the bytes that will ship."""
    expected = {key: value for key, value in CORE_MODEL.items() if key != 'filename'}
    if not isinstance(manifest, dict) or manifest.get('basic_pitch') != expected:
        raise ValueError('Basic Pitch provenance does not match the reviewed model identity')
    digest = hashlib.file_digest(model_stream, 'sha256').hexdigest()
    if digest != CORE_MODEL['sha256']:
        raise ValueError('Basic Pitch model checksum differs from the reviewed model identity')
    return dict(CORE_MODEL)


def verify_core_model_files(asset_root):
    asset_root = Path(asset_root)
    manifest_path = asset_root / 'model-manifest.json'
    model_path = asset_root / CORE_MODEL['filename']
    if not manifest_path.is_file() or not model_path.is_file():
        raise ValueError('Core model or model provenance is missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    with model_path.open('rb') as stream:
        return verify_core_model_payload(manifest, stream)


def archive_stem(version, profile):
    capabilities(profile)
    suffix = '' if profile == 'standard' else f'-{profile}'
    return f'ManiaLab-{version}-windows-x64{suffix}'


def verify_core_payload(files):
    """Check the actual distribution, not just its declared capabilities."""
    prefix = ('_internal', 'assets')
    expected = {('/'.join((*prefix, *PurePosixPath(name).parts))).casefold() for name in CORE_FILES}
    assets = set()
    seen = set()
    for name in files:
        parts = PurePosixPath(name).parts
        lowered = tuple(part.casefold() for part in parts)
        suffix = PurePosixPath(name).suffix.casefold()
        if suffix in {'.pt', '.pth', '.th', '.ckpt', '.safetensors'}:
            raise ValueError(f'Core distribution contains a model checkpoint: {name}')
        if lowered and lowered[-1] in NON_CORE_PAYLOAD_NAMES:
            raise ValueError(f'Core distribution contains an installer or runtime tool: {name}')
        if suffix == '.onnx' and lowered[-3:] != ('_internal', 'assets', 'basic-pitch.onnx'):
            raise ValueError(f'Core distribution contains an excluded ONNX model: {name}')
        if {'manialabaudio', 'manialainstrument',
                'manialabaudio.exe', 'manialainstrument.exe'}.intersection(lowered):
            raise ValueError(f'Core distribution contains an excluded worker: {name}')
        try:
            start = next(index for index in range(len(lowered)-1)
                         if lowered[index:index+2] == prefix)
        except StopIteration:
            continue
        relative_parts = parts[start+2:]
        if not relative_parts or name.endswith('/'):
            continue
        asset_name = '/'.join((*prefix, *relative_parts)).casefold()
        if asset_name in seen:
            raise ValueError(f'Core distribution contains duplicate asset paths: {name}')
        seen.add(asset_name)
        assets.add(asset_name)
    if assets != expected:
        raise ValueError(f'Core asset inventory mismatch: missing={sorted(expected-assets)}, '
                         f'extra={sorted(assets-expected)}')


def stage_assets(root, destination, profile, *, uv_path=None):
    """Fail before copying if a declared component is absent; never sweep assets/."""
    enabled = capabilities(profile)
    source = Path(root) / 'assets'
    def input_path(name):
        if name == 'runtime/uv.exe' and uv_path is not None:
            return Path(uv_path)
        if name == 'deep/piano-regression.onnx':
            from manialab.piano import piano_model_path
            return piano_model_path(root)
        return source / name
    files = list(CORE_FILES)
    directories = []
    if enabled['pattern_prior']:
        files.extend(MODEL_METADATA_FILES)
    if enabled['mapperatorinator_install']:
        files.extend(MAPPER_INSTALLER_FILES)
    if enabled['separation_four']:
        files.extend(DEEP_FILES)
        directories.append('deep/ManiaLabAudio')
    if enabled['guitar']:
        files.append('deep/guitar-fl.onnx')
    if enabled['instruments']:
        from manialab.instrument_bundle import MODELS, source_manifest
        files.extend('instruments/' + name for name in ('manifest.json', 'verification.json', 'LICENSE-tsumugi'))
        files.extend('instruments/checkpoints/best_model_' + name + '.pth' for name in MODELS)
        files.extend('instruments/source/' + name for name in source_manifest()['source_files'])
        directories.append('instruments/ManiaLabInstrument')
    for name in files:
        if not input_path(name).is_file():
            raise ValueError('Missing release component: assets/' + name)
    for name in directories:
        if not (source / name).is_dir():
            raise ValueError('Missing release directory: assets/' + name)
    core_model = verify_core_model_files(source)
    inventories = {}
    for name in directories:
        manifest = json.loads((source / name).with_name('manifest.json').read_text(encoding='utf-8'))
        inventories[name] = verify_worker_inventory(source / name, manifest.get('worker_files'))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path(name), target)
    for name in directories:
        for relative in inventories[name]:
            target = destination / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name / relative, target)
        verify_worker_inventory(destination / name, inventories[name])
    if verify_core_model_files(destination) != core_model:
        raise ValueError('Staged Basic Pitch model differs from the verified source')
    return enabled


def start_guide(version, enabled):
    components = ['混音／导入声轨通用采音']
    if enabled['separation_four']:
        components.append('四／六声部分离、节拍模型')
    if enabled['piano']:
        components.append('钢琴专用采音（默认关闭）')
    if enabled['guitar']:
        components.append('吉他专用采音（默认关闭，模型音尾另行选择）')
    if enabled['instruments']:
        components.append('贝斯／其他伴奏 Tsumugi 专用采音（默认关闭，模型音尾另行选择）')
    missing = []
    if not enabled['separation_four']:
        missing.append('内置四／六声部分离与节拍模型')
    if not enabled['piano']:
        missing.append('钢琴专用模型')
    if not enabled['guitar']:
        missing.append('吉他 FL 专用模型')
    if not enabled['instruments']:
        missing.append('Tsumugi 贝斯／其他伴奏组件')
    return (f'ManiaLab {version}\n完整解压后双击 ManiaLab.exe。关闭控制台窗口退出。\n'
        '本地功能无需安装 Python、Docker 或 CUDA。\n随包组件：' + '；'.join(components) + '。\n'
        + ('本包未包含：' + '、'.join(missing) + '；界面会禁用相应专用选项。\n' if missing else '')
        + '导入音频 → 分析 → 检查 BPM / Offset 并试听 → 设置段落表达 → 生成 → 下载 .osz。\n'
        '新项目制谱提前量默认 22.5ms，可改为 0。\n'
        '项目及日志：%LOCALAPPDATA%\\ManiaLab\\\n'
        '这是开发测试版，音乐与 Timing 尚未完成验收；完整落线不代表采音正确。\n'
        '组件范围以 release-manifest.json 的 capabilities 为准，使用说明见 docs/quick-start.md。\n')
