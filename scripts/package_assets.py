"""Explicit portable profiles; only declared assets enter the application."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
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
    'huggingface/models--openai--whisper-base/refs/main',
    'huggingface/models--openai--whisper-base/snapshots/e37978b90ca9030d5170a5c07aadb050351a65bb/config.json',
    'huggingface/models--openai--whisper-small/refs/main',
    'huggingface/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/config.json',
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
AUDIO_WORKER_PROFILES = ('four', 'six', 'four-piano')
BASE_FILES = CORE_FILES + MODEL_METADATA_FILES + MAPPER_INSTALLER_FILES + DEEP_FILES
PUBLIC_PROFILE = 'core'
_LORA_REQUIRED_FILES = ('adapter_config.json', 'adapter_model.safetensors')
_LORA_OPTIONAL_FILES = ('lora-manifest.json',)
PROFILES = {
    PUBLIC_PROFILE: dict(separation_four=False, separation_six=False, piano=False,
                         guitar=False, instruments=False, pattern_prior=False,
                         mapperatorinator_install=False, lora_support=False),
    'standard': dict(guitar=False, instruments=False),
    'extended': dict(guitar=True, instruments=True),
    'workbench': dict(separation_four=False, separation_six=False, piano=False,
                      guitar=False, instruments=False, pattern_prior=False,
                      mapperatorinator_install=True, lora_support=True),
}


def add_profile_argument(parser):
    parser.add_argument('--profile', choices=sorted(PROFILES), default=PUBLIC_PROFILE)
    return parser


def capabilities(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown portable profile: ' + profile)
    defaults = dict(audio=True, separation_four=True, separation_six=True, piano=True,
                    pattern_prior=True, mapperatorinator_install=True, lora_support=False)
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


def lora_adapter_files(lora_root):
    """List only complete, portable LoRA adapters; never sweep arbitrary user files."""
    root = Path(lora_root)
    if root.is_symlink() or root.is_junction():
        raise ValueError('LoRA asset root must be a real directory')
    if not root.exists():
        return ()
    if not root.is_dir():
        raise ValueError('LoRA asset root must be a real directory')
    adapters = []
    for adapter in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if adapter.is_symlink() or adapter.is_junction():
            raise ValueError(f'LoRA asset directory must not contain links: {adapter.name}')
        if not adapter.is_dir():
            if adapter.name != '.keep':
                continue
            continue
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', adapter.name):
            raise ValueError(f'LoRA adapter directory name is invalid: {adapter.name}')
        config = adapter / _LORA_REQUIRED_FILES[0]
        weights = adapter / _LORA_REQUIRED_FILES[1]
        if not config.exists() and not weights.exists():
            continue
        if not config.is_file() or not weights.is_file():
            raise ValueError(
                f'LoRA adapter {adapter.name} must include adapter_config.json and '
                'adapter_model.safetensors'
            )
        if config.is_symlink() or weights.is_symlink():
            raise ValueError(f'LoRA adapter {adapter.name} must not contain linked files')
        try:
            configuration = json.loads(config.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f'LoRA adapter {adapter.name} has invalid adapter_config.json') from error
        if not isinstance(configuration, dict) or weights.stat().st_size <= 0:
            raise ValueError(f'LoRA adapter {adapter.name} is incomplete')
        files = [config, weights]
        for name in _LORA_OPTIONAL_FILES:
            optional = adapter / name
            if optional.exists():
                if not optional.is_file() or optional.is_symlink():
                    raise ValueError(f'LoRA adapter {adapter.name} contains an invalid {name}')
                try:
                    if not isinstance(json.loads(optional.read_text(encoding='utf-8')), dict):
                        raise ValueError(f'LoRA adapter {adapter.name} has invalid {name}')
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f'LoRA adapter {adapter.name} has invalid {name}') from error
                files.append(optional)
        adapters.extend(files)
    return tuple(adapters)


def lora_adapter_inventory(lora_root):
    """Return a deterministic, content-addressed inventory for local adapters."""
    root = Path(lora_root)
    inventory = {}
    for source in lora_adapter_files(root):
        relative = source.relative_to(root).as_posix()
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        inventory[relative] = {'bytes': source.stat().st_size, 'sha256': digest}
    return dict(sorted(inventory.items()))


def archive_stem(version, profile):
    capabilities(profile)
    suffix = '' if profile == 'standard' else f'-{profile}'
    return f'ManiaLab-{version}-windows-x64{suffix}'


def checksum_filename(version, profile):
    stem = archive_stem(version, profile)
    return 'SHA256SUMS.txt' if profile == 'workbench' else f'{stem}-SHA256SUMS.txt'


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
    lora_source = source / 'lora'
    lora_files = lora_adapter_files(lora_source) if enabled['lora_support'] else ()
    inventories = {}
    for name in directories:
        manifest = json.loads((source / name).with_name('manifest.json').read_text(encoding='utf-8'))
        if name == 'deep/ManiaLabAudio':
            declared_profiles = manifest.get('profiles')
            if (not isinstance(declared_profiles, list)
                    or not all(profile in declared_profiles for profile in AUDIO_WORKER_PROFILES)):
                raise ValueError('Audio worker manifest is missing a supported profile')
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
    if enabled['lora_support']:
        lora_destination = destination / 'lora'
        lora_destination.mkdir(parents=True, exist_ok=True)
        for source_path in lora_files:
            relative = source_path.relative_to(lora_source)
            target = lora_destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
        if not lora_files:
            (lora_destination / '.keep').write_text(
                'Local LoRA adapters may be placed in named subdirectories.\n',
                encoding='ascii',
            )
        if lora_adapter_inventory(lora_destination) != lora_adapter_inventory(lora_source):
            raise ValueError('Staged LoRA adapters differ from the verified source')
    if verify_core_model_files(destination) != core_model:
        raise ValueError('Staged Basic Pitch model differs from the verified source')
    return enabled


def start_guide(version, enabled):
    if enabled.get('lora_support'):
        return (
            f'ManiaLab {version} 工作台版\n'
            '1. 完整解压文件夹，双击 ManiaLab.exe；无需另装 Python、Docker 或 CUDA。\n'
            '2. 导入歌曲（批量最多 20 首）并检查采音与 BPM / Offset；批量 V32 会逐首串行保存进度，'
            '刷新后可从最近批次继续未完成的歌曲。\n'
            '3. 选择 6–10K 与切指偏米 rice_stream、叠键偏米 rice_jack、'
            '综合随歌 hybrid_adaptive、精致 LN ln_focus 或已有谱 Torosu 改造 torosu_remap；'
            '可在精致 LN 预设中选择已扫描到的 LoRA。\n'
            '4. 检查竖屏下落预览与键型摘要，逐首导出 .osz 或下载整套谱包 ZIP。\n'
            '首次使用 V32 时按提示准备运行环境；NVIDIA GPU 可自动启用 CUDA，'
            '无 GPU 时使用 CPU。V32 模型权重按需下载，FFmpeg 与 Torosu CLI 已随包提供。\n'
            '本机 LoRA 放在 assets/lora/<名称>/，目录需含 adapter_config.json 与 '
            'adapter_model.safetensors；自定义 LoRA 可在界面选择。\n'
            '项目及日志：%LOCALAPPDATA%\\ManiaLab\\。结构检查和自动指标不替代真实歌曲试听与 osu! 实打。\n'
            '版本能力、组件与 LoRA 清单见 release-manifest.json。\n'
        )
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
