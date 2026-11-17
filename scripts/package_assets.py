"""Explicit portable profiles; only declared assets enter the application."""
from pathlib import Path
import json
import shutil
from scripts.worker_inventory import verify_worker_inventory

BASE_FILES = (
    'basic-pitch.onnx', 'pattern-prior.json', 'arrangement-profiles.json',
    'model-manifest.json', 'deep-models.json', 'piano-model.json', 'guitar-model.json',
    'download_mapper.py', 'run_mapper.py', 'mapper_sketch_context.py',
    'mania_event_constraints.py', 'mapperatorinator-requirements.txt', 'mapperatorinator-source.zip',
    'runtime/uv.exe', 'deep/manifest.json', 'deep/htdemucs_6s.th',
    'deep/hdemucs_high_trained.pt', 'deep/beat-this-final0.ckpt', 'deep/piano-regression.onnx',
)
PROFILES = {
    'standard': dict(guitar=False, instruments=False),
    'extended': dict(guitar=True, instruments=True),
}


def capabilities(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown portable profile: ' + profile)
    return dict(audio=True, separation_four=True, separation_six=True, piano=True, **PROFILES[profile])


def stage_assets(root, destination, profile):
    """Fail before copying if a declared component is absent; never sweep assets/."""
    enabled = capabilities(profile)
    source = Path(root) / 'assets'
    from manialab.piano import piano_model_path
    def input_path(name):
        return piano_model_path(root) if name == 'deep/piano-regression.onnx' else source / name
    files = list(BASE_FILES)
    directories = ['deep/ManiaLabAudio']
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
    return enabled


def start_guide(version, enabled):
    components = ['通用音高采音、四／六声部分离、节拍分析', '钢琴专用采音（默认关闭）']
    if enabled['guitar']:
        components.append('吉他专用采音（默认关闭，模型音尾另行选择）')
    if enabled['instruments']:
        components.append('贝斯／其他伴奏 Tsumugi 专用采音（默认关闭，模型音尾另行选择）')
    missing = []
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
