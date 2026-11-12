"""Build a portable directory and ZIP using the current locked Python environment."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab import __version__
from build_provenance import sha256, verify_worker, verify_instrument_worker, verify_guitar_model
os.chdir(ROOT)
if os.name!='nt':raise SystemExit('Build the Windows distribution on Windows.')
def source_snapshot():
    files=[ROOT/'launcher.py',*sorted((ROOT/'manialab').glob('*.py')),*sorted((ROOT/'assets').glob('*.py')),
           *sorted(p for p in (ROOT/'static').rglob('*') if p.is_file()),
           ROOT/'assets/pattern-prior.json',ROOT/'assets/arrangement-profiles.json',
           ROOT/'assets/piano-model.json',ROOT/'assets/guitar-model.json',ROOT/'assets/deep-models.json',ROOT/'manialab/tsumugi-source.json',
           *sorted((ROOT/'licenses').glob('HFGuitar-*'))]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
build_sources=source_snapshot()
required=['basic-pitch.onnx','pattern-prior.json','arrangement-profiles.json','deep/ManiaLabAudio/ManiaLabAudio.exe','deep/htdemucs_6s.th',
          'deep/hdemucs_high_trained.pt','deep/beat-this-final0.ckpt','deep/manifest.json']
for relative in required:
    if not (ROOT/'assets'/relative).is_file():raise SystemExit(f'Missing release component: assets/{relative}')
worker_identity=verify_worker(ROOT,ROOT/'assets/deep')
instrument_identity=None
if (ROOT/'assets/instruments').exists():
    instrument_identity=verify_instrument_worker(ROOT,ROOT/'assets/instruments')
guitar_identity=verify_guitar_model(ROOT)
optional_piano=ROOT/'assets/deep/piano-regression.onnx'
piano_identity=None
if optional_piano.exists():
    piano_identity=json.loads((ROOT/'assets/piano-model.json').read_text(encoding='utf-8'))
    with optional_piano.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=piano_identity['sha256']:
            raise SystemExit('Unexpected piano model checksum')
    if not (ROOT/piano_identity['license_file']).is_file():
        raise SystemExit('Missing piano model attribution/license')
for model in json.loads((ROOT/'assets/deep-models.json').read_text(encoding='utf-8'))['models']:
    with (ROOT/'assets/deep'/model['filename']).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=model['sha256']:
            raise SystemExit('Unexpected model checksum: '+model['name'])
uv=shutil.which('uv')
if not uv:raise SystemExit('Install uv first to bundle the optional engine installer.')
(ROOT/'assets/runtime').mkdir(exist_ok=True)
shutil.copy2(uv,ROOT/'assets/runtime/uv.exe')
lic=ROOT/'licenses/runtime';lic.mkdir(parents=True,exist_ok=True)
inventory=[]
for dist in metadata.distributions():
    name=dist.metadata['Name'];version=dist.version
    inventory.append({'name':name,'version':version,'license':dist.metadata.get('License-Expression') or dist.metadata.get('License') or 'See license files'})
    for entry in dist.files or []:
        if any(part.lower().startswith(('license','copying','notice','authors')) for part in Path(str(entry)).parts):
            source=Path(dist.locate_file(entry))
            if source.is_file():
                target=lic/name/str(entry).replace('../','parent/').replace('..\\','parent/')
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
(ROOT/'licenses/runtime-inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['LICENSE.txt','LICENSE']:
    source=Path(sys.base_prefix)/name
    if source.exists():shutil.copy2(source,lic/'Python-LICENSE.txt');break
for suffix in ['MIT','APACHE']:
    target=ROOT/f'licenses/uv-{suffix}.txt'
    if not target.exists():target.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/astral-sh/uv/0.12.1/LICENSE-{suffix}').read())
gpl=ROOT/'licenses/FFmpeg-GPLv3.txt'
if not gpl.exists():gpl.write_bytes(urllib.request.urlopen('https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/COPYING.GPLv3').read())
import imageio_ffmpeg
info=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-version'],capture_output=True,text=True,check=True).stdout
(ROOT/'licenses/FFmpeg-build-info.txt').write_text(info+'\nBinary provider: imageio-ffmpeg 0.8.1\nBuild distributor and source/build references: https://www.gyan.dev/ffmpeg/builds/\nFFmpeg source: https://ffmpeg.org/releases/\nBuild scripts: https://github.com/GyanD/codexffmpeg\n',encoding='utf-8')
build_id=os.environ.get('MANIALAB_BUILD_ID') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}'
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',build_id):
    raise SystemExit('MANIALAB_BUILD_ID must be a simple path component.')
build_root=ROOT/'build'/'portable'/build_id
try:
    build_root.mkdir(parents=True,exist_ok=False)
except FileExistsError:
    raise SystemExit(f'Build output already exists; choose a new MANIALAB_BUILD_ID: {build_root}')
workpath=build_root/'work'
distpath=build_root/'dist'
specpath=build_root/'spec'
package_path=build_root/'release'
for directory in (workpath,distpath,specpath,package_path):
    directory.mkdir()
args=[sys.executable,'-m','PyInstaller','--noconfirm','--name','ManiaLab','--console',
      '--add-data',f'{ROOT / "static"};static',
      '--add-data',f'{ROOT / "assets"};assets',
      '--add-data',f'{ROOT / "licenses"};licenses',
      '--add-data',f'{ROOT / "manialab/tsumugi-source.json"};manialab',
      '--add-data',f'{ROOT / "vendor/mapperatorinator/template.osu"};vendor/mapperatorinator',
      '--copy-metadata','numpy','--copy-metadata','scipy','--copy-metadata','onnxruntime',
      '--collect-all','imageio_ffmpeg','--collect-binaries','onnxruntime','--exclude-module','pytest',
      '--workpath',str(workpath),'--distpath',str(distpath),'--specpath',str(specpath),
      str(ROOT / 'launcher.py')]
subprocess.run(args,check=True)
target=distpath/'ManiaLab'
if source_snapshot()!=build_sources:
    raise SystemExit('Application sources changed during the build; rebuild before publishing.')
bundled_worker=verify_worker(ROOT,target/'_internal/assets/deep')
if bundled_worker!=worker_identity:
    raise SystemExit('Audio worker changed during the build; rebuild before publishing.')
if piano_identity and sha256(target/'_internal/assets/deep/piano-regression.onnx')!=piano_identity['sha256']:
    raise SystemExit('Piano model changed during the build; rebuild before publishing.')
if instrument_identity and verify_instrument_worker(ROOT,target/'_internal/assets/instruments')!=instrument_identity:
    raise SystemExit('Instrument worker changed during the build; rebuild before publishing.')
if verify_guitar_model(target/'_internal')!=guitar_identity:
    raise SystemExit('Guitar model or attribution changed during the build; rebuild before publishing.')
release_manifest=dict(version=__version__,sources=build_sources,
    executable_sha256=hashlib.sha256((target/'ManiaLab.exe').read_bytes()).hexdigest(),
    worker_sha256=bundled_worker['worker_sha256'],worker_sources=bundled_worker['worker_sources'],
    piano_model=piano_identity,guitar_model=guitar_identity,instrument_worker=instrument_identity)
(target/'release-manifest.json').write_text(json.dumps(release_manifest,indent=2),encoding='utf-8')
for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md']:
    shutil.copy2(ROOT/name,target/name)
shutil.copytree(ROOT/'licenses',target/'licenses',dirs_exist_ok=True)
shutil.copytree(ROOT/'docs',target/'docs',dirs_exist_ok=True)
(target/'开始使用.txt').write_text(f'ManiaLab {__version__}\n完整解压后双击 ManiaLab.exe。\n程序会自动打开浏览器；关闭控制台窗口退出。\n深度采音模型已随包提供，无需安装 Python、Docker、CUDA 或下载模型。\n导入音频 → 深度分析 → 检查 BPM / Offset 并试听 → 检查段落及主声部表达 → 保存方案 → 生成 → 下载 .osz。\n旧项目可点击“检测音乐段落”使用新流程；若要应用低音重复触发修正，请重新分析，模型缓存可复用。\n新项目制谱提前量默认 22.5ms，可改为 0 使用原始音频时间。\n项目数据：%LOCALAPPDATA%\\ManiaLab\\projects\n这是开发测试版；采音与段落建议仍需试听，尚非端到端的 10K 制谱模型。可勾选贝斯与其他伴奏专用采音；默认只用起音，允许模型音尾参与长条需另行勾选。吉他有独立的专用采音和模型音尾开关，需六声部分离或导入吉他轨；人声保持原路径。使用方式见 docs/instrument-choices.md。未通过对照的采音取舍研究权重未启用。\n',encoding='utf-8-sig')
name=f'ManiaLab-{__version__}-windows-x64'
archive=Path(shutil.make_archive(str(package_path/name),'zip',distpath,'ManiaLab'))
with archive.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
checksums=package_path/f'{name}-SHA256SUMS.txt'
checksums.write_text(f'{digest}  {archive.name}\n',encoding='utf-8')
print(f'Portable directory: {target}\nBuilt: {archive}\nSHA256 file: {checksums}\nSHA256: {digest}')
