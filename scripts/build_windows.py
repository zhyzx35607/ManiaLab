"""Build a portable directory and ZIP using the current locked Python environment."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if os.name!='nt':raise SystemExit('Build the Windows distribution on Windows.')
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
args=[sys.executable,'-m','PyInstaller','--noconfirm','--name','ManiaLab','--console',
      '--add-data','static;static','--add-data','assets;assets','--add-data','licenses;licenses',
      '--add-data','vendor/mapperatorinator/template.osu;vendor/mapperatorinator',
      '--collect-all','imageio_ffmpeg','--collect-binaries','onnxruntime','--exclude-module','pytest','launcher.py']
subprocess.run(args,check=True)
target=ROOT/'dist/ManiaLab'
for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md']:
    shutil.copy2(ROOT/name,target/name)
shutil.copytree(ROOT/'licenses',target/'licenses',dirs_exist_ok=True)
shutil.copytree(ROOT/'docs',target/'docs',dirs_exist_ok=True)
(target/'开始使用.txt').write_text('完整解压后双击 ManiaLab.exe。\n程序会自动打开浏览器；关闭控制台窗口退出。\n基础功能无需 Python、Docker 或 GPU。\n项目数据：%LOCALAPPDATA%\\ManiaLab\\projects\n这是实验版，尚未训练专用 10K 模型。\n',encoding='utf-8-sig')
release=ROOT/'release';release.mkdir(exist_ok=True)
name='ManiaLab-0.1.0-alpha.2-windows-x64'
archive=Path(shutil.make_archive(str(release/name),'zip',ROOT/'dist','ManiaLab'))
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(release/'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n',encoding='utf-8')
print(f'Built: {archive}\nSHA256: {digest}')
