"""Build a portable directory and ZIP using the current locked Python environment."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
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
from package_assets import (
    MAPPER_INSTALLER_FILES, PROFILES, archive_stem, capabilities, stage_assets, start_guide,
    verify_core_model_files,
)
from worker_inventory import worker_inventory
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile",choices=sorted(PROFILES),default="standard")
profile=parser.parse_args().profile
enabled=capabilities(profile)
from build_provenance import (
    sha256,
    verify_worker,
    verify_instrument_worker,
    verify_guitar_model,
    verify_piano_model,
)
os.chdir(ROOT)
if os.name!='nt':raise SystemExit('Build the Windows distribution on Windows.')
def source_snapshot():
    files=[ROOT/'launcher.py',ROOT/'README.md',ROOT/'LICENSE',ROOT/'THIRD_PARTY_NOTICES.md',
           ROOT/'scripts/build_windows.py',
           ROOT/'scripts/package_assets.py',ROOT/'scripts/worker_inventory.py',
           ROOT/'scripts/build_provenance.py',*sorted((ROOT/'manialab').glob('*.py')),*sorted((ROOT/'assets').glob('*.py')),
           *sorted((ROOT/'vendor').rglob('*.py')),ROOT/'vendor/mapperatorinator/template.osu',
           *sorted(p for p in (ROOT/'static').rglob('*') if p.is_file()),
           ROOT/'assets/pattern-prior.json',ROOT/'assets/arrangement-profiles.json',
           ROOT/'assets/model-manifest.json',ROOT/'assets/piano-model.json',
           ROOT/'assets/guitar-model.json',ROOT/'assets/deep-models.json',ROOT/'manialab/tsumugi-source.json',
           *sorted((ROOT/'licenses').glob('HFGuitar-*'))]
    if enabled['mapperatorinator_install']:
        # uv.exe is copied from the selected runtime later; these are source inputs.
        files += [ROOT/'assets'/name for name in MAPPER_INSTALLER_FILES if name != 'runtime/uv.exe']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
build_sources=source_snapshot()
core_model_identity=verify_core_model_files(ROOT/'assets')
required=['basic-pitch.onnx','arrangement-profiles.json']
if enabled['pattern_prior']:
    required.append('pattern-prior.json')
if enabled['separation_four']:
    required += ['deep/ManiaLabAudio/ManiaLabAudio.exe','deep/htdemucs_6s.th',
                 'deep/hdemucs_high_trained.pt','deep/beat-this-final0.ckpt','deep/manifest.json']
for relative in required:
    if not (ROOT/'assets'/relative).is_file():raise SystemExit(f'Missing release component: assets/{relative}')
piano_identity=verify_piano_model(ROOT) if enabled['piano'] else None
worker_identity=verify_worker(ROOT,ROOT/'assets/deep') if enabled['separation_four'] else None
instrument_identity=None
if enabled['instruments']:
    instrument_identity=verify_instrument_worker(ROOT,ROOT/'assets/instruments')
guitar_identity=verify_guitar_model(ROOT) if enabled['guitar'] else None
if enabled['guitar'] and guitar_identity is None:
    raise SystemExit('Extended profile requires the guitar model.')
if enabled['separation_four']:
    for model in json.loads((ROOT/'assets/deep-models.json').read_text(encoding='utf-8'))['models']:
        with (ROOT/'assets/deep'/model['filename']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=model['sha256']:
                raise SystemExit('Unexpected model checksum: '+model['name'])
uv=shutil.which('uv')
if enabled['mapperatorinator_install']:
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
runtime_license_files=worker_inventory(ROOT/'licenses/runtime')
runtime_license_inventory_path=ROOT/'licenses/runtime-inventory.json'
runtime_license_inventory=dict(
    bytes=runtime_license_inventory_path.stat().st_size,
    sha256=hashlib.sha256(runtime_license_inventory_path.read_bytes()).hexdigest())
if enabled['mapperatorinator_install']:
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
staged_assets=build_root/'assets'
stage_assets(ROOT,staged_assets,profile)
args=[sys.executable,'-m','PyInstaller','--noconfirm','--name','ManiaLab','--console',
      '--add-data',f'{ROOT / "static"};static',
      '--add-data',f'{staged_assets};assets',
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
bundled_worker=verify_worker(ROOT,target/'_internal/assets/deep') if enabled['separation_four'] else None
if bundled_worker!=worker_identity:
    raise SystemExit('Audio worker changed during the build; rebuild before publishing.')
bundled_core_model_identity=verify_core_model_files(target/'_internal/assets')
if bundled_core_model_identity!=core_model_identity:
    raise SystemExit('Basic Pitch model identity changed during the build; rebuild before publishing.')
bundled_piano_identity=verify_piano_model(target/'_internal') if enabled['piano'] else None
if bundled_piano_identity!=piano_identity:
    raise SystemExit('Piano model identity changed during the build; rebuild before publishing.')
if instrument_identity and verify_instrument_worker(ROOT,target/'_internal/assets/instruments')!=instrument_identity:
    raise SystemExit('Instrument worker changed during the build; rebuild before publishing.')
if enabled['guitar'] and verify_guitar_model(target/'_internal')!=guitar_identity:
    raise SystemExit('Guitar model or attribution changed during the build; rebuild before publishing.')
instrument_runtime_license_files=(
    worker_inventory(ROOT/'licenses/instrument-runtime') if enabled['instruments'] else None)
release_manifest=dict(version=__version__,sources=build_sources,build_profile=profile,capabilities=enabled,
    executable_sha256=hashlib.sha256((target/'ManiaLab.exe').read_bytes()).hexdigest(),
    core_models=bundled_core_model_identity,
    worker_sha256=bundled_worker.get('worker_sha256') if bundled_worker else None,
    worker_sources=bundled_worker.get('worker_sources') if bundled_worker else None,
    worker_files=bundled_worker.get('worker_files') if bundled_worker else None,
    piano_model=piano_identity,guitar_model=guitar_identity,instrument_worker=instrument_identity,
    instrument_runtime_license_files=instrument_runtime_license_files,
    runtime_license_files=runtime_license_files,
    runtime_license_inventory=runtime_license_inventory)
(target/'release-manifest.json').write_text(json.dumps(release_manifest,indent=2),encoding='utf-8')
for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md']:
    shutil.copy2(ROOT/name,target/name)
shutil.copytree(ROOT/'licenses',target/'licenses',dirs_exist_ok=True)
shutil.copytree(ROOT/'docs',target/'docs',dirs_exist_ok=True)
(target/'开始使用.txt').write_text(start_guide(__version__,enabled),encoding='utf-8-sig')
name=archive_stem(__version__,profile)
archive=Path(shutil.make_archive(str(package_path/name),'zip',distpath,'ManiaLab'))
with archive.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
checksums=package_path/f'{name}-SHA256SUMS.txt'
checksums.write_text(f'{digest}  {archive.name}\n',encoding='utf-8')
print(f'Portable directory: {target}\nBuilt: {archive}\nSHA256 file: {checksums}\nSHA256: {digest}')
