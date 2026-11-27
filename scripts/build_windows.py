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
    MAPPER_INSTALLER_FILES, add_profile_argument, archive_stem, checksum_filename, capabilities,
    lora_adapter_inventory, stage_assets, start_guide, verify_core_model_files,
)
from worker_inventory import worker_inventory
parser=argparse.ArgumentParser(description=__doc__)
add_profile_argument(parser)
profile=parser.parse_args().profile
enabled=capabilities(profile)
from build_provenance import (
    archive_payload_inventory,
    sha256,
    torosu_cli_binary,
    remove_duplicate_parent_torch_cpu_dll,
    source_snapshot_paths,
    download_uv_binary,
    verify_worker,
    verify_instrument_worker,
    verify_guitar_model,
    verify_piano_model,
    verify_uv_binary,
)
os.chdir(ROOT)
if os.name!='nt':raise SystemExit('Build the Windows distribution on Windows.')
try:
    torosu_cli=torosu_cli_binary(ROOT)
except FileNotFoundError as error:
    raise SystemExit(str(error)) from error
torosu_cli_identity=dict(bytes=torosu_cli.stat().st_size,sha256=sha256(torosu_cli))

def stage_license_inputs(source, destination, enabled=None):
    source = Path(source)
    destination = Path(destination)
    enabled = enabled or {}
    destination.mkdir(parents=True, exist_ok=False)
    for entry in source.iterdir():
        if entry.name in {'runtime', 'runtime-inventory.json'}:
            continue
        if entry.name == 'deep-runtime' and not enabled.get('separation_four', False):
            continue
        if entry.name == 'instrument-runtime' and not enabled.get('instruments', False):
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        elif entry.is_file():
            shutil.copy2(entry, target)


def stage_runtime_licenses(license_root, distributions, python_license_source=None):
    license_root = Path(license_root)
    runtime_directory = license_root / 'runtime'
    runtime_directory.mkdir()
    inventory = []
    for dist in distributions:
        name = dist.metadata['Name']
        inventory.append({
            'name': name,
            'version': dist.version,
            'license': dist.metadata.get('License-Expression') or dist.metadata.get('License')
                or 'See license files',
        })
        for entry in dist.files or []:
            if any(part.lower().startswith(('license', 'copying', 'notice', 'authors'))
                   for part in Path(str(entry)).parts):
                source = Path(dist.locate_file(entry))
                if source.is_file():
                    target = runtime_directory / name / str(entry).replace('../', 'parent/').replace('..\\', 'parent/')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)

    inventory_path = license_root / 'runtime-inventory.json'
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    if python_license_source is not None and Path(python_license_source).is_file():
        shutil.copy2(python_license_source, runtime_directory / 'Python-LICENSE.txt')

    runtime_files = worker_inventory(runtime_directory)
    inventory_bytes = inventory_path.read_bytes()
    runtime_inventory = {
        'bytes': len(inventory_bytes),
        'sha256': hashlib.sha256(inventory_bytes).hexdigest(),
    }
    return runtime_files, runtime_inventory


def source_snapshot():
    # uv.exe is fetched separately from its pinned upstream release.
    root = ROOT.resolve()
    snapshot = {}
    for path in source_snapshot_paths(root, enabled):
        relative = path.relative_to(root).as_posix()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f'Source snapshot path escapes build root: {relative}')
        snapshot[relative] = sha256(resolved)
    return snapshot
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
build_id=os.environ.get('MANIALAB_BUILD_ID') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}'
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',build_id):
    raise SystemExit('MANIALAB_BUILD_ID must be a simple path component.')
build_root=ROOT/'build'/'portable'/build_id
try:
    build_root.mkdir(parents=True,exist_ok=False)
except FileExistsError:
    raise SystemExit(f'Build output already exists; choose a new MANIALAB_BUILD_ID: {build_root}')
uv_binary=None
uv_identity=None
if enabled['mapperatorinator_install']:
    uv_binary=build_root/'inputs'/'uv.exe'
    uv_identity=download_uv_binary(uv_binary)
license_stage=build_root/'licenses'
stage_license_inputs(ROOT/'licenses', license_stage, enabled)
runtime_license_files, runtime_license_inventory = stage_runtime_licenses(
    license_stage,
    metadata.distributions(),
    next((Path(sys.base_prefix) / name for name in ['LICENSE.txt', 'LICENSE']
          if (Path(sys.base_prefix) / name).is_file()), None),
)
if enabled['mapperatorinator_install']:
    for suffix in ['MIT','APACHE']:
        target=license_stage/f'uv-{suffix}.txt'
        if not target.exists():target.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/astral-sh/uv/0.12.1/LICENSE-{suffix}').read())
gpl=license_stage/'FFmpeg-GPLv3.txt'
if not gpl.exists():gpl.write_bytes(urllib.request.urlopen('https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/COPYING.GPLv3').read())
import imageio_ffmpeg
info=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-version'],capture_output=True,text=True,check=True).stdout
(license_stage/'FFmpeg-build-info.txt').write_text(info+'\nBinary provider: imageio-ffmpeg 0.8.1\nBuild distributor and source/build references: https://www.gyan.dev/ffmpeg/builds/\nFFmpeg source: https://ffmpeg.org/releases/\nBuild scripts: https://github.com/GyanD/codexffmpeg\n',encoding='utf-8')
license_files=worker_inventory(license_stage)
workpath=build_root/'work'
distpath=build_root/'dist'
specpath=build_root/'spec'
package_path=build_root/'release'
for directory in (workpath,distpath,specpath,package_path):
    directory.mkdir()
staged_assets=build_root/'assets'
stage_assets(ROOT,staged_assets,profile,uv_path=uv_binary)
lora_source_inventory = (
    lora_adapter_inventory(staged_assets/'lora')
    if enabled.get('lora_support') else {}
)
if uv_identity and verify_uv_binary(staged_assets/'runtime/uv.exe',check_version=False)!=uv_identity:
    raise SystemExit('Staged uv executable differs from its pinned release identity.')
args=[sys.executable,'-m','PyInstaller','--noconfirm','--name','ManiaLab','--console',
      '--add-data',f'{ROOT / "static"};static',
      '--add-data',f'{staged_assets};assets',
      '--add-data',f'{license_stage};licenses',
      '--add-binary',f'{torosu_cli};engines/torosu_cli',
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
bundled_torosu_cli=target/'_internal/engines/torosu_cli/torosu-cli.exe'
if (not bundled_torosu_cli.is_file()
        or bundled_torosu_cli.stat().st_size!=torosu_cli_identity['bytes']
        or sha256(bundled_torosu_cli)!=torosu_cli_identity['sha256']):
    raise SystemExit('Bundled Torosu CLI differs from the verified native build input.')
bundled_worker=verify_worker(ROOT,target/'_internal/assets/deep') if enabled['separation_four'] else None
if bundled_worker!=worker_identity:
    raise SystemExit('Audio worker changed during the build; rebuild before publishing.')
if enabled['separation_four']:
    runtime_cleanup=remove_duplicate_parent_torch_cpu_dll(target)
    print('Portable runtime deduplication: '+json.dumps(runtime_cleanup,sort_keys=True))
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
bundled_uv_identity=(
    verify_uv_binary(target/'_internal/assets/runtime/uv.exe',check_version=False)
    if enabled['mapperatorinator_install'] else None
)
if bundled_uv_identity!=uv_identity:
    raise SystemExit('Bundled uv executable differs from its pinned release identity.')
if worker_inventory(license_stage)!=license_files:
    raise SystemExit('License files changed during the build; rebuild before publishing.')
instrument_runtime_license_files=(
    worker_inventory(ROOT/'licenses/instrument-runtime') if enabled['instruments'] else None)
for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md']:
    shutil.copy2(ROOT/name,target/name)
shutil.copytree(license_stage,target/'licenses',dirs_exist_ok=True)
shutil.copytree(ROOT/'docs',target/'docs',dirs_exist_ok=True)
guide = start_guide(__version__,enabled)
(target/'使用说明.txt').write_text(guide,encoding='utf-8-sig')
(target/'开始使用.txt').write_text(guide,encoding='utf-8-sig')
if source_snapshot()!=build_sources:
    raise SystemExit('Application sources changed while staging the portable package; rebuild before publishing.')
archive_payload=archive_payload_inventory(target)
lora_inventory = (
    lora_adapter_inventory(target/'_internal/assets/lora')
    if enabled.get('lora_support') else {}
)
if lora_inventory != lora_source_inventory:
    raise SystemExit('Packaged LoRA adapters differ from the verified workbench inputs.')
release_manifest=dict(version=__version__,sources=build_sources,build_profile=profile,capabilities=enabled,
    executable_sha256=hashlib.sha256((target/'ManiaLab.exe').read_bytes()).hexdigest(),
    torosu_cli=dict(path='_internal/engines/torosu_cli/torosu-cli.exe',**torosu_cli_identity),
    core_models=bundled_core_model_identity,
    uv_runtime=bundled_uv_identity,
    worker_sha256=bundled_worker.get('worker_sha256') if bundled_worker else None,
    worker_sources=bundled_worker.get('worker_sources') if bundled_worker else None,
    worker_files=bundled_worker.get('worker_files') if bundled_worker else None,
    piano_model=piano_identity,guitar_model=guitar_identity,instrument_worker=instrument_identity,
    lora_adapters=lora_inventory,
    instrument_runtime_license_files=instrument_runtime_license_files,
    runtime_license_files=runtime_license_files,
    runtime_license_inventory=runtime_license_inventory,
    license_files=license_files,archive_payload=archive_payload)
(target/'release-manifest.json').write_text(json.dumps(release_manifest,indent=2),encoding='utf-8')
name=archive_stem(__version__,profile)
archive=Path(shutil.make_archive(str(package_path/name),'zip',distpath,'ManiaLab'))
with archive.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
checksums=package_path/checksum_filename(__version__,profile)
checksums.write_text(f'{digest}  {archive.name}\n',encoding='utf-8')
print(f'Portable directory: {target}\nBuilt: {archive}\nSHA256 file: {checksums}\nSHA256: {digest}')
