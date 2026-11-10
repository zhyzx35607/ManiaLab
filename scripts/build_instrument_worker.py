"""Build the optional CPU instrument worker and preserve its source identity."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.instrument_bundle import RUNTIME_FILES, MODELS, runtime_identity, sha, verify_model, verify_source


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--assets',type=Path,default=ROOT/'assets/instruments')
    cli.add_argument('--dependencies',type=Path,default=ROOT/'test-data/rosvot-runtime')
    args = cli.parse_args()
    if os.name != 'nt':
        raise ValueError('Build the Windows worker on Windows')
    assets = args.assets.resolve()
    verify_source(assets/'source')
    for model in MODELS:
        verify_model(assets/'checkpoints',model)
    original = runtime_identity(ROOT)
    # Give both PyInstaller and its analysis subprocesses the same frozen
    # dependency location as the research forwards. No model inference here.
    env = os.environ.copy()
    paths = [str(assets/'source'),str(ROOT)]
    if args.dependencies.is_dir():
        paths.append(str(args.dependencies.resolve()))
    env['PYTHONPATH'] = os.pathsep.join(paths)
    command = [sys.executable,'-m','PyInstaller','--noconfirm','--name','ManiaLabInstrument',
        '--console','--distpath',str(assets),'--workpath',str(ROOT/'build/instruments'),
        '--specpath',str(ROOT/'build'),'--collect-binaries','torchaudio']
    for path in paths:
        command += ['--paths',path]
    for name in RUNTIME_FILES:
        destination = 'source-evidence/'+str(Path(name).parent).replace('\\','/')
        command += ['--add-data',str(ROOT/name)+';'+destination]
    command += ['--add-data',str(ROOT/'manialab/tsumugi-source.json')+';manialab']
    packages = ('torch','torchaudio','numpy','scipy','soundfile','einops','pretty-midi','tqdm')
    for name in packages:
        command += ['--copy-metadata',name]
    for name in ('transformers','matplotlib','librosa','pandas','pytest','tensorflow','tensorboard',
                 'cv2','PIL','IPython','sklearn','pyarrow','datasets','pydantic','huggingface_hub','aiohttp'):
        command += ['--exclude-module',name]
    command.append(str(ROOT/'scripts/transcribe_instruments.py'))
    subprocess.run(command,check=True,cwd=ROOT,env=env)
    if original != runtime_identity(ROOT):
        raise ValueError('Instrument runtime sources changed during build; rebuild before use')
    # Manifest records the build, not successful inference. A separate packaged
    # real-audio parity run is required before portable_runtime_verified changes.
    manifest = json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(worker_sources=original, worker_sha256=sha(assets/'ManiaLabInstrument/ManiaLabInstrument.exe'),
        prepared_only=False, portable_runtime_verified=False)
    for path in reversed(paths):
        sys.path.insert(0,path)
    manifest['runtime_packages'] = {name:metadata.version(name) for name in packages}
    (assets/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    licenses = ROOT/'licenses/instrument-runtime'
    licenses.mkdir(parents=True,exist_ok=True)
    inventory = []
    for name in (*packages,'mido','cffi','pycparser','sympy','mpmath','networkx','jinja2','markupsafe',
                 'filelock','fsspec','typing_extensions','packaging','colorama'):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        inventory.append(dict(name=name,version=dist.version))
        for entry in dist.files or ():
            if not any(part.lower().startswith(('license','copying','notice','authors')) for part in Path(str(entry)).parts):
                continue
            source = Path(dist.locate_file(entry))
            if source.is_file():
                relative = str(entry).replace('../','parent/').replace('..\\','parent/')
                target = licenses/name/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
    (licenses/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps(dict(built=True,worker_sha256=manifest['worker_sha256'],portable_runtime_verified=False)),flush=True)


if __name__=='__main__':
    main()
