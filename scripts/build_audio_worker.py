"""Build the portable audio worker in an isolated requirements-deep environment."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if os.name != "nt":
    raise SystemExit("Build the Windows worker on Windows")
import torch
import torchaudio

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'assets/deep')
output = parser.parse_args().output.resolve()
output.mkdir(parents=True, exist_ok=True)
source_model = Path(os.environ.get("MANIALAB_DEEP_MODEL", ROOT / "assets/deep/hdemucs_high_trained.pt"))
expected = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
if hashlib.sha256(source_model.read_bytes()).hexdigest() != expected:
    raise SystemExit("Unexpected separation model checksum")
if source_model.resolve() != (output / "hdemucs_high_trained.pt").resolve():
    shutil.copy2(source_model, output / "hdemucs_high_trained.pt")
beat_model = ROOT / "assets/deep/beat-this-final0.ckpt"
beat_expected = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
if hashlib.sha256(beat_model.read_bytes()).hexdigest() != beat_expected:
    raise SystemExit("Unexpected beat model checksum")
if beat_model.resolve() != (output / beat_model.name).resolve():
    shutil.copy2(beat_model, output / beat_model.name)
args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--name", "ManiaLabAudio", "--console",
    "--distpath", str(output), "--workpath", "build/deep", "--specpath", "build",
    "--collect-binaries", "torchaudio"]
for module in ("transformers", "matplotlib", "librosa", "pandas", "pytest", "tensorflow", "tensorboard", "cv2", "PIL", "IPython", "sklearn",
               "pyarrow", "datasets", "pydantic", "pydantic_core", "safetensors", "huggingface_hub", "aiohttp"):
    args += ["--exclude-module", module]
args += ["scripts/separate_audio.py"]
subprocess.run(args, check=True)
licenses = ROOT / "licenses/deep-runtime"
licenses.mkdir(parents=True, exist_ok=True)
inventory = []
for name in ("torch", "torchaudio", "numpy", "scipy", "sympy", "mpmath", "networkx", "jinja2", "markupsafe", "filelock", "fsspec", "typing_extensions",
             "beat-this", "einops", "rotary-embedding-torch", "soxr"):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    inventory.append(dict(name=name, version=dist.version))
    for entry in dist.files or ():
        if any(part.lower().startswith(("license", "copying", "notice", "authors")) for part in Path(str(entry)).parts):
            source = Path(dist.locate_file(entry))
            if source.is_file():
                relative = str(entry).replace("../", "parent/").replace("..\\", "parent/")
                destination = licenses / name / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
(licenses / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
manifest = dict(model="Hybrid Demucs MUSDB+", model_sha256=expected,
    model_url="https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt",
    torch=torch.__version__, torchaudio=torchaudio.__version__, runtime=sys.version,
    beat_model="Beat This! final0", beat_model_sha256=beat_expected,
    beat_model_url="https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt",
    sources=["drums", "bass", "other", "vocals"], sample_rate=44100)
(output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest), flush=True)
