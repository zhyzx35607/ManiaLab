"""Copy the pinned MIT instrument source/weights into optional portable assets."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.instrument_bundle import MODELS, MODEL_REVISION, SOURCE_COMMIT, sha, verify_model, verify_source


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi'))
    cli.add_argument('--output',type=Path,default=Path('assets/instruments'))
    args = cli.parse_args()
    files = verify_source(args.source)
    weights = {model:verify_model(args.source/'checkpoints',model) for model in MODELS}
    if args.output.exists():
        raise ValueError('Inspect existing assets before rebuilding; choose a new output folder')
    license_file = args.source/'LICENSE'
    license_text = license_file.read_text(encoding='utf-8')
    if 'MIT License' not in license_text:
        raise ValueError('Unexpected upstream source license')
    args.output.mkdir(parents=True)
    for name in files:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.source/name,target)
    for model,path in weights.items():
        target = args.output/'checkpoints'/path.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    shutil.copyfile(license_file,args.output/'LICENSE-tsumugi')
    verify_source(args.output/'source')
    for model in MODELS:
        verify_model(args.output/'checkpoints',model)
    manifest = dict(source_commit=SOURCE_COMMIT,model_revision=MODEL_REVISION,
        source_files=files,models=MODELS,license='MIT',license_sha256=sha(license_file),
        source_url='https://github.com/anime-song/tsumugi',
        model_url='https://huggingface.co/anime-song/instrument_agnostic_amt',
        voices=['bass','other'],prepared_only=True,portable_runtime_verified=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(prepared=True,voices=manifest['voices'],files=len(files))))


if __name__=='__main__':
    main()
