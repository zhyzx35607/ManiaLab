"""Create a separate, traceable LoRA-strength ablation; never overwrite training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--scale',type=float,required=True)
    args=cli.parse_args()
    if not 0<args.scale<1:raise ValueError('This ablation requires 0 < scale < 1')
    if args.output.exists():raise ValueError('Use a new output directory')
    from safetensors.torch import load_file,save_file
    source=args.source/'adapter_model.safetensors'
    weights=load_file(str(source));changed=[]
    for name in weights:
        if '.lora_B.' in name:
            weights[name]=weights[name]*args.scale;changed.append(name)
    if not changed:raise ValueError('No LoRA B matrices found')
    args.output.mkdir(parents=True)
    for file in args.source.glob('*.json'):shutil.copyfile(file,args.output/file.name)
    save_file(weights,str(args.output/source.name),metadata={'format':'pt'})
    report=dict(scale=args.scale,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scaled_matrices=len(changed),
        notice='Reduced adapter contribution; no additional training and no quality acceptance implied.')
    (args.output/'strength-ablation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
