"""Pinned, local CPU Banquet single-window separation research; no app defaults."""
from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import io
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_guitar_vocal_benchmark import read_json,require,sha,verified_hash
from scripts.research_hf_instrument import write_new

SOURCE=ROOT/'test-data/banquet-source'
COMMIT='79ed5bb75e5c3a40cd319d9d990cee913fc65c26'
CHECKPOINT=ROOT/'test-data/banquet-model/ev-pre-aug.ckpt'
CHECKPOINT_SHA='657295888781e62ef50593002720d2edb3858b9e5bbfabf0c54f715a0da4b9e2'
RATE=44100
DEPENDENCIES=('test-data/banquet-runtime','test-data/piano-release-probe/dependencies','test-data/rosvot-runtime')


def verify_assets():
    commit=subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    require(commit==COMMIT,'Banquet source revision differs')
    changes=subprocess.run(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=no'],
                           check=True,capture_output=True,text=True).stdout
    require(not changes.strip(),'Banquet tracked source was modified')
    require(CHECKPOINT.stat().st_size==645470187,'Banquet checkpoint length differs')
    verified_hash(CHECKPOINT,CHECKPOINT_SHA)
    paths=list((SOURCE/'core').rglob('*.py'))+[SOURCE/'config/models/bandit-query.yml']
    return dict(source_commit=commit,source_files={str(p.relative_to(SOURCE)):sha(p) for p in sorted(paths)},
                checkpoint_sha256=CHECKPOINT_SHA,checkpoint_bytes=645470187,
                checkpoint_origin='https://zenodo.org/records/13694558')


def load_model():
    identity=verify_assets()
    sys.path[:0]=[str(SOURCE),*[str(ROOT/name) for name in DEPENDENCIES]]
    import torch
    import yaml
    import hear21passt.base as passt_base
    from unittest.mock import patch
    from core.models.e2e.bandit.bandit import PasstFiLMConditionedBandit
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    require(all(k.startswith('model.') for k in saved['state_dict']), 'Unexpected checkpoint namespace')
    state={k[len('model.'):]:v for k,v in saved['state_dict'].items()}
    config=yaml.safe_load((SOURCE/'config/models/bandit-query.yml').read_text())
    require(config['cls']=='PasstFiLMConditionedBandit' and 'pretrain_encoder' not in config['kwargs'],
            'Inference config would load additional external weights')
    log=io.StringIO()
    # The complete query encoder is in this checkpoint. Suppress the redundant
    # constructor download only, then strictly verify every tensor, including it.
    constructor=functools.partial(passt_base.get_model_passt,pretrained=False)
    with patch.object(passt_base,'get_model_passt',constructor),contextlib.redirect_stdout(log):
        model=PasstFiLMConditionedBandit(**config['kwargs'])
    model.load_state_dict(state,strict=True)
    require(all(torch.equal(value,state[key]) for key,value in model.state_dict().items()),
            'Banquet state differs after strict loading')
    model.eval().requires_grad_(False).cpu()
    identity.update(state_keys=len(state),all_tensors_equal=True,device='cpu',intra_threads=2,inter_threads=1,
        constructor_pretrained_download=False,query_encoder_loaded_from_checkpoint=True,
        versions={name:importlib.metadata.version(name) for name in
                  ('torch','torchaudio','torchvision','hear21passt','timm','pytorch-lightning','torchmetrics','librosa','numpy')},
        adapter_sha256=sha(__file__),constructor_log=log.getvalue(),config=config,
        inference='author model.forward on one 6-second window; no CLI overlap-add',normalization=False)
    return model,identity


def separate(model,mixture,query):
    import numpy as np
    import torch
    from core.types import SimpleishNamespace
    require(mixture.dtype==query.dtype==np.float32 and mixture.shape==(2,6*RATE)
            and query.shape==(2,10*RATE) and np.isfinite(mixture).all() and np.isfinite(query).all(),
            'Expected finite stereo float32 6-second mixture and 10-second query')
    batch=SimpleishNamespace(mixture={'audio':torch.from_numpy(mixture[None].copy())},
                            query={'audio':torch.from_numpy(query[None].copy())},estimates={})
    with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
        output=model(batch).estimates['target'].audio[0].detach().cpu().numpy()
    require(output.shape==mixture.shape and np.isfinite(output).all(),'Invalid separated output')
    return output.copy()
