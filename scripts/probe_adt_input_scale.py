"""Compare a known looping ADT chunk under explicitly different PCM inputs."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    import numpy as np
    import torch
    import torchaudio
    from scipy.io import wavfile
    from safetensors.torch import load_file
    source=Path('test-data/adt-str/source');checkpoint=Path('test-data/adt-str/model')
    sys.path.insert(0,str(source.resolve()))
    from model import ADTModel
    from config import ADTModelConfig
    from manialab.percussion_tokens import decode_complete
    torch.set_num_threads(2);torch.manual_seed(42)
    model=ADTModel(ADTModelConfig(**json.loads((checkpoint/'config.json').read_text())))
    model.load_state_dict(load_file(str(checkpoint/'model.safetensors')),strict=True);model.eval()
    root=Path('test-data/sakura-benchmark');variants={};identities={}
    for name,path in [('legacy_int16',root/'bundled/drums/preview.wav'),('raw_float',root/'raw-pcm-audit-v1/drums.wav')]:
        rate,signal=wavfile.read(path);identities[name]=sha(path)
        signal=signal.astype(np.float32)/(32768 if signal.dtype==np.int16 else 1)
        waveform=torchaudio.functional.resample(torch.from_numpy(signal),rate,24000)
        variants[name]=waveform[4*61440:6*61440].reshape(2,61440)
    variants['raw_float_chunk_peak']=variants['raw_float']/variants['raw_float'].abs().amax(dim=1,keepdim=True).clamp_min(1e-8)
    results={};out=root/'adt-input-scale-diagnostic-v1.json'
    if out.exists():raise ValueError('Preserve prior scale diagnostic')
    with torch.inference_mode():
        for name,frames in variants.items():
            began=time.monotonic();tokens=model.sample(frames,None,None,max_length=64,start_token=2,end_token=3).tolist()
            results[name]=dict(seconds=time.monotonic()-began,tokens=tokens,peaks=frames.abs().amax(dim=1).tolist(),
                eos=[3 in row for row in tokens],unique_triples=[len(set(tuple(row[i:i+3]) for i in range(1,(row.index(3) if 3 in row else len(row))-2,3))) for row in tokens])
            print(json.dumps(dict(variant=name,**results[name])),flush=True)
    out.write_text(json.dumps(dict(audio_hashes=identities,weights_sha256=sha(checkpoint/'model.safetensors'),
        runner_sha256=sha(__file__),results=results,notice='Known failure diagnostic, max 64 tokens; no chart scoring or full transcription claim.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
