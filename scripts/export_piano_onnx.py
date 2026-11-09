"""Research export of the fixed piano graph; numerical checks precede any use."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_piano_acoustics import verify_specialist, CHECKPOINT_SHA256, SOURCE_COMMIT
from scripts.probe_mapper_adapter import sha


def main():
    root=Path('test-data/piano-release-probe')
    verify_specialist(root)
    output=Path('test-data/piano-onnx/export-v1')
    if output.exists():raise ValueError('Preserve earlier export')
    sys.path[:0]=[str((root/'source').resolve()),str((root/'dependencies').resolve())]
    # Keep the active research NumPy/Torch runtime; only discover ONNX here.
    sys.path.append(str(Path('test-data/piano-onnx/export-dependencies').resolve()))
    os.environ.update(TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',MPLBACKEND='Agg')
    import torch
    import onnx
    from piano_transcription_inference import PianoTranscription
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    upstream=PianoTranscription(checkpoint_path=str(root/'model.pth'),device='cpu')
    checkpoint=torch.load(root/'model.pth',map_location='cpu',weights_only=True)
    upstream.model.load_state_dict(checkpoint['model'],strict=True);upstream.model.eval()
    del checkpoint
    keys=['reg_onset_output','reg_offset_output','frame_output','velocity_output',
          'reg_pedal_onset_output','reg_pedal_offset_output','pedal_frame_output']
    class Outputs(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,audio):
            result=self.model(audio)
            return tuple(result[key] for key in keys)
    model=Outputs(upstream.model).eval()
    manifest=json.loads(Path('test-data/smd/selected-v1/benchmark/manifest.json').read_text())
    source=Path(manifest['cases'][0]['audio'])
    rate,samples=wavfile.read(source)
    audio=resample_poly(samples[:rate*10].astype(np.float32)/32768,320,441).astype(np.float32)
    input=torch.from_numpy(audio[None])
    output.mkdir(parents=True)
    np.save(output/'input.npy',audio,allow_pickle=False)
    begin=time.monotonic()
    with torch.inference_mode():
        reference=model(input)
    np.savez_compressed(output/'torch-reference.npz',**{k:v.numpy() for k,v in zip(keys,reference)})
    torch.onnx.export(model,input,str(output/'piano.onnx'),input_names=['audio'],output_names=keys,
                      dynamo=False,opset_version=17)
    onnx.checker.check_model(str(output/'piano.onnx'))
    report=dict(checkpoint_sha256=CHECKPOINT_SHA256,source_commit=SOURCE_COMMIT,
                graph_sha256=sha(output/'piano.onnx'),input_sha256=sha(output/'input.npy'),
                reference_sha256=sha(output/'torch-reference.npz'),source_audio_sha256=sha(source),
                input_samples=160000,sample_rate=16000,output_names=keys,
                torch=torch.__version__,onnx=onnx.__version__,numpy=np.__version__,
                elapsed_seconds=time.monotonic()-begin,enabled_in_application=False,
                runner_sha256=sha(__file__),notice='Export and checker only; independent runtime numerical verification required.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
