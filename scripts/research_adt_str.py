"""Frozen official ADT_STR inference on a local stem; no chart enters inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha
from manialab.percussion_tokens import audit_tokens


def token_audit(tokens):
    """Audit the documented triples before trusting a permissive zip decoder."""
    return audit_tokens(tokens)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path('test-data/adt-str/source'))
    cli.add_argument('--model',type=Path,default=Path('test-data/adt-str/model'))
    cli.add_argument('--audio',type=Path,default=Path('test-data/sakura-benchmark/bundled/drums/preview.wav'))
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--threads',type=int,default=2)
    cli.add_argument('--grammar',action='store_true',help='Experimental event grammar; fixed audio and model logits')
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve completed or failed probes')
    commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if commit!='77dbef225e7029478ebfb916f20c7a00274f0f12':raise ValueError('Source revision changed')
    source=json.loads((args.model/'source.json').read_text(encoding='utf-8'))
    for row in source['files']:
        if sha(args.model/row['file'])!=row['sha256']:raise ValueError('Checkpoint/config changed')
    sys.path.insert(0,str(args.source.resolve()))
    import numpy as np
    import torch
    import torchaudio
    import transformers
    from scipy.io import wavfile
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from model import ADTModel
    from config import ADTModelConfig
    from modules.midi_tokenizer import MidiTokenizer,MidiTokenizerConfig
    from utils.mapping_utils import MappingUtils
    torch.set_num_threads(args.threads);torch.manual_seed(42)
    cfg=OmegaConf.to_container(OmegaConf.load(args.model/'adt_config.yaml'),resolve=True)
    model=ADTModel(ADTModelConfig(**json.loads((args.model/'config.json').read_text())))
    model.load_state_dict(load_file(str(args.model/'model.safetensors')),strict=True);model.eval()
    tokenizer=MidiTokenizer(MidiTokenizerConfig(**cfg['tokenizer']))
    sr,pcm=wavfile.read(args.audio)
    if pcm.ndim!=1 or pcm.dtype not in (np.int16,np.float32) or not np.isfinite(pcm).all():
        raise ValueError('Expected finite prepared mono PCM int16 or float32')
    pcm_scale=32768 if pcm.dtype==np.int16 else 1
    audio=torchaudio.functional.resample(torch.from_numpy(pcm.astype(np.float32)/pcm_scale),sr,cfg['shared']['sample_rate'])
    rate=cfg['shared']['sample_rate'];size=round(cfg['shared']['input_sec']*rate)
    count=(len(audio)+size-1)//size
    chunks=torch.nn.functional.pad(audio,(0,count*size-len(audio))).reshape(count,size)
    identity=dict(source_commit=commit,checkpoint=source,audio_sha256=sha(args.audio),runner_sha256=sha(__file__),
                  torch=torch.__version__,torchaudio=torchaudio.__version__,transformers=transformers.__version__,
                  batch_size=2,max_length=cfg['inference']['max_length'],strict_weights=True,sample_rate=rate,chunk_samples=size,
                  pcm_dtype=str(pcm.dtype),pcm_divisor=pcm_scale,
                  grammar=args.grammar,grammar_sha256=sha('manialab/percussion_tokens.py') if args.grammar else None,
                  constrained_decoder_sha256=sha('scripts/adt_constrained_decode.py') if args.grammar else None)
    args.output.mkdir(parents=True)
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    import shutil
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    names=MappingUtils().GM_reduced_name_convention
    tokens_saved={};rows=[];audits=[];began=time.monotonic()
    with torch.inference_mode():
        for start in range(0,count,2):
            if args.grammar:
                from scripts.adt_constrained_decode import sample,TokenLimitError
                maxima=[min(255,int(np.ceil(min(size,len(audio)-i*size)/rate*100))-1) for i in range(start,min(count,start+2))]
                try:
                    predicted=sample(model,chunks[start:start+2],cfg['inference']['max_length'],
                                     tokenizer.BOS_token,tokenizer.EOS_token,maxima)
                except TokenLimitError as error:
                    np.savez_compressed(args.output/'failure-tokens.npz',
                        **{str(i):row.cpu().numpy() for i,row in enumerate(error.tokens,start)})
                    failure=dict(error=str(error),start_chunk=start,elapsed_seconds=time.monotonic()-began,
                        tokens_sha256=sha(args.output/'failure-tokens.npz'),completed_chunks=len(audits))
                    (args.output/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
                    raise
            else:
                predicted=model.sample(chunks[start:start+2],None,None,max_length=cfg['inference']['max_length'],
                                       start_token=tokenizer.BOS_token,end_token=tokenizer.EOS_token)
            for index,tokens in enumerate(predicted,start):
                saved=tokens.numpy();tokens_saved[str(index)]=saved
                issues=token_audit(saved)
                decoded=tokenizer.decode(saved)
                if decoded.numel() and (decoded.ndim!=2 or decoded.shape[1]!=4 or not torch.isfinite(decoded).all()):
                    raise ValueError('Invalid decoded drum events')
                if args.grammar:
                    from manialab.percussion_tokens import decode_complete
                    exact=decode_complete(saved)
                    if len(exact)!=len(decoded):raise ValueError('Official and independent event counts disagree')
                    for a,b in zip(exact,decoded.tolist()):
                        if abs(a['onset']-b[0])>1e-6 or (a['custom_pitch'],a['velocity'])!=(int(b[2]),int(b[3])):
                            raise ValueError('Official decoder misassociated an event')
                for onset,offset,pitch,velocity in decoded.tolist():
                    rows.append(dict(onset=onset+index*size/rate,offset=offset+index*size/rate,
                        custom_pitch=int(pitch),instrument=names.get(int(pitch),'unknown'),velocity=velocity,
                        chunk=index,offset_source='fixed_100ms_placeholder'))
                audits.append(dict(chunk=index,tokens=len(saved),issues=issues,decoded=len(decoded)))
            np.savez_compressed(args.output/'partial-tokens.npz',**tokens_saved)
            (args.output/'partial-chunks.json').write_text(json.dumps(audits),encoding='utf-8')
            print(json.dumps(dict(completed=min(count,start+2),total=count,elapsed_seconds=round(time.monotonic()-began,1))),flush=True)
    # Preserve the official unique-row postprocess and report duplicates.
    unique={tuple(r[k] for k in ('onset','offset','custom_pitch','velocity')):r for r in rows}
    events=sorted(unique.values(),key=lambda n:(n['onset'],n['custom_pitch'],n['velocity']))
    (args.output/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'tokens.npz',**tokens_saved)
    report=dict(identity=identity,duration_ms=len(audio)/rate*1000,elapsed_seconds=time.monotonic()-began,
        events=len(events),duplicate_rows=len(rows)-len(events),counts=dict(Counter(n['instrument'] for n in events)),
        chunks=audits,token_issues=dict(Counter(v for r in audits for v in r['issues'])),
        events_sha256=sha(args.output/'events.json'),tokens_sha256=sha(args.output/'tokens.npz'),enabled_in_application=False,
        notice='Custom 26-class pitches, not GM IDs. Fixed 100ms offsets are placeholders and must never become LN evidence.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('identity','chunks')}),flush=True)


if __name__=='__main__':main()
