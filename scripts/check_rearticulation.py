"""Negative control: repeated gated chords must not become continuous holds."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.audio import analyze
from manialab.arrange import parameters
from manialab.domain import note_from
from scripts.research_pitch_continuity import alternatives


if __name__=='__main__':
    root=Path('test-data/continuity-rearticulation');root.mkdir(exist_ok=True)
    rate=22050;data=np.zeros(rate*12,dtype=np.float32)
    for group in (0,5):
        for i in range(8):
            start=.115+group+i*.5;length=.47;t=np.arange(round(length*rate))/rate
            tone=sum(np.sin(2*np.pi*(440*2**((pitch-69)/12))*t) for pitch in (60,64,67))
            envelope=np.minimum(1,t/.005)*np.minimum(1,(length-t)/.005)
            lo=round(start*rate);data[lo:lo+len(t)]+=.2*tone*envelope
    wavfile.write(root/'repeated.wav',rate,(data*32767).astype(np.int16))
    analysis=analyze(root/'repeated.wav',root,Path('assets/basic-pitch.onnx'),parameters())
    notes=[note_from(n) for n in analysis['candidates'] if n['source']=='pitch']
    with np.load(root/'activations.npz') as values:
        extended=alternatives(values['frames'],values['times'],notes)
    result=dict(real_chord_attacks=16,real_pitched_events=48,real_duration_ms=470,
        silence_between_attacks_ms=30,model_pitched_events=len(notes),false_long_reinterpretations=len(extended),
        examples=[dict(start=n.start,end=n.end,pitch=n.pitch) for n in extended.values()])
    (root/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
