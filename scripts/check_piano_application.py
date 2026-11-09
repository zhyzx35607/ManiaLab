"""Exercise real provided-piano analysis with verified full-song activations."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters
from manialab.audio import analyze_deep,decode,SR
from manialab.domain import note_from,Note
from manialab.piano import PianoDetector,MODEL_SHA256
from manialab.provided_stems import prepare_provided_stems,ProvidedStemSeparator
from manialab.stems import separation_identity
from scripts.evaluate_sakura_piano import evaluate
from scripts.probe_mapper_adapter import sha


def main():
    root=Path('test-data/sakura-benchmark')
    output=root/'piano-application-v1'
    if output.exists():raise ValueError('Preserve application checks')
    portable=Path('test-data/piano-onnx/portable-v1')
    source=json.loads((portable/'report.json').read_text(encoding='utf-8'))
    model=Path('test-data/piano-onnx/export-v1/piano.onnx')
    if source['identity']['model_sha256']!=MODEL_SHA256 or sha(model)!=MODEL_SHA256:
        raise ValueError('Portable model identity differs')
    full_output=next(row for row in source['cases'] if row['name']=='sakura-nova-piano')
    if sha(root/'raw-pcm-audit-v1/piano.wav')!=full_output['audio_sha256']:
        raise ValueError('Original amplitude input changed')
    output.mkdir()
    mixed=root/'bundled/mix.mp3';piano=root/'bundled/piano.mp3'
    separation=prepare_provided_stems(mixed,{'piano':piano},output)
    separator=ProvidedStemSeparator(separation)
    folder,_=separator.separate(mixed,output)
    samples=np.load(folder/'piano.npy',allow_pickle=False)
    rate,actual=wavfile.read(root/'raw-pcm-audit-v1/piano.wav')
    if rate!=SR or not np.array_equal(samples,actual):
        raise ValueError('Imported signal differs from full portable inference')
    # This is a warm-cache integration check. Source bytes and the complete
    # sample array must match the already completed full network inference.
    identity=dict(revision=1,model_sha256=MODEL_SHA256,stem_identity=separation_identity(separation),
                  pcm_sha256=hashlib.sha256(np.asarray(samples,dtype='<f4').tobytes()).hexdigest(),
                  sample_rate=SR,samples=len(samples))
    previous=portable/'sakura-nova-piano-activations.npz'
    with np.load(previous,allow_pickle=False) as arrays:
        np.savez_compressed(folder/'piano-specialist-activations.npz',identity=json.dumps(identity,sort_keys=True),**dict(arrays))
    detector=PianoDetector(Path.cwd());detector.model=model
    analysis=analyze_deep(mixed,output,Path('assets/basic-pitch.onnx'),parameters(),separator,piano_detector=detector)
    assert analysis['cached'] and analysis['piano_engine']=='specialist'
    (output/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False),encoding='utf-8')
    notes=[note_from(n) for n in analysis['candidates']]
    human=(root/'reference.osu').read_text(encoding='utf-8-sig')
    raw=json.loads((portable/'sakura-nova-piano-notes.json').read_text(encoding='utf-8'))
    raw_notes=[Note(round(n['onset']*1000),round(n['offset']*1000),pitch=n['pitch'],source='pitch') for n in raw]
    report=dict(cached_activation_source_sha256=sha(previous),same_sample_array=True,cold_inference=False,
                raw_candidates=len(raw_notes),application_candidates=len(notes),
                raw_reference_diagnostic=evaluate(human,raw_notes),application_reference_diagnostic=evaluate(human,notes),
                application_filters=analysis['voices'],release_reasons=dict(Counter(n.release_reason for n in notes)),
                auto_timing=analysis['tempo'],runner_sha256=sha(__file__),
                notice='Original Nova piano; warm integration of completed full inference. Reference diagnoses onset selection only, not physical note offsets or piano layers.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('raw_candidates','application_candidates','application_reference_diagnostic','release_reasons')}),flush=True)


if __name__=='__main__':main()
