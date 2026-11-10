"""Fixed FL float-input research probe; never changes application defaults."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from math import gcd
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_hf_instrument as adapter
from scripts import evaluate_guitar_vocal_benchmark as evaluation
from manialab import guitar
from manialab.piano import decode

read, require, sha, verify = evaluation.read_json, evaluation.require, evaluation.sha, evaluation.verified_hash
MODEL = 'guitar_fl'
GRAPH_SHA = 'e02ee3b540717bcae07678764847f5e078213be12b1cba98400f091019df73f9'
SOURCE_SHA = 'fec9fa8feb07330f9dd51c35deaa5bf8f36c074aec9bece7a0b5c6a8d99136dc'
FLOAT_RATE = 22050
FROZEN_CODE = {
    'scripts/research_hf_instrument.py': 'b734e0806442ce028d1e16acf98e0808d0c96aaa1d123a9dc885d300a4b45bc1',
    'manialab/piano.py': '57855e0888edef8dcc442aa2baee53fa04b97abea410924aa9d9403c55ba9947',
    'scripts/evaluate_guitar_vocal_benchmark.py': '97accf3f1becbb275da10daf7fb2447f6af5bfa1cb6b151dc71711c016088b51',
    'scripts/evaluate_instrument_benchmark.py': 'e5723d8d3da394568a90759b338013d99bd0c8aa8ba93cd78e3fa875ac0d900f',
}
NATIVE = (
    dict(name='sakura-nova-guitar',
         path='test-data/sakura-benchmark/provided-pipeline/stems-provided/'
              '09130f7f521f46d2f856e5860190813d7569c6c06ced4414d9faa49972b03e6e/guitar.npy',
         sha256='b0f42b37b15623492068e4b8021c1cfd0f733298b762b7da68fb8c46226a115f',
         separation_sha256='b51c6ec1f97392f145d8f3579d419631bfb4b6989658a1b8e2f9ba703f431bbf',
         samples=6258672, source='mapper_confirmed_Nova_stem_from_beatmapset_2644803',
         limitation='Extremely weak guitar stem; no independent guitar note labels; development diagnostics only.'),
    dict(name='youdiantian-htdemucs6-guitar',
         path='test-data/nova-reference/youdiantian/stems-six/guitar.npy',
         sha256='1bc609bda3a1da0111dd3642f82eb8f1d612b031f1696c4457a0e11aca03ba5a',
         separation_sha256='a6884713795543d6fc467659ccd32eba1bb4c601335f3cdec316242bbbfc1dc2',
         samples=5186173, source='HTDemucs6_output_of_user_audio_not_Nova',
         limitation='No independent guitar note labels; development diagnostics only.'),
)


def pcm_hash(pcm):
    return hashlib.sha256(np.asarray(pcm, dtype='<f4').tobytes()).hexdigest()


def resample_float(pcm, source_rate, target_rate):
    """Preserve float waveform scale, overshoot, silence and ceil sample count."""
    require(type(source_rate) is int and source_rate > 0 and type(target_rate) is int and target_rate > 0,
            'Invalid sample rate')
    pcm = np.asarray(pcm)
    require(pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) > 0 and np.isfinite(pcm).all(),
            'Expected finite mono float32 PCM')
    common = gcd(source_rate, target_rate)
    result = (resample_poly(pcm, target_rate//common, source_rate//common).astype(np.float32)
              if source_rate != target_rate else pcm.copy())
    require(len(result) == (len(pcm)*target_rate+source_rate-1)//source_rate and np.isfinite(result).all(),
            'Resampling changed the expected ceil clock')
    return result


def derived_input(integer, rate):
    require(rate == 44100 and integer.dtype == np.int16 and integer.ndim == 1 and len(integer) > 0,
            'Derived cohort requires original mono 44100 Hz int16')
    original = integer.astype(np.float32)/32768
    floating = resample_float(original, rate, FLOAT_RATE)
    model = resample_float(floating, FLOAT_RATE, adapter.RATE)
    return floating, model, dict(
        conversion='int16/32768 float32 -> scipy.resample_poly 22050 -> scipy.resample_poly 16000',
        normalization=False, clipping=False, original_pcm_sha256=pcm_hash(original),
        float_pcm_sha256=pcm_hash(floating), model_pcm_sha256=pcm_hash(model),
        physical_samples=len(integer), physical_rate=rate, physical_duration_seconds=len(integer)/rate,
        float_samples=len(floating), float_rate=FLOAT_RATE, model_samples=len(model), model_rate=adapter.RATE,
        float_duration_seconds=len(floating)/FLOAT_RATE, model_duration_seconds=len(model)/adapter.RATE,
        mask_clock='original_44100_PCM_not_resampler_ceil_extent')


def validate_gate(report, names):
    """All 60 physical outputs must pass; preserve a separate all-window failure."""
    require(len(names) == len(set(names)) == 60, 'Prerequisite requires the complete 60-case order')
    plan = report.get('plan', {})
    require(plan.get('model') == MODEL and plan.get('graph_sha256') == GRAPH_SHA
            and plan.get('source_report_sha256') == SOURCE_SHA and plan.get('cases') == names,
            'Prerequisite identity or case order differs')
    require(report.get('completed_cases') == 60 and [r.get('name') for r in report.get('rows', [])] == names,
            'Prerequisite is not a complete ordered 60-case report')
    require(plan.get('activation_rtol') == .001 and plan.get('activation_atol') == .0001
            and plan.get('endpoint_tolerance_seconds') == .001
            and plan.get('device') == 'CPUExecutionProvider'
            and plan.get('intra_threads') == 2 and plan.get('inter_threads') == 1,
            'Prerequisite tolerances or runtime policy changed')
    flags = ('all_physical_activations_preserved', 'all_physical_events_preserved',
             'all_scoring_counts_and_scopes_preserved')
    require(all(report.get(key) is True for key in flags), 'Physical/events/scoring prerequisite did not pass')
    window_flags = []
    for row in report['rows']:
        require(set(row.get('physical_comparisons', {})) == set(adapter.OUTPUTS)
                and all(v.get('preserved') is True for v in row['physical_comparisons'].values())
                and row.get('events_preserved') is True and row.get('interval_scope_unchanged') is True
                and row.get('scoring_count_changes') == [], 'A prerequisite row failed or lacks evidence')
        require(set(row.get('window_comparisons', {})) == set(adapter.OUTPUTS)
                and all(type(v.get('preserved')) is bool for v in row['window_comparisons'].values()),
                'Missing complete-window prerequisite evidence')
        window_flags.extend(v['preserved'] for v in row['window_comparisons'].values())
    all_windows = all(window_flags)
    require(report.get('all_window_activations_preserved') is all_windows
            and report.get('original_all_window_protocol_passed') is all_windows,
            'Original all-window protocol status was hidden or changed')
    return dict(physical_events_scores_passed=True, all_window_activations_preserved=all_windows,
                original_all_window_protocol_passed=all_windows,
                interpretation='new_float_input_development_probe_not_retroactive_portability_acceptance')


def native_input(spec):
    path = ROOT/spec['path']
    verify(path, spec['sha256'])
    verify(path.parent/'separation.json', spec['separation_sha256'])
    separation = read(path.parent/'separation.json')
    require(separation['sample_rate'] == FLOAT_RATE, 'Native stem sample rate differs')
    floating = np.load(path, allow_pickle=False)
    require(floating.dtype == np.float32 and floating.shape == (spec['samples'],)
            and np.isfinite(floating).all(), 'Native float stem shape/values differ')
    if spec['name'] == 'sakura-nova-guitar':
        require(separation['source'] == 'provided' and separation['end_padding_samples']['guitar'] == 0
                and separation['decoded_files']['guitar']['npy_sha256'] == spec['sha256'],
                'Provided Nova stem provenance differs')
    else:
        require(separation['profile'] == 'six' and separation['frames'] == len(floating)
                and separation['model_sha256'] == '34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd'
                and separation['audio_sha256'] == '08b6aff5db874fca7718c6bab0b7c97e01b59648531bfd0c53d5b1628346010d',
                'HTDemucs6 stem provenance differs')
    return floating, dict(spec, path=str(path.resolve()), sample_rate=FLOAT_RATE,
        pcm_sha256=pcm_hash(floating), peak_absolute=float(np.max(np.abs(floating))),
        reference_available=False, accuracy_claim=False)


def prepare(export_root, verification_root):
    export_root, verification_root = Path(export_root), Path(verification_root)
    for path, digest in FROZEN_CODE.items():
        verify(ROOT/path, digest)
    export = read(export_root/'report.json')
    require(export['model'] == MODEL and export['graph_sha256'] == GRAPH_SHA
            and export['source_report_sha256'] == SOURCE_SHA
            and export['identity']['thresholds'] == adapter.THRESHOLDS, 'Wrong frozen FL export')
    verify(export_root/'guitar.onnx', GRAPH_SHA)
    source_path = Path(export['source_report'])
    verify(source_path, SOURCE_SHA)
    source = read(source_path)
    cases = source['plan']['cases']
    names = [c['name'] for c in cases]
    require(source['plan']['model_order'] == ['guitar_gaps', MODEL]
            and [(r['model'], r['name']) for r in source['rows']]
            == [(m, n) for m in ['guitar_gaps', MODEL] for n in names], 'Original cohort changed')
    gate = read(verification_root/'report.json')
    status = validate_gate(gate, names)
    require(gate['plan'] == read(verification_root/'plan.json'), 'Prerequisite plan differs')
    require(gate['plan']['export_report_sha256'] == sha(export_root/'report.json'), 'Prerequisite used another export')
    require(gate['plan']['ort'] == importlib.metadata.version('onnxruntime')
            and gate['plan']['numpy'] == np.__version__, 'Application runtime differs from the prerequisite')
    for row in gate['rows']:
        folder = verification_root/row['name']
        require(read(folder/'report.json') == row, 'Prerequisite per-case report differs')
        for filename, key in [('windows.npz','windows_sha256'),('activations.npz','activations_sha256'),
                              ('notes.json','notes_sha256')]:
            verify(folder/filename, row[key])
    require(Counter(c['metadata']['mode'] for c in cases) == {'comp':30,'solo':30}
            and Counter(re.match('[A-Za-z]+',c['metadata']['style']).group() for c in cases)
            == dict.fromkeys(('BN','Funk','Jazz','Rock','SS'),12), 'Original strata changed')
    prepared = []
    originals = {r['name']:r for r in source['rows'] if r['model'] == MODEL}
    for case in cases:
        verify(case['audio'], case['audio_sha256'])
        rate, integer = wavfile.read(case['audio'])
        require(rate == case['sample_rate'] == 44100 and len(integer) == case['samples'], 'Original clock differs')
        _, _, identity = derived_input(integer, rate)
        reference = case['references']['strings']
        verify(reference['path'], reference['sha256'])
        row = originals[case['name']]
        folder = source_path.parent/MODEL/case['name']
        verify(folder/'report.json', row['report_sha256'])
        verify(folder/'notes.json', row['notes_sha256'])
        before, scope = evaluation.window_metrics(read(reference['path']),read(folder/'notes.json'),
            begin=0, end=case['seconds'], context_start=0, context_end=case['duration_seconds'])
        require(before == row['metrics'] and scope == row['interval_scope'], 'Original FL score differs')
        prepared.append(dict(case, float_identity=identity, baseline=row, baseline_folder=str(folder.resolve())))
    native = [native_input(spec)[1] for spec in NATIVE]
    sakura = ROOT/'test-data/sakura-benchmark'
    verify(sakura/'manifest.json','913573b657fd1531e63328f67de696a4c9f194c147d77522108c7c8409dc68d8')
    verify(sakura/'original.osz','fee43757b5ea94869abc9c2626708de022da4506382bb055f0f90ec51e88fe9c')
    guitar_mp3_sha = '6bb1103126f4f7d7bdbef7429e17a05c5a696ed455266eadf173dcc6e65d98b5'
    verify(sakura/'bundled/guitar.mp3',guitar_mp3_sha)
    with zipfile.ZipFile(sakura/'original.osz') as archive:
        require(any(hashlib.sha256(archive.read(n)).hexdigest() == guitar_mp3_sha
                    for n in archive.namelist() if n.lower().endswith('.mp3')), 'Original archive lacks the guitar source')
    require('torch' not in sys.modules and 'onnxruntime' not in sys.modules, 'Preparation loaded a neural runtime')
    return dict(revision=1, model=MODEL, graph_path=str((export_root/'guitar.onnx').resolve()),graph_sha256=GRAPH_SHA,
        export_report=str((export_root/'report.json').resolve()),export_report_sha256=sha(export_root/'report.json'),
        source_report=str(source_path.resolve()),source_report_sha256=SOURCE_SHA,
        verification_root=str(verification_root.resolve()),verification_report_sha256=sha(verification_root/'report.json'),
        prerequisite=status, code_sha256=FROZEN_CODE|{'scripts/check_hf_guitar_float_inputs.py':sha(__file__),
                                                    'manialab/guitar.py':sha(ROOT/'manialab/guitar.py')},
        thresholds=adapter.THRESHOLDS, window=adapter.WINDOW, hop=adapter.HOP, fps=adapter.FPS,
        device='CPUExecutionProvider',intra_threads=2,inter_threads=1,
        runtime={k:importlib.metadata.version(k) for k in ('numpy','scipy','onnxruntime')},
        executable=str(Path(sys.executable).resolve()),cases=prepared,native_inputs=native,
        normalization=False,clipping=False,timing_quantization=False,training_performed=False,
        comparison='60_derived_inputs_vs_original_FL',native_evaluation='two_unlabelled_full_recording_diagnostics')


def infer_windows(session, pcm, physical_samples, physical_rate):
    padded = ((len(pcm)+adapter.WINDOW-1)//adapter.WINDOW)*adapter.WINDOW
    starts = np.arange(0,padded-adapter.WINDOW+1,adapter.HOP,dtype=np.int64)
    pieces = {key:[] for key in adapter.OUTPUTS}
    for start in starts:
        chunk = np.zeros(adapter.WINDOW,dtype=np.float32)
        available = pcm[int(start):int(start)+adapter.WINDOW]
        chunk[:len(available)] = available
        outputs = session.run(list(adapter.OUTPUTS), {'audio':chunk[None]})
        require(len(outputs) == len(adapter.OUTPUTS), 'Missing model heads')
        for key,value in zip(adapter.OUTPUTS,outputs):
            require(value.shape == (1,1001,88) and value.dtype == np.float32 and np.isfinite(value).all(),
                    'Invalid float-input model output')
            pieces[key].append(value[0])
    windows = {key:np.stack(value) for key,value in pieces.items()}
    heads = adapter.stitch_windows(windows)
    physical,times,valid = adapter.physical_heads(heads,physical_samples,physical_rate)
    return starts,windows,heads,physical,times,valid


def metric_changes(before, after):
    changes = []
    for tolerance in evaluation.TOLERANCES:
        for metric in evaluation.METRICS:
            for key in ('available','reason','reference','predicted','matched','precision','recall','f1'):
                a,b = before[tolerance][metric].get(key),after[tolerance][metric].get(key)
                if a != b:
                    changes.append(dict(tolerance=tolerance,metric=metric,field=key,before=a,after=b))
    return changes


def replay_runtime(floating,pcm,starts,windows,original_physical):
    """Exercise production infer/candidates against recorded windows, with no NN."""
    class RecordedSession:
        def __init__(self):self.calls=0
        def run(self,names,inputs):
            require(tuple(names) == adapter.OUTPUTS and set(inputs) == {'audio'}, 'Runtime changed model interface')
            require(self.calls < len(starts), 'Runtime requested extra inference windows')
            start=int(starts[self.calls]);expected=np.zeros((1,adapter.WINDOW),dtype=np.float32)
            part=pcm[start:start+adapter.WINDOW];expected[0,:len(part)]=part
            actual=inputs['audio']
            require(actual.dtype == np.float32 and np.array_equal(actual,expected),
                    'Runtime changed a float input sample, gain, padding or window clock')
            result=[windows[key][self.calls:self.calls+1].copy() for key in names]
            self.calls+=1
            return result
    replay=RecordedSession()
    heads=guitar.infer(floating,FLOAT_RATE,replay)
    require(replay.calls == len(starts),'Runtime omitted recorded inference windows')
    stitched=adapter.stitch_windows(windows)
    expected,_,mask=adapter.physical_heads(stitched,len(floating),FLOAT_RATE)
    for key in adapter.OUTPUTS:
        require(np.array_equal(heads[key],expected[key]),'Runtime physical activations differ from its own float clock')
    original_frames=len(original_physical[adapter.OUTPUTS[0]])
    application_frames=len(heads[adapter.OUTPUTS[0]])
    common=min(original_frames,application_frames)
    require(all(np.array_equal(heads[key][:common],original_physical[key][:common]) for key in adapter.OUTPUTS),
            'Runtime changed shared physical frames')
    events=[dict(n,program=-1,is_drum=False,voice=MODEL) for n in guitar.decode(heads,**guitar.THRESHOLDS)]
    modes={}
    for enabled in (False,True):
        notes,trace=guitar.candidates(events,allow_predicted_releases=enabled)
        require(len(notes)==len(trace)==len(events),'Candidate conversion changed event count')
        for note,proof,raw in zip(notes,trace,events):
            require(proof['raw_event']==raw and proof['authored_ln_verified'] is False
                    and proof['model_predicted_release_hypothesis'] is bool(enabled and raw['release_observed'])
                    and note.start==round(raw['onset']*1000) and note.end==round(raw['offset']*1000)
                    and note.pitch==raw['pitch'] and note.voice=='guitar' and note.strength==1.,
                    'Candidate clocks or provenance changed')
            if not enabled or not raw['release_observed']:
                require(note.release_reason in guitar.UNOBSERVED_RELEASES,'Unknown or disabled tail became LN eligible')
        modes['predicted_releases' if enabled else 'default_onsets'] = dict(candidates=[n.data() for n in notes],trace=trace)
    return heads,events,modes,dict(recorded_windows_replayed=replay.calls,new_neural_inference=False,
        input_samples_exact=True,shared_physical_frames_exact=True,original_clock_frames=original_frames,
        application_float_clock_frames=application_frames,application_extra_physical_frames=application_frames-original_frames,
        application_physical_mask=mask.tolist(),application_uses_ceil_float_length=True,
        statement='Extra application frames are retained and decoded, never cropped to manufacture parity')


def save_prediction(folder,session,floating,pcm,identity):
    folder.mkdir(parents=True)
    np.save(folder/'input-22050.npy',floating,allow_pickle=False)
    np.save(folder/'model-input-16000.npy',pcm,allow_pickle=False)
    starts,windows,heads,physical,times,valid = infer_windows(session,pcm,identity['physical_samples'],identity['physical_rate'])
    events = [dict(n,program=-1,is_drum=False,voice=MODEL) for n in decode(physical,**adapter.THRESHOLDS)]
    require(all(n['release_observed'] is (n['release_reason'] in ('frame_decay','release_regression')) for n in events),
            'Unknown tail was promoted')
    np.savez_compressed(folder/'windows.npz',starts=starts,**windows)
    np.savez_compressed(folder/'activations.npz',times=times,valid_frame=valid,**heads)
    adapter.write_new(folder/'notes.json',events)
    runtime_heads,runtime_events,modes,replay=replay_runtime(floating,pcm,starts,windows,physical)
    runtime_mask=np.asarray(replay.pop('application_physical_mask'),dtype=bool)
    np.savez_compressed(folder/'runtime-activations.npz',times=times[runtime_mask],**runtime_heads)
    np.savez_compressed(folder/'runtime-physical-mask.npz',times=times,valid_frame=runtime_mask)
    adapter.write_new(folder/'runtime-notes.json',runtime_events)
    adapter.write_new(folder/'runtime-candidates.json',modes)
    return events,dict(input_identity=identity,events=len(events),windows=len(starts),physical_frames=int(valid.sum()),
        runtime_events=len(runtime_events),runtime_replay=replay,
        release_reasons=dict(Counter(n['release_reason'] for n in events)),
        runtime_release_reasons=dict(Counter(n['release_reason'] for n in runtime_events)),
        artifacts={name:sha(folder/name) for name in ('input-22050.npy','model-input-16000.npy','windows.npz','activations.npz',
            'notes.json','runtime-activations.npz','runtime-physical-mask.npz','runtime-notes.json','runtime-candidates.json')})


def run(plan,output):
    require(Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve(),
            'Run inference with the application .venv Python')
    require(not output.exists(), 'Preserve prior float-input probe evidence')
    # Recheck the report immediately before importing ORT or constructing a session.
    for path,digest in plan['code_sha256'].items():verify(ROOT/path,digest)
    gate_root = Path(plan['verification_root'])
    verify(gate_root/'report.json',plan['verification_report_sha256'])
    validate_gate(read(gate_root/'report.json'),[c['name'] for c in plan['cases']])
    verify(plan['graph_path'],GRAPH_SHA)
    import onnxruntime as ort
    options = ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session = ort.InferenceSession(plan['graph_path'],sess_options=options,providers=['CPUExecutionProvider'])
    output.mkdir(parents=True)
    adapter.write_new(output/'plan.json',plan)
    rows = []
    for index,case in enumerate(plan['cases']):
        started = time.monotonic()
        verify(case['audio'],case['audio_sha256'])
        rate,integer = wavfile.read(case['audio'])
        floating,pcm,identity = derived_input(integer,rate)
        require(identity == case['float_identity'], 'Input changed after preparation')
        folder = output/'derived-cohort'/case['name']
        notes,row = save_prediction(folder,session,floating,pcm,identity)
        labels = case['references']['strings'];verify(labels['path'],labels['sha256'])
        metrics,scope = evaluation.window_metrics(read(labels['path']),notes,begin=0,end=case['seconds'],
            context_start=0,context_end=case['duration_seconds'])
        before = case['baseline']
        require(scope['reference_indices'] == before['interval_scope']['reference_indices']
                and scope['unobservable_reference_indices'] == before['interval_scope']['unobservable_reference_indices'],
                'Reference scoring scope changed')
        runtime_metrics,runtime_scope=evaluation.window_metrics(read(labels['path']),read(folder/'runtime-notes.json'),
            begin=0,end=case['seconds'],context_start=0,context_end=case['duration_seconds'])
        row.update(name=case['name'],metrics=metrics,interval_scope=scope,original_metrics=before['metrics'],
            runtime_metrics=runtime_metrics,runtime_interval_scope=runtime_scope,
            runtime_metric_changes=metric_changes(before['metrics'],runtime_metrics),
            original_interval_scope=before['interval_scope'],metric_changes=metric_changes(before['metrics'],metrics),
            original_events=before['interval_scope']['predicted_total'],seconds=time.monotonic()-started)
        adapter.write_new(folder/'report.json',row);rows.append(row)
        print(json.dumps(dict(stage='derived',index=index+1,total=60,name=case['name'],events=len(notes),
            metric_changes=row['metric_changes'])),flush=True)
    native_rows = []
    for spec,prepared in zip(NATIVE,plan['native_inputs']):
        started = time.monotonic();floating,details=native_input(spec)
        require(details == prepared,'Native input changed after preparation')
        pcm = resample_float(floating,FLOAT_RATE,adapter.RATE)
        identity = dict(physical_samples=len(floating),physical_rate=FLOAT_RATE,
            physical_duration_seconds=len(floating)/FLOAT_RATE,float_pcm_sha256=pcm_hash(floating),
            model_pcm_sha256=pcm_hash(pcm),float_samples=len(floating),float_rate=FLOAT_RATE,
            model_samples=len(pcm),model_rate=adapter.RATE,normalization=False,clipping=False,
            mask_clock='manifest_native_22050_float_PCM',conversion='native float32 -> scipy.resample_poly 16000')
        folder=output/'native-diagnostics'/spec['name']
        _,row=save_prediction(folder,session,floating,pcm,identity)
        row.update(name=spec['name'],source=details,reference_available=False,accuracy_metrics=None,
            musical_acceptance=False,seconds=time.monotonic()-started)
        adapter.write_new(folder/'report.json',row);native_rows.append(row)
        print(json.dumps(dict(stage='native_unlabelled',name=spec['name'],events=row['events'])),flush=True)
    require([r['name'] for r in rows] == [c['name'] for c in plan['cases']] and len(rows)==60
            and [r['name'] for r in native_rows] == [s['name'] for s in NATIVE], 'Incomplete float-input probe')
    strata={}
    for dimension in ('mode','style'):
        labels={c['name']:(c['metadata'][dimension] if dimension=='mode' else re.match('[A-Za-z]+',c['metadata']['style']).group())
                for c in plan['cases']}
        strata[dimension]={label:dict(before=evaluation.aggregate([c['baseline'] for c in plan['cases'] if labels[c['name']]==label]),
            after=evaluation.aggregate([r for r in rows if labels[r['name']]==label]),
            application_after=evaluation.aggregate([dict(name=r['name'],metrics=r['runtime_metrics'])
                for r in rows if labels[r['name']]==label])) for label in sorted(set(labels.values()))}
    for path,digest in plan['code_sha256'].items():verify(ROOT/path,digest)
    verify(plan['source_report'],SOURCE_SHA);verify(plan['graph_path'],GRAPH_SHA)
    require('torch' not in sys.modules,'Float probe loaded Torch')
    result=dict(plan=plan,derived_rows=rows,native_diagnostics=native_rows,
        derived_groups=dict(before=evaluation.aggregate([c['baseline'] for c in plan['cases']]),after=evaluation.aggregate(rows)),
        application_float_clock_group=evaluation.aggregate([dict(name=r['name'],metrics=r['runtime_metrics']) for r in rows]),
        derived_strata=strata,derived_cases=60,native_unlabelled_cases=2,
        original_all_window_protocol_passed=plan['prerequisite']['original_all_window_protocol_passed'],
        enabled_in_application=False,musical_acceptance=False,training_performed=False,
        seconds=sum(r['seconds'] for r in rows+native_rows),
        limitations=['Derived sampling-chain comparison, not new independent performances.',
            'Public GuitarSet upstream training overlap unknown.',
            'Native stems have no independent guitar ground truth and no accuracy score.',
            'Unknown tails retained; no LN or note-grid acceptance.'])
    adapter.write_new(output/'report.json',result)
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--export',dest='export_root',type=Path,default=ROOT/'test-data/hf-guitar-fl-onnx-v1')
    cli.add_argument('--verification',type=Path,required=True,help='Complete 60-case original-input ORT report directory')
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--prepare-only',action='store_true')
    args=cli.parse_args()
    require(not args.output.exists(),'Preserve existing output; use a new destination')
    plan=prepare(args.export_root,args.verification)
    if args.prepare_only:
        args.output.mkdir(parents=True)
        adapter.write_new(args.output/'plan.json',plan)
        adapter.write_new(args.output/'preparation.json',dict(prepared=True,inference_performed=False,
            derived_cases=60,native_unlabelled_cases=2,prerequisite=plan['prerequisite']))
    else:
        run(plan,args.output)


if __name__=='__main__':
    main()
