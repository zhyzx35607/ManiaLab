"""Explain frozen YourMT3 endpoints before treating them as LN evidence.

Re-decodes saved tokens with the official code and requires identical notes.
Does not repair, select or extend any notes, and never reads an authored chart.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path('test-data/yourmt3/source'))
    cli.add_argument('--runs',type=Path,nargs='+',required=True)
    args=cli.parse_args()
    sys.path.insert(0,str((args.source/'amt/src').resolve()))
    import numpy as np
    from utils.task_manager import TaskManager
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    for folder in args.runs:
        report=json.loads((folder/'report.json').read_text())
        identity=report['identity']
        if identity['source_commit']!=revision:raise ValueError('Decoder source changed')
        manager=TaskManager(task_name=identity['task'])
        with np.load(folder/'tokens.npz',allow_pickle=False) as data:
            tokens=[data[str(i)] for i in range(len(data.files))]
        starts=[report['context_start_seconds']+i*report['segment_samples']/report['sample_rate']
            for i in range(len(tokens))]
        all_notes=[];forced=set();explicit=set();attacks=set()
        endpoint=lambda program,pitch,time:(program,pitch,round(time,7))
        for channel in range(manager.num_decoding_channels):
            zipped,_,_=manager.detokenize_list_batches([p[:,channel,:] for p in tokens],starts,return_events=True)
            previous=[]
            for events,ties,active,start in zipped:
                tie_keys={(n.program,n.pitch) for n in ties}
                for program,pitch in previous:
                    if (program,pitch) not in tie_keys:forced.add(endpoint(program,pitch,start))
                for event in events:
                    if event.is_drum:continue
                    (explicit if event.velocity==0 else attacks).add(endpoint(event.program,event.pitch,event.time))
                previous=active
            notes,_=merge_zipped_note_events_and_ties_to_notes(zipped)
            all_notes.append(notes)
        decoded=[asdict(n) for n in mix_notes(all_notes)]
        frozen=json.loads((folder/'notes.json').read_text())
        if decoded!=frozen:
            raise ValueError('Archived tokens do not reproduce notes; reject endpoint audit')
        provenance=[];counts=Counter()
        for index,n in enumerate(decoded):
            key=endpoint(n['program'],n['pitch'],n['offset'])
            flags=[]
            if n['is_drum']:flags=['drum_fixed_10ms']
            else:
                if key in forced:flags.append('forced_segment_boundary')
                if key in explicit:flags.append('explicit_model_note_off')
                if key in attacks:flags.append('same_pitch_reattack')
                if not flags:flags=['unresolved_or_postprocessed']
            counts.update(flags)
            provenance.append(dict(note_index=index,endpoint_evidence=flags))
        (folder/'release-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        shutil.copyfile(__file__,folder/'audit-runner-snapshot.py')
        result=dict(source_commit=revision,tokens_sha256=sha(folder/'tokens.npz'),
            provenance_sha256=sha(folder/'release-provenance.json'),
            notes_sha256=sha(folder/'notes.json'),runner_sha256=sha(__file__),
            decoded_notes_equal=True,notes=len(decoded),endpoint_counts=dict(counts),
            limits=['Endpoint evidence is from model tokens, not acoustic or authored-LN truth.',
                'Flags can overlap; ambiguous endpoints remain ambiguous.',
                'Forced segment ends and unresolved/postprocessed ends are not supported sustain releases.'])
        (folder/'release-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(dict(run=str(folder),**result)),flush=True)


if __name__=='__main__':main()
