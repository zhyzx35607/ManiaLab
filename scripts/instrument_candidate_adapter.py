"""Research bridge from audited instrument predictions to source candidates.

This does not run or ship YourMT3, infer confidence from velocity, or define
authored LN releases. Every accepted/rejected event retains an audit record.
"""
import math

from manialab.domain import Note


def accompaniment_candidates(events, provenance, duration_ms, voice='other'):
    flags={row['note_index']:row['endpoint_evidence'] for row in provenance}
    if len(flags)!=len(provenance) or set(flags)!=set(range(len(events))):
        raise ValueError('Endpoint audit must cover every event exactly once')
    notes,trace=[],[]
    for index,event in enumerate(events):
        if (not all(math.isfinite(event[k]) for k in ('onset','offset','pitch','program')) or
            event['offset']<=event['onset'] or not 0<=event['pitch']<=127 or
            isinstance(event['program'],bool) or int(event['program'])!=event['program'] or
            not 0<=event['program']<=(128 if event['is_drum'] else 127)):
            raise ValueError('Invalid model event must be audited before integration')
        evidence=flags[index]
        if not evidence or set(evidence)-{'explicit_model_note_off','same_pitch_reattack',
            'forced_segment_boundary','unresolved_or_postprocessed','drum_fixed_10ms'}:
            raise ValueError('Unknown release provenance')
        identifier=f'{voice}:instrument:{index}'
        row=dict(note_index=index,id=identifier,program=event['program'],pitch=event['pitch'],
                 endpoint_evidence=evidence,onset=event['onset'],offset=event['offset'])
        # Supplied source role is accompaniment. Report incompatible predictions,
        # rather than selecting only the best-matching instrument from a chart.
        if event['is_drum']:
            row['decision']='incompatible_drum_prediction'
        elif event['program'] in (100,101):
            row['decision']='incompatible_vocal_prediction'
        elif not 0<=event['onset']*1000<duration_ms:
            row['decision']='outside_audio'
        else:
            reason=('input_boundary' if event['offset']*1000>=duration_ms else
                'forced_segment_boundary' if 'forced_segment_boundary' in evidence else
                'unresolved_or_postprocessed' if 'unresolved_or_postprocessed' in evidence else
                'explicit_model_note_off' if 'explicit_model_note_off' in evidence else
                'same_pitch_reattack' if 'same_pitch_reattack' in evidence else None)
            if reason is None:raise ValueError('Non-drum has no interpretable release audit')
            note=Note(round(event['onset']*1000),round(event['offset']*1000),pitch=int(event['pitch']),
                strength=1.,source='pitch',voice=voice,id=identifier,release_reason=reason,
                instrument_id=f'yourmt3:program:{int(event["program"])}')
            notes.append(note)
            row.update(decision='candidate',release_reason=reason,instrument_id=note.instrument_id,
                       weight_semantics='unit_weight_not_confidence')
        trace.append(row)
    return notes,trace
