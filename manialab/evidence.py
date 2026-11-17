"""Small immutable-by-copy source snapshots for generated chart inspection.

Detection and expression use original audio time. A chart's evidence_end is
the pre-quantization expression in chart time, not necessarily a detected tail.
Never look up an old candidate in a later analysis just because IDs match.
"""
from copy import deepcopy


DETECTED_FIELDS = ('start', 'end', 'voice', 'pitch', 'source', 'strength',
                   'release_reason', 'instrument_id')
EXPRESSED_FIELDS = ('start', 'end', 'phrase', 'release_reason', 'support', 'support_coverage')


def capture_source_evidence(analysis, evidence, output, *, locked=(), previous=None):
    detected = {n['id']: n for n in analysis.get('candidates', [])}
    expressed = {n.id: n.data() for n in evidence}
    locked_ids = {n.id for n in locked}
    result = {}
    for note in output:
        identity = note.id
        if identity in locked_ids:
            # Missing legacy provenance stays unknown, even if a fresh analysis
            # reuses that ID for a different candidate.
            if identity in (previous or {}):
                result[identity] = deepcopy(previous[identity])
        elif identity in detected and identity in expressed:
            result[identity] = {
                'detected': {k: deepcopy(detected[identity][k]) for k in DETECTED_FIELDS
                             if k in detected[identity]},
                'expressed': {k: deepcopy(expressed[identity][k]) for k in EXPRESSED_FIELDS},
            }
    return result
