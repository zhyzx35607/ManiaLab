"""Load audited Nova/YourMT3 observations without reading either author chart."""
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from scripts.probe_mapper_adapter import sha


def load_evidence(root):
    evidence = {}; identity = {}
    for role in ('piano', 'other'):
        folder = root/(role+'-yourmt3-full')
        notes_path = folder/'notes.json'
        audit = json.loads((folder/'release-audit.json').read_text(encoding='utf-8'))
        if sha(notes_path) != audit['notes_sha256']: raise ValueError('Audited transcription changed')
        notes = json.loads(notes_path.read_text(encoding='utf-8'))
        provenance = json.loads((folder/'release-provenance.json').read_text(encoding='utf-8'))
        forced = [False]*len(notes)
        for row in provenance:
            forced[row['note_index']] = 'forced_segment_boundary' in row['endpoint_evidence']
        voices = ('piano',) if role == 'piano' else ('other', 'bass', 'guitar')
        signal = None; hashes = {}
        for voice in voices:
            path = root/f'bundled/{voice}/preview.wav'; rate, y = wavfile.read(path)
            if rate != 22050 or y.ndim != 1 or y.dtype != np.int16: raise ValueError('Unexpected source clock')
            hashes[str(path)] = sha(path)
            y = y.astype(np.float32)/32768
            signal = y if signal is None else signal+y
        transcription = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if hashes != transcription['identity']['sources']: raise ValueError('Audio and note identities disagree')
        # RMS per physical 10ms-like bin, on a fixed PCM full-scale reference.
        # No whole-song percentile leaks later audio levels into earlier rows.
        hop = 221
        padded = np.pad(signal, (0, (-len(signal)) % hop)).reshape(-1, hop)
        energy = np.sqrt(np.mean(padded*padded, axis=1))
        energy = np.clip(4*energy, 0, 4)
        evidence[role] = dict(notes=notes, forced=forced, energy=energy, hop_ms=hop/rate*1000)
        identity[role] = dict(notes_sha256=sha(notes_path), provenance_sha256=sha(folder/'release-provenance.json'), sources=hashes)
    return evidence, identity
