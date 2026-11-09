"""Research-only diagnostics for inactive frames immediately after piano attacks."""
import numpy as np
from manialab.piano import decode, regression_peaks, OUTPUTS, FPS


def diagnose(activations):
    """Preserve detected attacks; propose tails only after observed activation/decay."""
    original = decode(activations)
    arrays = [np.asarray(activations[key]) for key in OUTPUTS]
    onset, _ = regression_peaks(arrays[0], .3, 2)
    release, release_shift = regression_peaks(arrays[1], .3, 4)
    rows, proposed = [], [dict(event) for event in original]
    event_index = 0
    last = len(arrays[0])-1
    for key in range(88):
        attacks = np.flatnonzero(onset[:, key])
        for index, start in enumerate(attacks):
            start = int(start)
            next_start = int(attacks[index+1]) if index+1 < len(attacks) else last+1
            boundary = min(start+600, last, next_start-1)
            if boundary <= start:
                continue
            event = original[event_index]
            assert event['pitch'] == key+21
            row = dict(index=event_index, pitch=event['pitch'], onset=event['onset'],
                       old_offset=event['offset'], old_reason=event['release_reason'],
                       start_frame=start, search_boundary=boundary,
                       status='existing_active_tail', proposed_offset=None)
            if event['release_reason'] == 'frame_inactive':
                stop = min(start+5, boundary)
                active = np.flatnonzero(arrays[2][start:stop+1, key] > .1)
                if not len(active):
                    row['status'] = ('reattack_before_activation' if index+1 < len(attacks) and next_start <= start+5 else
                        'input_boundary_before_activation' if last < start+5 else 'no_activation_in_window')
                else:
                    first = start+int(active[0])
                    end = first
                    while end <= boundary and arrays[2][end, key] > .1:
                        end += 1
                    row.update(first_active_frame=first, delay_ms=(first-start)*1000/FPS,
                               active_frames=end-first)
                    if end-first < 2:
                        row['status'] = 'isolated_activation'
                    elif end > boundary:
                        row['status'] = 'active_without_observed_decay'
                    else:
                        row['first_decay_frame'] = end
                        peaks = np.flatnonzero(release[first:end+1, key])+first
                        tail, reason, shift = end, 'delayed_frame_decay', 0.
                        if len(peaks) and int(peaks[0])-start > end-int(peaks[0]):
                            tail = int(peaks[0])
                            reason = 'delayed_release_regression'
                            shift = float(release_shift[tail, key])
                        value = float(np.float32((tail+shift)/FPS))
                        row.update(status=reason, first_release_frame=int(peaks[0]) if len(peaks) else None,
                                   proposed_offset=value)
                        proposed[event_index].update(offset=value, release_reason=reason, release_observed=True)
            rows.append(row)
            event_index += 1
    assert event_index == len(original)
    for old, new in zip(original, proposed):
        assert {k:v for k,v in old.items() if k not in ('offset','release_reason','release_observed')} == {
            k:v for k,v in new.items() if k not in ('offset','release_reason','release_observed')}
    return original, proposed, rows
