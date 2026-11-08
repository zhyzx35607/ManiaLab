"""Context-preserving MMRS windows without modifying vendored source."""
from bisect import bisect_left


def has_musical_target(labels, event_ids):
    """Release-only and sustained windows are useful LN supervision too."""
    return any(token in event_ids for token in labels)


def target_windows(frame_count, frame_capacity, target_frames):
    """Each frame is a target once; earlier frames are unlabelled context."""
    if not 0 < target_frames < frame_capacity:
        raise ValueError('A target window must leave room for previous audio')
    history = frame_capacity - target_frames
    for target_start in range(0, frame_count, target_frames):
        yield max(0, target_start-history), target_start, min(frame_count, target_start+target_frames)


def context_dataset_class():
    import torch
    from osuT5.dataset.mmrs_dataset import MmrsDataset, BeatmapDatasetIterable

    class ContextIterable(BeatmapDatasetIterable):
        def _create_sequences(self, frames, frame_times, out_context, in_context, extra_data=None):
            # The original V32 encoder covers 16.376 s. Keep 8.192 s of actual
            # previous audio/events, exceeding the 8 s LN sustain interval.
            target_frames = (self.frame_seq_len-1)//2
            output = []
            for begin, target, end in target_windows(len(frames), self.frame_seq_len, target_frames):
                sequences = super()._create_sequences(frames[begin:end], frame_times[begin:end], out_context, in_context, extra_data)
                if len(sequences) != 1:
                    raise ValueError('Expected a single context window')
                sequence = sequences[0]
                end_time = frame_times[end-1] + self.args.hop_length/self.args.sample_rate*1000
                # The upstream helper uses a following frame to delimit the
                # final events. Our sliced audio needs an explicit exclusive end.
                for source, sliced in zip(out_context+in_context, sequence['out_context']+sequence['in_context']):
                    first = bisect_left(source['event_times'], frame_times[begin])
                    last = bisect_left(source['event_times'], end_time)
                    sliced['events'] = source['events'][first:last]
                for source, sliced in zip(out_context, sequence['out_context']):
                    sliced['labels_offset'] = (bisect_left(source['event_times'], frame_times[target]) -
                                               bisect_left(source['event_times'], frame_times[begin]))
                sequence['song_position'] = torch.tensor([begin/len(frames), end/len(frames)], dtype=torch.float32)
                sequence['context_window'] = dict(start_ms=float(frame_times[begin]), target_ms=float(frame_times[target]),
                    end_ms=float(end_time))
                output.append(sequence)
            return output

        def process_sequences(self, sequences, beatmap_path):
            for sequence in sequences:
                self.maybe_change_dataset()
                sequence = self._normalize_time_shifts(sequence, beatmap_path)
                sequence = self._tokenize_sequence(sequence)
                contexts = sequence['in_context'] + sequence['out_context']
                required = 1 + len(sequence['special_tokens']) + sum(
                    len(c['tokens']) + len(c['special_tokens']) + 2*int(c['add_type']) for c in contexts)
                if required > self.args.tgt_seq_len:
                    # A synthetic EOS after a truncated group teaches a false
                    # segment ending. Do not silently train on that sequence.
                    self.audit['over_capacity'] += 1
                    continue
                self.audit['complete_windows'] += 1
                sequence = self._pad_frame_sequence(sequence)
                sequence = self._pad_and_split_token_sequence(sequence)
                yield sequence

    class ContextDataset(MmrsDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit = dict(over_capacity=0, complete_windows=0)

        def _iterable_factory(self, metadata):
            instance = ContextIterable(metadata, self.args, self.path, self.parser, self.tokenizer,
                                       self.test, self.shared, self.sample_weights)
            instance.audit = self.audit
            return instance

    return ContextDataset


def typed_time_positions(labels, event_start, event_end):
    """Locate time labels for each type-last V32 event, not its type token."""
    time_index = None
    positions = {name: [] for name in ('ln_head_time', 'ln_tail_time', 'tap_time')}
    types = {event_start[name]: category for name, category in
             [('hold_note', 'ln_head_time'), ('hold_note_end', 'ln_tail_time'), ('circle', 'tap_time')]}
    for i, token in enumerate(labels):
        if event_start['t'] <= token < event_end['t']:
            time_index = i
        elif token in types:
            if time_index is not None:
                positions[types[token]].append(time_index)
            time_index = None
        elif token < 0:
            time_index = None
    return positions
