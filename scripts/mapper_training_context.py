"""Context-preserving MMRS windows without modifying vendored source."""
from bisect import bisect_left


def window_in_ranges(window, ranges):
    """Keep all audio/history in one declared half-open range, not only labels."""
    if ranges is None:
        return True
    if not ranges or any(len(r) != 2 or not 0 <= r[0] < r[1] for r in ranges):
        raise ValueError('Invalid allowed audio ranges')
    return any(a <= window['start_ms'] <= window['target_ms'] < window['end_ms'] <= b
               for a, b in ranges)


def key_schedule(available, requested=None):
    """Retain the mixed-key schedule or explicitly select supported buckets."""
    result = list(requested or [10, 6, 10, 7, 10, 8, 10, 9])
    if not result or any(k not in range(6, 11) or k not in available for k in result):
        raise ValueError('Key schedule requests absent or unsupported training keys')
    if set(result) != set(available):
        raise ValueError('Key schedule must cover every prepared training key')
    return result


def has_musical_target(labels, event_ids):
    """Release-only and sustained windows are useful LN supervision too."""
    return any(token in event_ids for token in labels)


def cold_prefix_positions(labels, time_range, group_end_tokens):
    """Locate the first complete standalone event, excluding context markers."""
    first=next((i for i,t in enumerate(labels) if time_range[0]<=t<time_range[1]),None)
    if first is None:return []
    for end in range(first,len(labels)):
        if labels[end]<0:raise ValueError('Ignored label inside the first cold-start event')
        if labels[end] in group_end_tokens:return list(range(first,end+1))
    raise ValueError('Incomplete first cold-start event')


def target_windows(frame_count, frame_capacity, target_frames):
    """Each frame is a target once; earlier frames are unlabelled context."""
    if not 0 < target_frames < frame_capacity:
        raise ValueError('A target window must leave room for previous audio')
    history = frame_capacity - target_frames
    for target_start in range(0, frame_count, target_frames):
        yield max(0, target_start-history), target_start, min(frame_count, target_start+target_frames)


def cold_start_group_indices(groups):
    """Keep complete heads/taps; never invent a head for an incoming LN.

    Input tuples are (kind, column, original_indices) in time order. This is
    only for additional standalone examples; normal history windows stay intact.
    """
    active=set();kept=[]
    for kind,column,indices in groups:
        if kind=='hold_note':
            active.add(column)
        elif kind in ('hold_note_end','hold_note_sustain'):
            if column not in active:
                continue
            if kind=='hold_note_end':active.remove(column)
        kept.extend(indices)
    return kept


def cold_start_event_indices(events, type_names, types_first=False):
    """Split complete parser events, including its LN sustain sentinel.

    Upstream get_groups() does not treat hold_note_sustain as a delimiter;
    using it here attaches an orphan sustain to the following tap/hold group.
    Tuples are (event type string, value), independent of upstream namespaces.
    """
    delimiters=set(type_names)|{'hold_note_sustain'}
    groups=[];indices=[];kind=None

    def append_group():
        column=next((events[i][1] for i in indices if events[i][0]=='column'),None)
        groups.append((kind,column,list(indices)))

    for index,(name,_) in enumerate(events):
        if types_first and name in delimiters:
            if kind is not None:append_group()
            elif indices:raise ValueError('Incomplete leading event group')
            indices=[];kind=name
        indices.append(index)
        if not types_first and name in delimiters:
            kind=name;append_group();indices=[];kind=None
    if indices:
        if not types_first or kind is None:
            raise ValueError('Incomplete trailing event group')
        append_group()
    return cold_start_group_indices(groups)


def context_dataset_class():
    import copy
    import torch
    from osuT5.dataset.mmrs_dataset import MmrsDataset, BeatmapDatasetIterable
    from osuT5.dataset.data_utils import TYPE_EVENTS

    class ContextIterable(BeatmapDatasetIterable):
        def _get_next_beatmap(self, audio_samples, i, beatmap_metadata, set_metadata,
                             speed, flip=(False, False)):
            # An input-only GD sketch stays available in set_metadata, but
            # never becomes a reverse task with the finished answer as input.
            if self.target_ids is not None and int(i) not in self.target_ids:
                return
            yield from super()._get_next_beatmap(audio_samples, i, beatmap_metadata, set_metadata, speed, flip)

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
                    end_ms=float(end_time),kind='history')
                output.append(sequence)
                if self.cold_start_windows and begin<target:
                    cold=copy.deepcopy(sequence)
                    for context in cold['out_context']:
                        keep=cold_start_event_indices([(e.type.value,e.value) for e in context['events']],
                            [kind.value for kind in TYPE_EVENTS],types_first=self.args.types_first)
                        context['events']=[context['events'][i] for i in keep]
                        context['labels_offset']=0
                    cold['context_window']=dict(start_ms=float(frame_times[begin]),target_ms=float(frame_times[begin]),
                        end_ms=float(end_time),kind='cold_start')
                    output.append(cold)
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
        def __init__(self, *args, target_ids=None, cold_start_windows=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_ids = None if target_ids is None else set(target_ids)
            self.cold_start_windows = cold_start_windows
            self.audit = dict(over_capacity=0, complete_windows=0)

        def _iterable_factory(self, metadata):
            instance = ContextIterable(metadata, self.args, self.path, self.parser, self.tokenizer,
                                       self.test, self.shared, self.sample_weights)
            instance.audit = self.audit
            instance.target_ids = self.target_ids
            instance.cold_start_windows = self.cold_start_windows
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
