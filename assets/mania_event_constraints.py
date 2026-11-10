"""Opt-in research decoding grammar for time-first mania MAP events.

Only syntax and held-column state are constrained. Times, sound selection and
LN ends remain model decisions. EOS never invents a release for an open hold.
"""
CODEC='mania-event-state-v1'


class ManiaEventState:
    def __init__(self,starts,ends,keys,eos,held=None,last_attack=None,*,snapping=True,hitsounds=True,sv=True):
        if keys not in range(6,11):raise ValueError('Experimental grammar supports 6–10K')
        self.starts=starts;self.ends=ends;self.keys=keys;self.eos=set(eos)
        self.held=dict(held or {});self.last_attack=dict(last_attack or {})
        self.snapping=snapping;self.hitsounds=hitsounds;self.sv=sv
        self.stage='start';self.time=None;self.column=None;self.snapped=False

    def span(self,name):return range(self.starts[name],self.ends[name])

    def columns(self,time=None):
        time=self.time if time is None else time
        return [self.starts['column']+c for c in range(self.keys)
                if self.last_attack.get(c)!=time and self.held.get(c,float('-inf'))<time]

    def allowed(self):
        s=self.starts
        if self.stage=='start':
            # Do not enter a timestamp at which every column is unavailable.
            # A complete chord/hold start can exhaust all columns at that time;
            # masking only its later column token would leave no continuation.
            return [t for t in self.span('t') if self.sv or self.columns((t-s['t'])*10)]+list(self.eos)
        if self.stage=='time':
            return ([*self.span('snap')] if self.snapping and (self.columns() or self.sv) else
                [] if self.snapping else self.columns())+(
                [s['column']+c for c in self.held if self.held[c]<self.time] if self.snapping else [])+(
                list(self.span('scroll_speed')) if self.sv else [])
        if self.stage=='snap':return self.columns()+(list(self.span('scroll_speed')) if self.sv else [])
        if self.stage=='column':
            if self.column in self.held:
                if not self.snapping:return [s['hold_note_end'],s['hold_note_sustain']]
                return [s['hold_note_end']] if self.snapped else [s['hold_note_sustain']]
            if self.hitsounds:return list(self.span('hitsound'))
            return [s['circle'],s['hold_note']]
        if self.stage=='hitsound':return list(self.span('volume'))
        if self.stage=='volume':return [s['circle'],s['hold_note']]
        if self.stage=='sv':return [s['scroll_speed_change']]
        raise ValueError('Unknown event stage')

    def consume(self,token):
        if token not in self.allowed():raise ValueError(f'Illegal mania event token {token} at {self.stage}')
        s=self.starts
        if token in self.eos:return
        if token in self.span('t'):
            self.time=(token-s['t'])*10;self.stage='time';self.snapped=False
        elif token in self.span('snap'):self.stage='snap';self.snapped=True
        elif token in self.span('column'):self.column=token-s['column'];self.stage='column'
        elif token in self.span('hitsound'):self.stage='hitsound'
        elif token in self.span('volume'):self.stage='volume'
        elif token in self.span('scroll_speed'):self.stage='sv'
        else:
            if token in (s['circle'],s['hold_note']):self.last_attack[self.column]=self.time
            if token==s['hold_note']:self.held[self.column]=self.time
            elif token==s['hold_note_end']:self.held.pop(self.column)
            self.stage='start'


def context_state(events,frame_time):
    """Use generated output history only; input sketch columns are not state."""
    held={};attacks={};time=None;column=None
    for kind,value in events:
        # Match Processor._encode: decoded events include a +5ms midpoint,
        # which must not make a repeated output token look like a new attack.
        if kind=='t':time=int((float(value)-frame_time)/10)*10
        elif kind=='column':column=int(value)
        elif kind in ('circle','hold_note','hold_note_end','hold_note_sustain'):
            if time is None or column is None:raise ValueError('Incomplete output-history event')
            if kind in ('circle','hold_note'):attacks[column]=time
            if kind=='hold_note':held[column]=time
            elif kind=='hold_note_end':held.pop(column,None)
            elif kind=='hold_note_sustain':held.setdefault(column,float('-inf'))
            time=None;column=None
        elif kind in ('scroll_speed_change','beat','measure','timing_point'):
            time=None;column=None
    return held,attacks


def install_runtime():
    import torch
    from osuT5.osuT5.inference.processor import Processor
    from osuT5.osuT5.inference import server
    from osuT5.osuT5.event import ContextType
    original_prepare=Processor.prepare_context_sequences
    original_generate=Processor.model_generate
    original_build=server.build_logits_processors
    current=[None]

    def prepare(self,contexts,frame_time,out_context,req_special_tokens):
        result=original_prepare(self,contexts,frame_time,out_context,req_special_tokens)
        if out_context:
            if len(contexts)!=1 or contexts[0]['context_type']!=ContextType.MAP:
                raise ValueError('State constraint requires a single MAP output')
            self._mania_event_state=context_state([(e.type.value,e.value) for e in contexts[0]['events']],frame_time)
        return result

    class Constraint:
        def __init__(self,state,prompt_length):self.state=state;self.seen=prompt_length
        def __call__(self,input_ids,scores):
            if input_ids.shape[0]!=1:raise ValueError('State constraint requires batch/beam size 1')
            for token in input_ids[0,self.seen:].tolist():self.state.consume(token)
            self.seen=input_ids.shape[1]
            allowed=self.state.allowed()
            masked=torch.full_like(scores,-torch.inf);masked[:,allowed]=scores[:,allowed]
            if not torch.isfinite(masked).any():
                raise ValueError('No finite legal mania event continuation: '+str(dict(
                    stage=self.state.stage,time=self.state.time,column=self.state.column,
                    held=self.state.held,attacks=self.state.last_attack,allowed=allowed,
                    finite=torch.isfinite(scores).sum().item(),recent=input_ids[0,-12:].tolist())))
            return masked

    def generate(self,model_kwargs,**kwargs):
        if self.types_first or self.num_beams!=1 or self.cfg_scale!=1 or self.fast_decoder_loop:
            raise ValueError('State constraint requires sequential time-first decoding without CFG')
        starts={k.value:v for k,v in self.tokenizer.event_start.items()}
        ends={k.value:v for k,v in self.tokenizer.event_end.items()}
        held,attacks=self._mania_event_state
        data=self.args.train.data
        state=ManiaEventState(starts,ends,int(self.args.keycount),
            [self.tokenizer.eos_id,self.tokenizer.context_eos[ContextType.MAP]],held,attacks,
            snapping=data.add_snapping,hitsounds=data.add_hitsounds,sv=data.add_mania_sv)
        current[0]=Constraint(state,model_kwargs['decoder_input_ids'].shape[1])
        try:return original_generate(self,model_kwargs,**kwargs)
        finally:current[0]=None

    def build(*args,**kwargs):
        result=original_build(*args,**kwargs)
        if current[0] is None:raise ValueError('Missing mania state for constrained decoding')
        result.append(current[0]);return result
    Processor.prepare_context_sequences=prepare
    Processor.model_generate=generate
    server.build_logits_processors=build
    print('ManiaLab research event constraints active: '+CODEC,flush=True)
