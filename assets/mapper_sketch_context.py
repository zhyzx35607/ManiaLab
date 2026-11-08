"""Explicit experimental source-sketch codec for the pinned mania checkpoint.

That checkpoint has MAP delimiters but no GD delimiters. Reuse the MAP pair
for an input-only sketch before the target MAP; do not shift event IDs or
resize embeddings. This is a NEW adapter training task, not native GD support.
"""
CODEC = 'map-delimited-sketch-v1'


def tokenizer_with_sketch(state):
    result=dict(state)
    for name in ('context_sos','context_eos'):
        mapping={getattr(k,'value',k):v for k,v in state[name].items()}
        if 'map' not in mapping:
            raise ValueError('Sketch codec requires pretrained MAP delimiters')
        if 'gd' in mapping and mapping['gd']!=mapping['map']:
            raise ValueError('Checkpoint has different GD semantics; use a matching codec')
        mapping['gd']=mapping['map'];result[name]=mapping
    return result


def install_runtime():
    """Patch only the isolated optional-engine process; upstream files stay pinned."""
    # The upstream inference entry point imports the nested namespace. The
    # training launcher uses a different sys.path; codec state is namespace-free.
    from osuT5.osuT5.tokenizer import Tokenizer
    from osuT5.osuT5.event import ContextType
    from osuT5.osuT5.inference.processor import Processor
    load_state=Tokenizer.load_state_dict
    def load(self,state):
        return load_state(self,tokenizer_with_sketch(state))
    Tokenizer.load_state_dict=load
    choose=Processor._get_viable_template
    def choose_checked(self,*args,**kwargs):
        result=choose(self,*args,**kwargs)
        if ContextType.GD not in result[0]:
            raise ValueError('Requested sketch input was silently discarded by the upstream template')
        return result
    Processor._get_viable_template=choose_checked
    get_context=Processor.get_in_context
    def get_checked(self,**kwargs):
        result=get_context(self,**kwargs)
        sketches=[r for r in result if r['context_type']==ContextType.GD]
        if len(sketches)!=1 or not sketches[0]['events']:
            raise ValueError('The source sketch did not enter decoder input')
        print('ManiaLab sketch context active: '+str(len(sketches[0]['events']))+' events; '+CODEC,flush=True)
        return result
    Processor.get_in_context=get_checked
    generate=Processor.model_generate
    def generate_traced(self,model_kwargs,**kwargs):
        result=generate(self,model_kwargs,**kwargs)
        # Research trace contains the supplied sketch/generated tokens and is
        # kept with private run artifacts, never in the application response.
        import json
        from pathlib import Path
        prompt=model_kwargs['decoder_input_ids'][0].detach().cpu().tolist()
        row=dict(prompt=prompt,generated=result[0][0,len(prompt):].detach().cpu().tolist(),
            context=kwargs.get('context_type'),lookback_time=kwargs.get('lookback_time'),
            lookahead_time=kwargs.get('lookahead_time'))
        with (Path(self.args.output_path)/'sketch-decoder-trace.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        return result
    Processor.model_generate=generate_traced
