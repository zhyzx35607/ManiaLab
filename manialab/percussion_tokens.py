"""Research grammar for ADT_STR custom percussion tokens, never GM pitches.

Only token type, physical chunk bounds and duplicate events are constrained.
No beat grid, chart density, class whitelist by song or reference is supplied.
"""
import numpy as np


def audit_tokens(tokens, max_time_index=255):
    """Inspect the whole sequence, including repetitions and data after EOS."""
    tokens=[int(t) for t in tokens]
    if not 0<=max_time_index<=255:raise ValueError('Invalid chunk clock')
    if not tokens or tokens[0]!=2:return ['missing_bos']
    issues=[]
    if 3 in tokens:
        stop=tokens.index(3)
        if any(t!=3 for t in tokens[stop:]):issues.append('events_after_eos')
    else:
        issues.append('missing_eos');stop=len(tokens)
    body=tokens[1:stop]
    if not body or body==[0]:return issues
    if len(body)%3:issues.append('incomplete_triple')
    seen=set();last=-1
    for i in range(0,len(body)-2,3):
        t,p,v=body[i:i+3]
        if not 4<=t<=4+max_time_index:issues.append('time_outside_chunk')
        if not 335<=p<=360:issues.append('invalid_custom_percussion')
        if not 400<=v<=527:issues.append('invalid_velocity')
        if t<last:issues.append('decreasing_timestamp')
        if (t,p) in seen:issues.append('duplicate_percussion_event')
        seen.add((t,p));last=t
    return issues


def legal_next(prefix,vocabulary=1400,max_time_index=255):
    prefix=[int(t) for t in prefix]
    if not prefix or prefix[0]!=2 or not 0<=max_time_index<=255 or vocabulary<528:
        raise ValueError('Invalid percussion prefix or chunk clock')
    legal=np.zeros(vocabulary,bool)
    body=prefix[1:]
    if 3 in body:
        if any(t!=3 for t in body[body.index(3):]):raise ValueError('Events after EOS')
        legal[3]=True;return legal
    if body and body[0]==0:
        if len(body)!=1:raise ValueError('Silence must end the chunk')
        legal[3]=True;return legal
    completed=len(body)//3;seen=set();last_time=-1
    for i in range(completed):
        t,p,v=body[3*i:3*i+3]
        if (not 4<=t<=4+max_time_index or not 335<=p<=360 or not 400<=v<=527
                or t<last_time or (t,p) in seen):
            raise ValueError('Malformed, unordered or duplicate percussion event')
        seen.add((t,p));last_time=t
    pending=body[completed*3:]
    if not pending:
        legal[3]=True
        if not completed:legal[0]=True
        legal[max(4,last_time):5+max_time_index]=True
        if last_time>=4 and sum(t==last_time for t,p in seen)==26:legal[last_time]=False
    elif len(pending)==1:
        t=pending[0]
        if not max(4,last_time)<=t<=4+max_time_index:raise ValueError('Invalid pending timestamp')
        legal[335:361]=True
        for seen_t,p in seen:
            if seen_t==t:legal[p]=False
    else:
        t,p=pending
        if not max(4,last_time)<=t<=4+max_time_index or not 335<=p<=360 or (t,p) in seen:
            raise ValueError('Invalid pending pitch')
        legal[400:528]=True
    if not legal.any():raise ValueError('No legal percussion continuation')
    return legal


def decode_complete(tokens):
    """Parse exact (time,pitch,velocity) triples; no zip-based reassociation."""
    tokens=[int(t) for t in tokens]
    issues=audit_tokens(tokens)
    if issues:raise ValueError('Invalid percussion sequence: '+', '.join(sorted(set(issues))))
    body=tokens[1:tokens.index(3)]
    if not body or body==[0]:return []
    if len(body)%3:raise ValueError('Incomplete percussion event')
    rows=[]
    for i in range(0,len(body),3):
        t,p,v=body[i:i+3]
        if not 4<=t<260 or not 335<=p<=360 or not 400<=v<=527:raise ValueError('Invalid percussion token')
        rows.append(dict(onset=(t-4)/100,custom_pitch=p-300,velocity=v-400))
    return rows
