"""Small learned chord preference, with inspectable features and coefficients.

It scores candidate lane subsets in a fixed rhythm. It has no authority to
change timing, object counts or hold lengths, and is not an audio model.
"""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import numpy as np

FEATURES = ("left_share", "right_share", "hand_imbalance", "same_hand_chord", "span",
    "adjacent_pairs", "outer_share", "centre_share", "overlap_previous", "overlap_two_back",
    "repeat_shape", "hand_switch", "centroid_motion", "short_jacks", "short_releases",
    "held_hand_pressure", "post_hold_imbalance", "lane_recent_load", "direction_reversal", "chord_gap")

LANE_FEATURES = ("attack", "hold_head", "new_duration", "since_attack", "since_release", "hold_remaining", "previous_row", "older_row", "valid_lane")
NETWORK_FEATURES = tuple(f"lane_{i}_{name}" for i in range(10) for name in LANE_FEATURES) + tuple(
    [f"keys_{k}" for k in range(6,11)] + ["chord_share", "previous_chord_share", "gap_beats", "previous_gap_beats", "mean_duration", "beat_seconds"])


def network_features(choices, keys, start, ends, last, history=(), durations=None, beat_ms=500):
    """State and candidate features for a compact learned nonlinear preference."""
    choices=np.asarray(choices,dtype=int)
    count,size=choices.shape
    durations=np.zeros(size) if durations is None else np.asarray(durations)
    value=np.zeros((count,10,len(LANE_FEATURES)),dtype=np.float32)
    batches=np.arange(count)[:,None]
    value[batches,choices,0]=1
    value[batches,choices,1]=durations>0
    value[batches,choices,2]=np.log1p(np.maximum(0,durations)/500)/np.log(17)
    value[:,:keys,3]=np.clip((start-last)/1000,0,2)
    value[:,:keys,4]=np.where(ends>last,np.clip((start-ends)/200,0,2),2)
    value[:,:keys,5]=np.clip((ends-start)/4000,0,2)
    value[:,:keys,8]=1
    recent=history[-1] if history and start-history[-1][0]<beat_ms*4 else None
    older=history[-2] if len(history)>1 and start-history[-2][0]<beat_ms*4 else None
    if recent:value[:,:keys,6]=[(recent[1]>>i)&1 for i in range(keys)]
    if older:value[:,:keys,7]=[(older[1]>>i)&1 for i in range(keys)]
    global_values=np.zeros((count,11),np.float32)
    global_values[:,keys-6]=1
    global_values[:,5]=size/keys
    global_values[:,6]=(recent[2]/keys) if recent else 0
    global_values[:,7]=min(4,(start-recent[0])/beat_ms)/4 if recent else 1
    global_values[:,8]=min(4,(recent[0]-older[0])/beat_ms)/4 if recent and older else 1
    global_values[:,9]=min(2,float(np.mean(durations))/4000)
    global_values[:,10]=beat_ms/1000
    return np.concatenate((value.reshape(count,-1),global_values),axis=1)


def mirror_features(x, keys):
    result=np.array(x,copy=True)
    lanes=result[:,:90].reshape(-1,10,len(LANE_FEATURES))
    lanes[:,:keys]=lanes[:,:keys][:,::-1].copy()
    return result


def score_network(x, layers):
    output=np.asarray(x,dtype=np.float32)
    for index,layer in enumerate(layers):
        weight,bias=np.asarray(layer['weight'],dtype=np.float32),np.asarray(layer['bias'],dtype=np.float32)
        output=output@weight.T+bias
        if index+1<len(layers):output=np.maximum(output,0)
    return output[:,0]


def features(choices, keys, start, ends, last, history=(), durations=None, beat_ms=500):
    """Vectorized features for many candidate chords with one fixed history."""
    choices = np.asarray(choices, dtype=int)
    count, size = choices.shape
    centre = (keys - 1) / 2
    left = choices < keys // 2
    right = choices >= (keys + 1) // 2
    share_l, share_r = left.mean(axis=1), right.mean(axis=1)
    span = (choices.max(axis=1) - choices.min(axis=1)) / max(1, keys - 1)
    adjacent = (np.diff(choices, axis=1) == 1).sum(axis=1) / max(1, size - 1)
    outer = ((choices == 0) | (choices == keys - 1)).mean(axis=1)
    middle = (abs(choices - centre) <= 1).mean(axis=1)
    previous = history[-1] if history else None
    two_back = history[-2] if len(history) > 1 else None
    previous_mask = previous[1] if previous and start - previous[0] <= beat_ms * 2 else 0
    previous_cols = [i for i in range(keys) if previous_mask & (1 << i)]
    older_mask = two_back[1] if two_back and start - two_back[0] <= beat_ms * 2 else 0
    overlap = np.array([[(previous_mask >> lane) & 1 for lane in range(keys)]])[0][choices].mean(axis=1)
    overlap2 = np.array([[(older_mask >> lane) & 1 for lane in range(keys)]])[0][choices].mean(axis=1)
    masks = np.sum(1 << choices, axis=1)
    repeat = (masks == previous_mask).astype(float) if previous_mask else np.zeros(count)
    previous_centre = np.mean(previous_cols) if previous_cols else centre
    motion = abs(choices.mean(axis=1) - previous_centre) / max(1, keys - 1) if previous_cols else np.zeros(count)
    switch = ((choices.mean(axis=1) - centre) * (previous_centre - centre) < 0).astype(float) if previous_cols else np.zeros(count)
    window = min(220., max(85., beat_ms * .42))
    jacks = np.maximum(0., 1 - (start - last[choices]) / window).mean(axis=1)
    releases = (np.maximum(0., 1 - (start - ends[choices]) / 80) * (ends[choices] > last[choices])).mean(axis=1)
    held_l = np.sum((ends > start) & (np.arange(keys) < keys // 2))
    held_r = np.sum((ends > start) & (np.arange(keys) >= (keys + 1) // 2))
    held_pressure = (share_l * held_l + share_r * held_r) / max(1, keys / 2)
    hold = np.asarray(durations) > 0 if durations is not None else np.zeros(size, dtype=bool)
    imbalance = abs(held_l + (left * hold).sum(axis=1) - held_r - (right * hold).sum(axis=1)) / keys
    recent = np.exp(-np.maximum(0, start - last[choices]) / max(100, beat_ms)).mean(axis=1)
    older_cols = [i for i in range(keys) if older_mask & (1 << i)]
    reversal = ((choices.mean(axis=1) - previous_centre) * (previous_centre - np.mean(older_cols)) < 0).astype(float) if older_cols and previous_cols else np.zeros(count)
    gap = min(4., max(0., (start - previous[0]) / beat_ms)) / 4 if previous else 1.
    return np.column_stack((share_l, share_r, abs(share_l - share_r),
        ((share_l == 1) | (share_r == 1)).astype(float) if size > 1 else np.zeros(count), span,
        adjacent, outer, middle, overlap, overlap2, repeat, switch, motion, jacks, releases,
        held_pressure, imbalance, recent, reversal, np.full(count, gap)))


@lru_cache(maxsize=1)
def load_prior():
    path = Path(__file__).resolve().parents[1] / "assets/pattern-prior.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("revision")==1 and value.get("features")==list(FEATURES):
            return value
        if value.get("revision")==2 and value.get("features")==list(NETWORK_FEATURES):
            width=len(NETWORK_FEATURES)
            for layer in value['network']:
                weight,bias=np.asarray(layer['weight'],dtype=np.float32),np.asarray(layer['bias'],dtype=np.float32)
                if weight.ndim!=2 or weight.shape[1]!=width or bias.shape!=(weight.shape[0],) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
                    return None
                width=len(bias)
                layer['weight'],layer['bias']=weight,bias
            return value if width==1 else None
    except (OSError,ValueError,KeyError,TypeError):
        return None
    return None


def prior_status():
    prior=load_prior()
    return dict(available=prior is not None, revision=prior.get('revision') if prior else None,
        keys=[int(k) for k,m in prior.get('models',{}).items() if m.get('enabled')] if prior else [],
        purpose='固定起止时间下的轨道偏好；不预测音频或 LN 长度')


def preference_cost(choices, keys, start, ends, last, history, durations, beat_ms, strength=.35):
    prior = load_prior()
    model = prior.get("models", {}).get(str(keys)) if prior else None
    if not model or not model.get("enabled", False):
        return np.zeros(len(choices))
    if prior['revision']==2:
        x=network_features(choices,keys,start,ends,last,history,durations,beat_ms)
        score=(score_network(x,prior['network'])+score_network(mirror_features(x,keys),prior['network']))/2
    else:
        x = features(choices, keys, start, ends, last, history, durations, beat_ms)
        score = x @ np.asarray(model["coefficients"], dtype=float)
    # A bounded tie preference cannot overrule physical hold occupancy or
    # dominate the hand/jack costs; unseen chord histories back off naturally.
    return -strength * np.tanh(score - np.median(score))
