"""Explicit deterministic development sensor faults, not native acquisition."""
from copy import deepcopy

import numpy as np

CASES=('narrow_depth','narrow_depth_bias_0_02','narrow_depth_bias_0_2','depth_dropout')
START_TICK=80
END_TICK=140
FRAME_COUNT=181


def perturb(policy,depth,case,*,first_ns):
    if case not in CASES: raise ValueError('fixed development perturbation required')
    p,d=deepcopy(policy),deepcopy(depth)
    now=p['sensor_state']['decision_ns']; offset=now-first_ns
    if offset<0 or offset%100_000_000: raise ValueError('fixed causal replay cadence')
    tick=offset//100_000_000
    if START_TICK<=tick<(START_TICK+3 if case=='depth_dropout' else END_TICK):
        if case=='depth_dropout': d['valid'][:]=False
        else:
            d['valid'][:,:280]=False; d['valid'][:,360:]=False
        d['depth_m'][~d['valid']]=0.
    bias=.02 if case=='narrow_depth_bias_0_02' else .2 if case=='narrow_depth_bias_0_2' else 0.
    force=p['sensor_state']['sensed']['specific_force']
    # Apply to measurement time, not arrival/tick: overlapping history values
    # remain identical across packets. Bias persists after depth recovery.
    selected=(force['measured_ns']>=first_ns+START_TICK*100_000_000)&force['valid'][:,1]
    force['values'][selected,1]+=bias
    return p,d
