"""Exact pulse-target pairing; outcome labels and controller success are absent.

The caller must independently bind/audit the raw command tape and packets.
Requested commands condition the model; actuator slew is NOT called executed
velocity. Missing future execution or RGB is censored, not padded as evidence.
"""
from numbers import Integral
import numpy as np
import torch
from lewm.causal_subtrajectory_development import frame_lookup,HISTORY_OFFSETS_NS
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.rgb_body_tensor_interface_development import observation_tensors
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan,validate_timed_plan

TICK_NS=100_000_000


def pulse_window(frames,tape,*,departure_tick,departure_ns,command,pulse_ticks):
    """Keep every declared window, including censored/missing-history windows.

    Exact pipeline indexing: command tick zero starts at native sample749.
    A target at departure+k ticks consumes commands [departure,departure+k),
    never the action selected AT that target observation.
    """
    for v in (departure_tick,departure_ns):
        if isinstance(v,bool) or not isinstance(v,Integral) or v<0:raise ValueError('nonnegative integer departure required')
    lookup=frame_lookup(frames)
    if departure_ns not in lookup or departure_ns!=1_500_000_000+departure_tick*TICK_NS:
        raise ValueError('actual co-timed pulse departure required')
    blocks,mask=pulse_brake_plan(command,pulse_ticks)
    active,offsets=validate_timed_plan(blocks[None],mask[None],1)
    if not isinstance(tape,list):raise ValueError('explicit audited command tape required')
    for i,row in enumerate(tape):
        if (row['tick']!=i or row['pre_sample_index']!=749+i*50
                or type(row['completed']) is not bool
                or type(row['post_sample_index']) is not int
                or not row['pre_sample_index']<=row['post_sample_index']<=row['pre_sample_index']+50
                or (row['completed'] and row['post_sample_index']!=row['pre_sample_index']+50)):
            raise ValueError('invalid actual command interval')
        values=np.asarray(row['requested_command'],float)
        if values.shape!=(3,) or not np.isfinite(values).all():raise ValueError('finite requested command required')
    history=[lookup.get(departure_ns+d) for d in HISTORY_OFFSETS_NS]
    expected=[list(command)]*pulse_ticks+[[0.,0.,0.]]*20
    targets=[]
    for enabled,offset in zip(active[0].tolist(),offsets[0].tolist(),strict=True):
        if not enabled:
            targets.append(dict(offset_ns=0,target_ns=None,command_prefix_executed=False,
                observation_available=False,future_observation_index=None,future_valid=False,reason='UNKNOWN_PLAN'))
            continue
        ticks=offset//TICK_NS;end=departure_tick+ticks
        present=end<=len(tape)
        executed=present and all(row['completed'] and row['requested_command']==c
            for row,c in zip(tape[departure_tick:end],expected[:ticks],strict=True))
        ns=int(departure_ns+offset);index=lookup.get(ns)
        reason='AVAILABLE' if executed and index is not None else 'UNEXECUTED_PREFIX' if not executed else 'MISSING_RAW_OBSERVATION'
        targets.append(dict(offset_ns=offset,target_ns=ns,command_prefix_executed=executed,
            observation_available=index is not None,future_observation_index=index,
            future_valid=executed and index is not None,reason=reason))
    return dict(departure_tick=int(departure_tick),decision_ns=int(departure_ns),command=list(command),pulse_ticks=pulse_ticks,
        history_observation_indices=history,history_ready=all(i is not None for i in history),targets=targets)


def observation_pair_tensors(reader,window):
    """Target-side materialization, never a model forward or training decision.

    No dependence on visual-estimator availability, contact, native pose or
    final mission success. Physical outcome masks/labels must be added separately.
    """
    if not window['history_ready'] or any(i is None for i in window['history_observation_indices']):
        raise ValueError('cannot fabricate missing past RGB')
    packets=[reader.packet(i)[0] for i in window['history_observation_indices']]
    history=causal_history_tensors(packets,window['decision_ns'])
    blocks,mask=pulse_brake_plan(tuple(window['command']),window['pulse_ticks'])
    active,offsets=validate_timed_plan(blocks[None],mask[None],1)
    future={k:torch.zeros((8,*v.shape[1:]),dtype=v.dtype) for k,v in history.items()}
    valid=torch.zeros(8,dtype=torch.bool)
    if len(window['targets'])!=8:raise ValueError('all eight target slots required')
    for i,t in enumerate(window['targets']):
        if t['offset_ns']!=int(offsets[0,i]):raise ValueError('pulse target time mismatch')
        expected=window['decision_ns']+t['offset_ns'] if active[0,i] else None
        if t['target_ns']!=expected or t['future_valid']!=(bool(active[0,i]) and t['command_prefix_executed'] and t['observation_available']):
            raise ValueError('target validity mismatch')
        if not t['future_valid']:continue
        policy=reader.packet(t['future_observation_index'])[0]
        if policy['image']['measured_ns']!=expected or policy['sensor_state']['decision_ns']!=expected:
            raise ValueError('actual future packet timestamp mismatch')
        if policy['sensor_state']['identity']!=packets[-1]['sensor_state']['identity']:
            raise ValueError('cross-episode future target forbidden')
        values=observation_tensors(policy)
        for k,v in values.items():future[k][i]=v
        valid[i]=True
    return dict(inputs=dict(observation_history=history,known_action_blocks=blocks,known_action_valid=mask),
                targets=dict(future_observations=future,future_valid=valid,target_offsets_ns=offsets[0]))
