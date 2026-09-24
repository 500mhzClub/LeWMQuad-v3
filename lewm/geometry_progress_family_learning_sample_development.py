"""Audited layout-family packets and labels -> existing JEPA tensor contract, no fit.

Caller must authenticate the complete raw-audit receipt and episode artifacts.
No directory discovery, native-state reading, sibling-context substitution,
role reassignment, optimizer steps or outcome-conditioned episode exclusions.
"""
import numpy as np
import torch
from lewm.geometry_progress_layout_family_development import assignments, timed_candidate
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.rgb_body_tensor_interface_development import observation_tensors


def materialize(reader, report):
    trial=report['trial']
    if trial not in assignments() or any(report[k]!=v for k,v in assignments()[trial].items()):
        raise ValueError('exact prospective episode/action assignment required')
    if not report['raw_sensor_reconstruction_pass'] or not report['command_stop_replay_pass']:
        raise ValueError('completed raw replay required before tensorization')
    labels=report['targets']
    if labels is None:
        raise ValueError('missing departure has no fabricated model sample')
    if (labels['target_only'] is not True or labels['departure_tick']!=3
            or labels['departure_ns']!=1_800_000_000 or labels['history_observation_indices']!=[0,1,2,3]
            or len(labels['targets'])!=8):
        raise ValueError('exact actual departure and eight horizon slots required')
    packets=[reader.packet(i)[0] for i in range(4)]
    history=causal_history_tensors(packets,1_800_000_000)
    blocks,valid=timed_candidate(report['action'])
    future={k:torch.zeros((8,*v.shape[1:]),dtype=v.dtype) for k,v in history.items()}
    motion=torch.full((8,3),float('nan'));contact=torch.full((8,),float('nan'))
    mv=torch.zeros(8,dtype=torch.bool);cv=mv.clone();fv=mv.clone()
    offsets=torch.arange(1,9,dtype=torch.int64)*500_000_000
    positive_seen=False
    for i,t in enumerate(labels['targets']):
        if t['offset_ns']!=int(offsets[i]) or any(type(t[k]) is not bool for k in
            ('motion_valid','contact_valid','future_image_valid')):
            raise ValueError('exact clocks and independent boolean modality masks required')
        mv[i],cv[i],fv[i]=t['motion_valid'],t['contact_valid'],t['future_image_valid']
        if t['contact_valid']:
            if type(t['contact']) is not float or t['contact'] not in (0.,1.):
                raise ValueError('binary measured contact label required')
            if positive_seen and t['contact']==0.:raise ValueError('cumulative contact cannot revert')
            positive_seen|=t['contact']==1.;contact[i]=t['contact']
        elif t['contact'] is not None:raise ValueError('unknown contact must stay null')
        if t['motion_valid']:
            values=np.asarray(t['motion'],float)
            if not t['contact_valid'] or t['contact']!=0. or values.shape!=(3,) or not np.isfinite(values).all():
                raise ValueError('finite contact-free native motion label required')
            motion[i]=torch.tensor(values,dtype=torch.float32)
        elif t['motion'] is not None:raise ValueError('missing motion must stay null')
        index=t['future_observation_index']
        if t['future_image_valid']:
            if type(index) is not int or index!=3+5*(i+1):
                raise ValueError('future packet must be the actual cumulative boundary frame')
            packet=reader.packet(index)[0];now=1_800_000_000+int(offsets[i])
            if (packet['image']['measured_ns']!=now or packet['sensor_state']['decision_ns']!=now
                    or packet['sensor_state']['identity']!=packets[-1]['sensor_state']['identity']):
                raise ValueError('actual same-episode future packet clock required')
            for k,v in observation_tensors(packet).items():future[k][i]=v
        elif index is not None:raise ValueError('missing future image must have no substituted index')
    return dict(inputs=dict(observation_history=history,known_action_blocks=blocks,known_action_valid=valid),
        targets=dict(motion=motion,motion_valid=mv,contact=contact,contact_valid=cv,
            future_observations=future,future_valid=fv,target_offsets_ns=offsets))
