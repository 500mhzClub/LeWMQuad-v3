"""Same past tensors and separately admitted actual short-horizon targets."""
import numpy as np
import torch
from lewm.observation_horizon_plan_development import plan
from lewm.rgb_body_tensor_interface_development import observation_tensors


def inference_inputs(original_inputs,row):
    if set(original_inputs)!={'observation_history','known_action_blocks','known_action_valid'}:
        raise ValueError('exact original policy-only input fields required')
    offset=row['offset_ticks'] if row['source']=='family' else 0
    blocks,valid=plan(row['action'],offset_ticks=offset)
    if (not torch.equal(blocks[:,0],original_inputs['known_action_blocks'].reshape(40,3)[:8])
            or not torch.equal(valid[:,0],original_inputs['known_action_valid'].reshape(40)[:8])):
        raise ValueError('short plan must be exact actual prefix of admitted original plan')
    return dict(observation_history=original_inputs['observation_history'],known_action_blocks=blocks,known_action_valid=valid)


def materialize_training(reader,row,inputs):
    if row['data_role']!='train' or row['available'] is not True:
        raise ValueError('future-image materialization requires an available training context')
    history=inputs['observation_history'];receipt=row['observation_horizon_receipt']
    future={k:torch.zeros((8,*v.shape[1:]),dtype=v.dtype) for k,v in history.items()}
    motion=torch.full((8,3),float('nan'));contact=torch.full((8,),float('nan'))
    mv=torch.zeros(8,dtype=torch.bool);cv=mv.clone();fv=mv.clone();offsets=torch.zeros(8,dtype=torch.int64)
    seen=False
    if len(row['targets'])!=8:raise ValueError('complete eight short target slots required')
    for i,t in enumerate(row['targets']):
        active=bool(inputs['known_action_valid'][i,0]);ns=(i+1)*100_000_000 if active else 0
        if (type(t['in_plan']) is not bool or t['in_plan']!=active or t['offset_ns']!=ns
                or any(type(t[k]) is not bool for k in ('motion_valid','contact_valid','future_image_valid'))
                or not active and any(t[k] for k in ('motion_valid','contact_valid','future_image_valid'))):
            raise ValueError('exact short-plan masks and independent actual target validity required')
        offsets[i]=ns;mv[i]=t['motion_valid'];cv[i]=t['contact_valid'];fv[i]=t['future_image_valid']
        if cv[i]:
            if type(t['contact']) is not float or t['contact'] not in (0.,1.) or seen and t['contact']==0.:
                raise ValueError('measured cumulative contact cannot revert')
            seen|=t['contact']==1.;contact[i]=t['contact']
        elif t['contact'] is not None:raise ValueError('unknown contact must remain null')
        if mv[i]:
            values=np.asarray(t['motion'],float)
            if not cv[i] or t['contact']!=0. or values.shape!=(3,) or not np.isfinite(values).all():
                raise ValueError('finite contact-free native motion required')
            motion[i]=torch.tensor(values,dtype=torch.float32)
        elif t['motion'] is not None:raise ValueError('missing motion must remain null')
        index=t['future_observation_index']
        if fv[i]:
            if not mv[i] or type(index) is not int or index!=receipt['departure_tick']+i+1:
                raise ValueError('actual contact-free future packet boundary required')
            packet=reader.packet(index)[0];now=receipt['departure_ns']+ns
            if (packet['image']['measured_ns']!=now or packet['sensor_state']['decision_ns']!=now
                    or tuple(packet['sensor_state']['identity'])!=(0,0,0)):
                raise ValueError('actual same-episode future packet clock required')
            for k,v in observation_tensors(packet).items():future[k][i]=v
        elif index is not None:raise ValueError('missing future image cannot substitute an index')
    return dict(inputs=inputs,targets=dict(motion=motion,motion_valid=mv,contact=contact,contact_valid=cv,
        future_observations=future,future_valid=fv,target_offsets_ns=offsets))
