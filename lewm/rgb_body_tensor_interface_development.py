"""Convert an already validated causal observation into explicit model tensors."""
import numpy as np
import torch
import torch.nn.functional as F

from lewm.simulated_body_observation_development import validate_policy_packet


def observation_tensors(packet):
    validate_policy_packet(packet)
    state=packet['sensor_state']
    body_values,body_valid,body_age=[],[],[]
    scales={'gyro':np.array([2.]*3),'specific_force':np.array([9.81]*3),'joints':np.array([1.]*12+[10.]*12)}
    for name in ('gyro','specific_force','joints'):
        row=state['sensed'][name]
        body_values.append(row['values']/scales[name])
        body_valid.append(row['valid'].astype(np.float32))
        body_age.append(np.where(row['measured_ns']>=0,(state['decision_ns']-row['measured_ns'])/1e9,.4)[:,None])
    body=np.concatenate([*body_values,*body_valid,*body_age],axis=1)
    row=state['control']['applied_command']
    age=np.where(row['measured_ns']>=0,(state['decision_ns']-row['measured_ns'])/1e9,1.5)[:,None]
    control=np.concatenate([row['values']/np.array([.3,1.,.5]),row['valid'].astype(np.float32),age],axis=1)
    rgb=torch.from_numpy(packet['image']['rgb'].copy()).permute(2,0,1).float()[None]/255.
    return {'rgb':F.interpolate(rgb,size=(96,128),mode='area')[0],
        'body':torch.from_numpy(body.astype(np.float32)), 'control':torch.from_numpy(control.astype(np.float32))}
