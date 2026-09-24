"""Evaluation-only prefix identities and observed fixed three-second targets."""
import hashlib
import json
import math

import numpy as np

from lewm.causal_subtrajectory_development import frame_lookup,HISTORY_OFFSETS_NS
from lewm.counterfactual_maze_development import HORIZONS_NS
from lewm.physical_execution_development import rotation_xyzw
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import array_binding


def reference_context(directory,member,window):
    """Read only explicit already-audited source artifacts; never a policy input."""
    raw=histories=camera=None
    for name in ('physics_trace.npz','policy_histories.npz','camera_audit.json'):
        path=directory/name
        if path.resolve().parent!=directory.resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=member['artifact_sha256'][name]:
            raise ValueError('reference artifact binding changed')
        if name=='camera_audit.json': camera=json.loads(path.read_text())
        else:
            with np.load(path,allow_pickle=False) as archive: values={k:archive[k] for k in archive.files}
            if name=='physics_trace.npz': raw=values
            else: histories=values
    if window['offset_ns']!=1_000_000_000 or window['scene_id']!=member['scene_id'] or member['action_index'] not in range(1,5):
        raise ValueError('exact one-second moving reference required')
    start=member['prefix_terminal_sample_index']+500
    packet=window['history_observation_indices'][-1]
    if (start>=len(raw['timestamp_s']) or camera[packet]['physical_sample_index']!=start
            or raw['physics_contact'][:start+1].any()
            or int(round(float(raw['timestamp_s'][start])*1e9))!=window['decision_ns']):
        raise ValueError('unavailable moving reference')
    prefix={k:v[:start+1] for k,v in raw.items()}
    binding={'physics_arrays':array_binding(prefix),
        'history_arrays':array_binding({k:v[packet] for k,v in histories.items()}),
        'rgb_pixels_sha256':camera[packet]['rgb_sha256'],'physics_samples':start+1,'timestamp_ns':window['decision_ns']}
    return {'scene_id':member['scene_id'],'layout_id':member['layout_id'],'data_role':member['data_role'],
        'branch_start_observation_index':packet,'prefix_terminal_sample_index':start,
        'teacher_terminal_sample_index':member['prefix_terminal_sample_index'],'prefix_binding':binding,
        'history_observation_indices':window['history_observation_indices']}


def suffix_targets(raw,start_index,end_index,frames):
    """No release samples or commands beyond the known three-second suffix."""
    if isinstance(start_index,bool) or isinstance(end_index,bool) or not 0<=start_index<=end_index<len(raw['timestamp_s']):
        raise ValueError('observed suffix bounds')
    times=np.rint(np.asarray(raw['timestamp_s'])*1e9).astype(np.int64)
    if np.any(np.diff(times)!=2_000_000): raise ValueError('physical clock')
    pose=raw['base_pose_world']; contact=np.asarray(raw['physics_contact'],dtype=bool)
    if contact[:start_index+1].any(): raise ValueError('conditioning state at/after contact')
    t0=int(times[start_index]); lookup=frame_lookup(frames)
    physical={int(t):i for i,t in enumerate(times[:end_index+1])}
    history=[lookup[t0+d] for d in HISTORY_OFFSETS_NS]
    indices=np.flatnonzero(contact[start_index+1:end_index+1])+start_index+1
    first=int(times[indices[0]]) if len(indices) else None
    rotation=rotation_xyzw(pose[start_index,3:]); yaw0=math.atan2(rotation[1,0],rotation[0,0])
    targets=[]
    for horizon in HORIZONS_NS:
        target=t0+horizon; in_plan=horizon<=3_000_000_000
        event=bool(in_plan and first is not None and first<=target)
        cv=bool(in_plan and (target<=times[end_index] or event))
        mv=bool(in_plan and target in physical and (first is None or target<first))
        delta=future_index=None
        if mv:
            if target not in lookup: raise ValueError('valid future state lacks actual RGB')
            future=pose[physical[target]]; r=rotation_xyzw(future[3:]); dyaw=math.atan2(r[1,0],r[0,0])-yaw0
            movement=rotation.T@(future[:3]-pose[start_index,:3])
            delta=[float(movement[0]),float(movement[1]),math.atan2(math.sin(dyaw),math.cos(dyaw))]
            future_index=lookup[target]
        targets.append({'horizon_ns':horizon,'in_plan':in_plan,'motion_valid':mv,
            'delta_xy_yaw_current_body':delta,'future_observation_index':future_index,
            'contact_valid':cv,'contact_by_horizon':event if cv else None})
    return {'decision_ns':t0,'history_observation_indices':history,'remaining_ticks':30,'targets':targets}
