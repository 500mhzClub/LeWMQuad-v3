"""Target-side window indices and censored physical labels; never planner inputs."""
import math
from numbers import Integral

import numpy as np

from lewm.counterfactual_maze_development import HORIZONS_NS
from lewm.physical_execution_development import rotation_xyzw

OFFSETS_NS=tuple(range(0,4_000_000_000,500_000_000))
HISTORY_OFFSETS_NS=(-300_000_000,-200_000_000,-100_000_000,0)


def frame_lookup(frames):
    result={}
    for i,row in enumerate(frames):
        ns=row['image_ns']
        if isinstance(ns,bool) or not isinstance(ns,Integral) or ns<0 or row['decision_ns']!=ns or ns in result:
            raise ValueError('invalid or duplicate causal frame clock')
        if result and ns<=next(reversed(result)): raise ValueError('unordered frame clock')
        result[int(ns)]=i
    return result


def branch_windows(raw,start_index,frames,canonical_frames):
    """Only fixed-action branch intervals; privileged pose appears solely in labels."""
    ns=np.rint(np.asarray(raw['timestamp_s'])*1e9).astype(np.int64)
    pose=np.asarray(raw['base_pose_world']); contact=np.asarray(raw['physics_contact'],dtype=bool)
    if (ns.ndim!=1 or len(ns)==0 or pose.shape!=(len(ns),7) or contact.shape!=ns.shape
            or not np.isfinite(pose).all() or np.any(np.diff(ns)!=2_000_000)):
        raise ValueError('invalid raw physical series')
    if isinstance(start_index,bool) or not isinstance(start_index,Integral) or not 0<=start_index<len(ns):
        raise ValueError('invalid branch start')
    if contact[:start_index+1].any(): raise ValueError('branch starts after contact')
    lookup=frame_lookup(frames); canonical=frame_lookup(canonical_frames)
    t0=int(ns[start_index]); end=t0+4_000_000_000
    physical={int(t):i for i,t in enumerate(ns)}
    contacts=np.flatnonzero(contact); first=None if not len(contacts) else int(ns[contacts[0]])
    windows=[]
    for offset in OFFSETS_NS:
        current=t0+offset
        if current not in physical or (first is not None and first<=current): continue
        history=canonical if offset==0 else lookup
        times=[current+delta for delta in HISTORY_OFFSETS_NS]
        if any(t not in history for t in times): raise ValueError('causal past image missing')
        origin=pose[physical[current]]; rotation=rotation_xyzw(origin[3:])
        yaw0=math.atan2(rotation[1,0],rotation[0,0]); targets=[]
        for horizon in HORIZONS_NS:
            target=current+horizon; in_plan=target<=end
            observed=target in physical
            motion_valid=bool(in_plan and observed and (first is None or target<first))
            event=bool(in_plan and first is not None and first<=target)
            contact_valid=bool(in_plan and (target<=int(ns[-1]) or event))
            motion=None; future_index=None
            if motion_valid:
                if target not in lookup: raise ValueError('valid physical target lacks actual RGB')
                future_index=lookup[target]; future=pose[physical[target]]
                r=rotation_xyzw(future[3:]); dyaw=math.atan2(r[1,0],r[0,0])-yaw0
                delta=rotation.T@(future[:3]-origin[:3])
                motion=[float(delta[0]),float(delta[1]),math.atan2(math.sin(dyaw),math.cos(dyaw))]
            targets.append({'horizon_ns':horizon,'in_plan':in_plan,'motion_valid':motion_valid,
                'delta_xy_yaw_current_body':motion,'future_observation_index':future_index,
                'contact_valid':contact_valid,'contact_by_horizon':event if contact_valid else None})
        windows.append({'offset_ns':offset,'decision_ns':current,'history_observation_indices':[history[t] for t in times],
            'remaining_ticks':(end-current)//100_000_000,'targets':targets})
    return windows
