"""Frozen audit-only V4 adapters. No physics, fitting or live-controller mutation."""
import copy
import hashlib
import math
import pickle
import threading
import zlib
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.decision_headroom_reference_development import fixed_target_world, wrap
from lewm.dense_native_observation_development import dense_native_context
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.physical_execution_development import rotation_xyzw
from lewm.short_pulse_navigation_runtime_development import command_predictions, past_commands
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts import run_go2_dense_horizon_navigation_development as deployed

ACTIONS=('hold','forward','left_arc','right_arc','left_turn','right_turn')
PHASES=('outbound_exploration','goal_approach_settle','return')
PERMUTATION=np.array([1,2,3,4,5,0])


def active_objective(packet):
    """Names the actual captured objective, without inventing a view cost."""
    state=packet['controller_state'];mission=state['mission']
    terminal=packet['source_selection'].get('prefix_aware_terminal_approach',{}).get('terminal_mode',state.get('terminal_position_approach',False))
    phase=('return' if mission.phase=='RETURN' else 'goal_approach_settle'
           if terminal or (mission.last or {}).get('hold_required')
           else 'outbound_exploration')
    target=packet.get('active_objective',packet.get('active_target'))
    positional=target is not None and target['kind'] in ('initial_frame_xy','observed_map_xy')
    return dict(phase=phase,mode=('positional_terminal' if terminal else 'positional_route') if positional else 'view_seeking',
                target=copy.deepcopy(target),reference_regret_applicable=positional)


class PhaseReservoir:
    """Priority sampling; at most 24 candidates per phase, only final members saved."""
    def __init__(self, seed, run_id, per_run=24):
        self.seed=seed;self.run_id=run_id;self.quota=per_run
        self.counts=dict.fromkeys(PHASES,0);self.rows={p:[] for p in PHASES};self.log=[]

    def consider(self, frame, phase):
        if phase not in PHASES:raise ValueError('fixed phase required')
        self.counts[phase]+=1
        key=int.from_bytes(hashlib.sha256(f'{self.seed}/{self.run_id}/{frame}'.encode()).digest(),'big')
        row=dict(frame=int(frame),phase=phase,key=key)
        candidates=self.rows[phase]+[row]
        keep=sorted(candidates,key=lambda r:(r['key'],r['frame']))[:self.quota]
        replaced=[r['frame'] for r in self.rows[phase] if r not in keep]
        admitted=row in keep;self.rows[phase]=keep
        self.log.append(dict(frame=int(frame),phase=phase,priority_hex=f'{key:064x}',reservoir_admitted=admitted))
        return admitted,replaced

    def final(self):
        quotas={p:min(8,self.counts[p]) for p in PHASES}
        while sum(quotas.values())<min(self.quota,sum(self.counts.values())):
            for p in PHASES:
                if sum(quotas.values())>=self.quota:break
                if quotas[p]<self.counts[p]:quotas[p]+=1
        return [dict(r,inclusion_probability=quotas[p]/self.counts[p],
                     weight=self.counts[p]/quotas[p],population=self.counts[p],phase_quota=quotas[p])
                for p in PHASES for r in self.rows[p][:quotas[p]]]


def encode_snapshot(value):
    raw=pickle.dumps(value,protocol=4)
    if len(raw)>64*1024**2:raise ValueError('UNRESOLVED_SNAPSHOT_OVER_64_MIB')
    packed=zlib.compress(raw,1)
    if len(packed)>16*1024**2:raise ValueError('UNRESOLVED_COMPRESSED_SNAPSHOT_OVER_16_MIB')
    return packed,dict(raw_bytes=len(raw),bytes=len(packed),sha256=hashlib.sha256(packed).hexdigest(),codec='zlib1_pickle4')


def outcome_tensor(motion):
    m=np.asarray(motion,np.float32)
    if m.shape!=(6,8,3) or not np.isfinite(m).all():raise ValueError('finite six by eight XY/yaw required')
    return np.concatenate((m[:,:,:2],np.sin(m[:,:,2:3]),np.cos(m[:,:,2:3]),np.full((6,8,1),-1000,np.float32)),axis=-1)


class FixedMotionModel(torch.nn.Module):
    """Only allowed motion enters the scorer; no physical geometry/contact inputs."""
    def __init__(self, packet, motion, identity):
        super().__init__();self.packet=packet;self.motion=outcome_tensor(motion);self.readout_identity=identity
    def set_native_context(self, packets, *, observed_ns):
        if observed_ns!=self.packet['measured_ns']:raise ValueError('wrong decision clock')
    def forward(self, **inputs):
        tape=(inputs['known_action_blocks'][:,:,0]*torch.tensor([.3,1.,.5])).cpu().numpy()
        np.testing.assert_allclose(tape,self.packet['candidate_requested_commands'],atol=1e-7,rtol=0)
        return dict(rollout_outcomes=torch.from_numpy(self.motion.copy()),
                    target_offsets_ns=torch.arange(1,9).mul(100_000_000).expand(6,8),
                    prediction_valid=torch.ones(6,8,dtype=torch.bool),contact_prediction_available=False)


def controller_from_packet(packet, model, *, reactive=False):
    cls=deployed.DenseReactiveNavigationRuntime if reactive else deployed.DenseNavigationRuntime
    ctrl=object.__new__(cls);ctrl.__dict__.update(copy.deepcopy(packet['controller_state']))
    ctrl.model=model;ctrl.model_input_hook=model.register_forward_pre_hook(ctrl._check_model_inputs,with_kwargs=True);ctrl.lock=threading.RLock();ctrl.correction_pose_lock=threading.RLock()
    ctrl.profile_current=None;ctrl.plan_profile_rows=[];ctrl.events=[];ctrl.visual_dispatch_events=[]
    return ctrl


def selector(packet, motion, *, reactive=False, identity=None):
    """Call the unchanged deployed chain; substitute only its corrected motion."""
    identity=identity or dict(training_horizons_ms=list(range(100,801,100)))
    ctrl=controller_from_packet(packet,FixedMotionModel(packet,motion,identity),reactive=reactive)
    def substituted(self,prediction,*args):return prediction,dict(audit_motion_substitution=True)
    ctrl._correct_prediction=MethodType(substituted,ctrl)
    selected,_=ctrl._select_action(SimpleNamespace(**copy.deepcopy(packet['packet'])),
        copy.deepcopy(packet['evidence']),copy.deepcopy(packet['committed_prefix']),
        copy.deepcopy(packet['route_target_body_xy']),packet['scan_error_rad'],
        copy.deepcopy(packet['observed_map']),np.array(packet['observed_position']),np.array(packet['observed_rotation']))
    ctrl.model_input_hook.remove()
    return selected


def eligibility(packet, selection):
    """Candidate gates are separate from the selector's recovery/score overrides."""
    obs_allowed=set(('hold','left_turn','right_turn') if packet['scan_error_rad'] is not None else ACTIONS)
    memory={r['action']:r for r in selection['memory_forecast_candidates']}
    stopping={r['action']:r for r in selection['planned_stopping_projection']['candidates']}
    rows=[]
    for action in ACTIONS:
        reasons=[]
        if action not in obs_allowed:reasons.append('OBSERVATION_SCAN_VIEW_SUBSET')
        if not memory[action]['nominal_predicted_path_clear']:reasons.append('MOTION_MEMORY_CLEARANCE')
        if not stopping[action]['projection_clear']:reasons.append('MOTION_STOPPING_PROJECTION')
        rows.append(dict(action=action,eligible=not reasons,reasons=reasons,binding_rule=reasons[0] if reasons else None,
                         observation_allowed=action in obs_allowed,memory=memory[action],stopping=stopping[action]))
    outside=not rows[ACTIONS.index(selection['action'])]['eligible']
    if outside and selection['action']!='hold':
        gate=rows[ACTIONS.index(selection['action'])]
        gate.update(base_gate_exclusions=gate['reasons'],eligible=True,reasons=[],binding_rule=None,
                    admission_rule='EXPLICIT_DEPLOYED_SELECTION_OVERRIDE')
    return dict(candidates=rows,selected_action=selection['action'],
                selected_outside_candidate_mask=outside,
                overrides={k:selection[k] for k in ('clearance_turn','planned_stopping_projection','interrupted_route_turn_memory') if k in selection})


def command_motion(packet):
    state=packet['controller_state']
    tape=np.asarray(packet['candidate_requested_commands'])
    return command_predictions(state['command_model'],past_commands(packet['packet']['history'],packet['measured_ns']),tape)


def true_motion(traces,stamp):
    result=[]
    for trace in traces:
        ts=np.rint(trace['timestamp_s']*1e9).astype(np.int64)
        indices=[np.flatnonzero(ts==stamp+h*100_000_000) for h in range(9)]
        if any(len(i)!=1 for i in indices):
            result.append(np.full((8,3),np.nan));continue
        pose=trace['base_pose_world'][[i[0] for i in indices]];Q=rotation_xyzw(pose[0,3:])
        xy=(pose[1:,:3]-pose[0,:3])@Q
        yaw=[math.atan2((Q.T@rotation_xyzw(p[3:]))[1,0],(Q.T@rotation_xyzw(p[3:]))[0,0]) for p in pose[1:]]
        result.append(np.column_stack((xy[:,:2],yaw)))
    return np.asarray(result,np.float32)


def magnitude_only(predicted,true,epsilon=1e-6):
    norms=np.linalg.norm(predicted[:,:,:2],axis=-1);undefined=(norms<epsilon)|~np.isfinite(true).all(axis=-1)
    if undefined.any():return None,undefined.tolist()
    result=predicted.copy();result[:,:,:2]*=(np.linalg.norm(true[:,:,:2],axis=-1)/norms)[:,:,None]
    return result,undefined.tolist()


@torch.inference_mode()
def feature_motions(model, heads, packet, future_rgb=None):
    """Identical frozen preprocessing/predictor/readouts; dense tensors stay in RAM."""
    native=dense_native_context(packet['native_context'],observed_ns=packet['measured_ns'])
    device=next(model.predictor.parameters()).device
    context=F.layer_norm(model.encoder.tokens(native['pixels'].to(device)).float(),(1024,))[None]
    current=pool_tokens(context[:,-1]);control=native['past_applied_commands'][:,[0,2]].reshape(3,5,2).to(device)
    control=((control-model.control_mean)/model.control_std)[None]
    actions=torch.as_tensor(np.asarray(packet['candidate_applied_commands'])[:,:,[0,2]],dtype=torch.float32,device=device)
    mask=torch.ones(6,768,dtype=torch.bool,device=device)
    result={name:{'R4':[],'R4s':[],'R3':[]} for name in heads}
    for horizon in range(1,9):
        unique,inverse=torch.unique(actions[:,:horizon].reshape(6,-1),dim=0,return_inverse=True)
        indices=torch.stack([(inverse==j).nonzero()[0,0] for j in range(len(unique))]);n=len(unique)
        features=F.layer_norm(model.predictor(context.expand(n,-1,-1,-1),actions[indices],
            torch.full((n,),horizon,dtype=torch.long,device=device),mask[:n],control=control.expand(n,-1,-1,-1)).float(),(1024,))
        future=pool_tokens(features)
        for name,head in heads.items():
            motion=head(current.expand(n,-1,-1),future)[inverse].cpu().numpy()
            result[name]['R4'].append(motion);result[name]['R4s'].append(motion[PERMUTATION])
        if future_rgb is not None:
            from PIL import Image
            from scripts.dev_frozen_dense_representation_encoders_v1 import _normalise,_to_chw
            for index in range(6):
                pixels=future_rgb[index][horizon-1]
                if np.asarray(pixels).shape!=(480,640,3):raise ValueError('native future RGB required')
                tensor=_normalise(_to_chw(Image.fromarray(pixels).resize((512,384),Image.Resampling.BICUBIC)))[None].to(device)
                encoded=pool_tokens(F.layer_norm(model.encoder.tokens(tensor).float(),(1024,)))
                for name,head in heads.items():result[name]['R3'].append(head(current,encoded)[0].cpu().numpy())
    for name in heads:
        result[name]['R4']=np.stack(result[name]['R4'],axis=1)
        result[name]['R4s']=np.stack(result[name]['R4s'],axis=1)
        result[name]['R3']=np.stack(result[name]['R3']).reshape(8,6,3).transpose(1,0,2) if future_rgb is not None else None
    return result


def row_panel(packet,motions,true,*,state_id):
    """Phase 2 row adapter; callers performing checks must discard comparisons."""
    rows={};command=command_motion(packet)
    for key,motion in [('R2',true),('R5c',command)]:
        try:
            known=np.isfinite(motion).all(axis=(1,2)) if motion is not None else np.zeros(6,dtype=bool)
            selected=selector(packet,np.nan_to_num(motion) if motion is not None else np.zeros((6,8,3)))
            gates=eligibility(packet,selected)
            for i,valid in enumerate(known):
                if not valid:gates['candidates'][i]=dict(action=ACTIONS[i],eligible=None,reasons=['TRUE_MOTION_UNAVAILABLE'],binding_rule=None,observation_allowed=gates['candidates'][i]['observation_allowed'])
            rows[key]=dict(status='available' if known.all() else 'unresolved',selection=selected if known.all() else None,eligibility=gates)
            if not known.all():gates['selected_action']=None;gates['selected_outside_candidate_mask']=None;gates['overrides']={}
        except Exception as exc:rows[key]=dict(status='unresolved',reason=repr(exc))
    for head,values in motions.items():
        for row in ('R3','R4','R4s','R2b'):
            motion,undefined=((magnitude_only(values['R4'],true) if true is not None else (None,None)) if row=='R2b' else (values[row],None))
            name=f'{row}/{head}'
            if motion is None:
                rows[name]=dict(status='unresolved',reason='UNDEFINED_TRANSLATION_DIRECTION' if row=='R2b' else 'RGB_UNAVAILABLE',undefined=undefined);continue
            try:
                selected=selector(packet,motion);rows[name]=dict(status='available',selection=selected,eligibility=eligibility(packet,selected))
            except Exception as exc:rows[name]=dict(status='unresolved',reason=repr(exc))
    try:rows['R5r']=dict(status='available',selection=selector(packet,command,reactive=True))
    except Exception as exc:rows['R5r']=dict(status='unresolved',reason=repr(exc))
    # R0 is tied to the declared R5c eligibility; it never uses physical safety.
    eligible=[i for i,r in enumerate(rows.get('R5c',{}).get('eligibility',{}).get('candidates',[])) if r['eligible']]
    seed=int.from_bytes(hashlib.sha256(f'2026092308/{state_id}'.encode()).digest()[:8],'big')
    rows['R0']=dict(status='available',action_index=int(np.random.default_rng(seed).choice(eligible))) if eligible else dict(status='unresolved',reason='NO_R5C_ELIGIBLE_CANDIDATE')
    return rows


class ArticulatedSteps:
    """Existing FK and support functions only; no collision library."""
    def __init__(self,walls):
        self.model=ArticulatedCollisionGeometry(URDF);self.walls=walls
        self.centres=np.array([w['centre_xyz'] for w in walls]);self.halves=np.array([w['size_xyz'] for w in walls])/2
        self.axes=np.array([[[math.cos(w['yaw_rad']),math.sin(w['yaw_rad']),0],[-math.sin(w['yaw_rad']),math.cos(w['yaw_rad']),0],[0,0,1]] for w in walls])
        self.normals=self.axes.reshape(-1,3);self.wall_projection=np.einsum('wij,wj->wi',self.axes,self.centres)
        self.radii=np.array([np.linalg.norm(s['dimensions']/2) if s['kind']=='box' else s['dimensions'][0]
            if s['kind']=='sphere' else math.hypot(s['dimensions'][0],s['dimensions'][1]/2) for s in self.model._shapes])

    def evaluate(self,trace):
        lower=[];upper=[];centres=[];rotations=[]
        for pose,q in zip(trace['base_pose_world'],trace['joint_position'],strict=True):
            Q=rotation_xyzw(pose[3:]);transforms,_=self.model.transforms(q)
            support=self.model.supports(q,self.normals@Q);translation=self.normals@pose[:3]
            lo=(np.array([s['lower'] for s in support['shapes']])+translation).reshape(27,-1,3)
            hi=(np.array([s['upper'] for s in support['shapes']])+translation).reshape(27,-1,3)
            lower.append(np.maximum(lo-(self.wall_projection+self.halves),self.wall_projection-self.halves-hi).max(axis=2).min(axis=1))
            Ts=[transforms[s['link']]@s['origin'] for s in self.model._shapes]
            world_centres=np.array([pose[:3]+Q@T[:3,3] for T in Ts]);world_rotations=np.array([Q@T[:3,:3] for T in Ts])
            centres.append(world_centres);rotations.append(world_rotations)
            witness_bounds=[]
            for shape,centre,orientation in zip(self.model._shapes,world_centres,world_rotations,strict=True):
                delta=centre-self.centres;local=np.einsum('wij,wj->wi',self.axes,delta)
                nearest=np.clip(local,-self.halves,self.halves)
                direction=np.einsum('wji,wj->wi',self.axes,nearest-local)
                local_dir=direction@orientation;point=np.zeros_like(local_dir);dims=shape['dimensions']
                if shape['kind']=='box':point=np.sign(local_dir)*dims/2
                elif shape['kind']=='sphere':
                    norms=np.linalg.norm(local_dir,axis=1,keepdims=True);point=np.divide(local_dir*dims[0],norms,out=point,where=norms>0)
                else:
                    norms=np.linalg.norm(local_dir[:,:2],axis=1,keepdims=True)
                    point[:,:2]=np.divide(local_dir[:,:2]*dims[0],norms,out=point[:,:2],where=norms>0)
                    point[:,2]=np.sign(local_dir[:,2])*dims[1]/2
                witness=centre+point@orientation.T
                boxcoords=np.einsum('wij,wj->wi',self.axes,witness-self.centres)
                witness_bounds.append(float(np.linalg.norm(np.maximum(np.abs(boxcoords)-self.halves,0),axis=1).min()))
            upper.append(witness_bounds)
        lower=np.asarray(lower);upper=np.asarray(upper);centres=np.asarray(centres);rotations=np.asarray(rotations)
        relative=np.einsum('spji,spjk->spik',rotations[:-1],rotations[1:])
        angle=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))
        displacement=np.linalg.norm(np.diff(centres,axis=0),axis=-1)+angle*self.radii
        robust=np.minimum(lower[:-1],lower[1:])-displacement
        ts=np.rint(trace['timestamp_s']*1e9).astype(np.int64)
        complete=len(ts)==401 and np.all(np.diff(ts)==2_000_000) and ts[-1]-ts[0]==800_000_000
        contact=bool(np.asarray(trace['physics_contact']).any())
        def classify(threshold):
            if contact:return 'unsafe'
            if float(upper.min())<threshold:return 'unsafe'
            if not complete:return 'unresolved'
            # A lower bound below threshold does not prove the exact distance fails.
            if float(lower.min())<threshold:return 'unresolved_sampled_separation_bound'
            return 'safe' if float(robust.min())>=threshold else 'unresolved_interval_robustness'
        return dict(hard=classify(.005),operating=classify(.020),contact=contact,complete=complete,
            sampled_min_m=float(lower.min()),interval_min_m=float(robust.min()) if len(robust) else None,
            sampled_upper_min_m=float(upper.min()),per_step_primitive_separation_lower_m=lower.tolist(),per_step_primitive_separation_upper_m=upper.tolist(),per_interval_primitive_fk_displacement_m=displacement.tolist(),
            per_interval_primitive_robust_lower_m=robust.tolist(),primitive_max_radius_m=self.radii.tolist(),
            primitive_ids=[s['shape_id'] for s in self.model._shapes],
            native_contact_positive_maze_gap_indices=np.flatnonzero(np.asarray(trace['physics_contact'],bool)&(lower.min(axis=1)>1e-5)).tolist(),
            sampled_zero_maze_gap_without_native_contact_indices=np.flatnonzero((upper.min(axis=1)<1e-9)&~np.asarray(trace['physics_contact'],bool)).tolist(),
            contact_identity_evidence='Sibling contact_events.json; distinguish permitted support from maze obstacles and disallowed body/ground contact',
            native_discrete_ground_truth=True,continuous_between_samples_certified=False,
            robustness_convention='endpoint FK centre translation + SO(3) angle times primitive circumradius; not an unobserved trajectory speed bound')


def filter_observation(eligibility_record,safety,criterion):
    rows=[]
    for gate,physical in zip(eligibility_record['candidates'][1:],safety[1:],strict=True):
        status=physical[criterion];resolved=status in ('safe','unsafe') and gate['eligible'] is not None
        rows.append(dict(action=gate['action'],eligible=gate['eligible'],safety=status,resolved=resolved,
            excluded_but_safe=resolved and not gate['eligible'] and status=='safe',
            admitted_but_unsafe=resolved and gate['eligible'] and status=='unsafe',
            binding_rule=gate['binding_rule'],all_exclusion_reasons=gate['reasons']))
    all_excluded=all(r['eligible'] is False for r in rows);any_safe=any(r['safety']=='safe' for r in rows)
    event=True if all_excluded and any_safe else False if any(r['eligible'] is True for r in rows) or all(r['resolved'] for r in rows) else None
    return dict(candidates=rows,all_movement_excluded_despite_safe=event,
        safe_count=sum(r['resolved'] and r['safety']=='safe' for r in rows),unsafe_count=sum(r['resolved'] and r['safety']=='unsafe' for r in rows),
        unresolved_count=sum(not r['resolved'] for r in rows),
        binding_rules=[r['binding_rule'] for r in rows if r['excluded_but_safe']])


def scalar_reference(geometry,trace,*,arrival_settling,parameters):
    """Original finite cost components; articulated safety is a separate mask."""
    pose=trace['base_pose_world'][-1];twist=trace['base_twist_world'][-1]
    endpoint=geometry.distance_and_heading(pose[:2])
    if not endpoint['valid']:return dict(status='unresolved',reason=endpoint['reason'],cost_s=None)
    Q=rotation_xyzw(pose[3:]);yaw=math.atan2(Q[1,0],Q[0,0]);heading=endpoint['heading_rad']
    settling=bool(arrival_settling and endpoint['distance_m']<=parameters['arrival_radius_m'])
    error=0 if heading is None or settling else wrap(heading-yaw)
    direction=np.array([math.cos(heading),math.sin(heading)]) if heading is not None else np.zeros(2)
    components=dict(remaining_travel_s=endpoint['distance_m']/parameters['nominal_travel_speed_m_s'],
        heading_alignment_s=abs(error)/parameters['nominal_turn_speed_rad_s'],
        reverse_motion_braking_s=0 if settling else max(0.,-float(twist[:2]@direction))/parameters['linear_braking_acceleration_m_s2'],
        opposing_turn_braking_s=0 if settling else max(0.,-float(twist[5])*np.sign(error))/parameters['angular_braking_acceleration_rad_s2'],
        arrival_linear_settling_s=max(0.,float(np.linalg.norm(twist[:2]))-parameters['quiet_linear_speed_m_s'])/parameters['linear_braking_acceleration_m_s2'] if settling else 0.,
        arrival_angular_settling_s=max(0.,abs(float(twist[5]))-parameters['quiet_angular_speed_rad_s'])/parameters['angular_braking_acceleration_rad_s2'] if settling else 0.)
    return dict(status='available',cost_s=float(sum(components.values())),components=components,endpoint=endpoint)


def reference_panel(costs,safety):
    """Unknown bank members make only this state's optimum unresolved."""
    if any(r['hard'].startswith('unresolved') or r['operating'].startswith('unresolved') for r in safety):
        return dict(status='unresolved',reason='BANK_SAFETY_UNRESOLVED')
    physical=[i for i,r in enumerate(safety) if r['hard']=='safe']
    margin=[i for i in physical if safety[i]['operating']=='safe']
    if any(costs[i]['cost_s'] is None for i in physical):return dict(status='unresolved',reason='BANK_COST_UNRESOLVED')
    if not physical or not margin:return dict(status='unresolved',reason='NO_PHYSICAL_OR_MARGIN_ALTERNATIVE')
    best=min(margin,key=lambda i:(costs[i]['cost_s'],i));base=min(costs[i]['cost_s'] for i in physical)
    return dict(status='available',action_index=best,cost_s=costs[best]['cost_s'],physical=physical,margin=margin,
                numerical_optimal_indices=[i for i in margin if costs[i]['cost_s']<=costs[best]['cost_s']+1e-8],
                practical_near_optimal_indices=[i for i in margin if costs[i]['cost_s']<=costs[best]['cost_s']+.10],
                operating_margin_cost_s=costs[best]['cost_s']-base)
