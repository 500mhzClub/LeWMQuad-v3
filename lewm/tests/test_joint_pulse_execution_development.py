"""Joint sensor admission and actual closed-loop synthetic plant transactions.

Synthetic pose/plant fixtures exercise control; rendered RGB-D tests separately
exercise real observer integration. Neither is native quadruped validation.
"""
import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.joint_sensor_anchored_goal_development import JointAnchoredGoal, current_joint_pose
from lewm.joint_continuous_pulse_execution_development import JointInnerGoalPulseExecution
from lewm.joint_inner_goal_room_return_development import JointInnerGoalRoomReturn, RawJointInnerGoalExecution
from lewm.sensor_anchored_goal_development import current_pose
from lewm.joint_rgbd_rigid_pose_development import angle
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_coupled_pulse_rollout_development import table
from lewm.coupled_pulse_rollout_development import compose
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
from lewm.tests.test_measured_pose_seeded_rgbd_development import sequence
from scripts.fixed_nominal_pulse_table_development import load_fixed_table


def joint_visual(frame, state=(0.,0.,0.), previous=None):
    e,now=visual(frame,*state);p=e['current_pose']
    p.update(mode='joint',gyro_role='consistency_monitor_only',reference_frame=max(0,frame-1),
        promoted_keyframe=False,native_pose_input=False,global_history_reset=False,
        position_error_bound=None,orientation_error_bound=None,uncertainty_model_validated=False)
    e.update(continuity_evidence_current=True,command_integration_used=False,
        bridge_is_command_or_inertial_extrapolation=False,anchor_promotion_from_bridge=False)
    c=dict(status='INITIAL_REFERENCE',bridge_frames=0,uncertainty_calibrated=False)
    if frame:
        old=previous['current_pose'];ref=np.asarray(old['rotation_initial_body_from_current_body'])
        R=np.asarray(p['rotation_initial_body_from_current_body']);local=ref.T@R
        w=dict(reference_frame=frame-1,current_frame=frame,reference_measured_ns=now-100_000_000,
            fitting_mode='joint',candidate_envelope_passed=True,witness_alone_grants_pose=False,
            reference_rotation_initial_body_from_reference_body=ref.tolist(),
            fitted_rotation_reference_body_from_current_body=local.tolist(),
            composed_rotation_initial_body_from_current_body=R.tolist(),
            gyro_rotation_reference_body_from_current_body=local.tolist(),
            gyro_disagreement_rad=angle(local.T@local),position_initial_body_m=p['position_initial_body_m'])
        c.update(status='ANCHOR_MEASUREMENT',previous_frame=frame-1,previous_measured_ns=now-100_000_000,
            rotation_fitting_mode='joint',gyro_role='consistency_monitor_only',gyro_bias_estimated=False,
            rotation_measurement_witnesses=[w],anchor_available=True,incremental_available=True,
            selected_anchor_rotation_witness=w,incremental_rotation_witness=w,
            incremental_rotation_witness_saved=True,disagreement_m=0.,disagreement_rad=angle(R.T@R))
    e['continuity_evidence']=c
    return e,now


def method(path, cls, name):
    tree=ast.parse(Path(path).read_text())
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return deepcopy(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name))


def test_control_algorithm_and_mission_transitions_are_narrow_transcriptions():
    pairs=[('inner_goal_pulse_feedback','InnerGoalPulseServo','joint_inner_goal_pulse_feedback','JointInnerGoalPulseServo','step'),
        ('continuous_pulse_execution','ContinuousPulseExecution','joint_continuous_pulse_execution','JointInnerGoalPulseExecution','observe'),
        ('continuous_pulse_execution','ContinuousPulseExecution','joint_continuous_pulse_execution','JointInnerGoalPulseExecution','begin'),
        ('sensor_anchored_goal','AnchoredGoal','joint_sensor_anchored_goal','JointAnchoredGoal','from_observation')]
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,n):
            n.id={'current_joint_pose':'current_pose','JointAnchoredGoal':'AnchoredGoal',
                'JointInnerGoalPulseServo':'AnchoredPulseServo'}.get(n.id,n.id);return n
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='JointInnerGoalPulseServo':
                assert ast.unparse(n.args[1])=='self.table';n.args=n.args[:1]
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if n.value=='joint-measured incremental yaw bound exceeded':n.value='gyro-supported incremental yaw bound exceeded'
            return n
    for a,ac,b,bc,name in pairs:
        old=method('lewm/'+a+'_development.py',ac,name)
        new=Normalize().visit(method('lewm/'+b+'_development.py',bc,name))
        assert ast.dump(old)==ast.dump(new),(name,a)


def test_joint_goal_is_explicit_and_input_is_not_relabelled_or_aliased():
    e,t=joint_visual(0);before=deepcopy(e)
    g=JointAnchoredGoal.from_observation(e,[.4,0],.3,identity=(0,0,0),now_ns=t)
    assert e==before and g.snapshot()['pose_mode']=='joint'
    with pytest.raises(SensorContractError):current_pose(e,identity=(0,0,0),now_ns=t)
    old,t=visual(0)
    with pytest.raises(SensorContractError):current_joint_pose(old,identity=(0,0,0),now_ns=t)
    e['current_pose']['position_initial_body_m'][0]=10
    assert g.anchor_position==(0.,0.,0.)


@pytest.mark.parametrize('fault',['missing','stale','future','episode','reflection','label','witness',
    'composition','selected_position','gyro_gate','gap','reset','promotion','bridge_budget'])
def test_bad_current_joint_evidence_latches_zero_and_forbids_next_leg(fault):
    m=JointInnerGoalPulseExecution(table());initial,t=joint_visual(0)
    assert m.observe(initial,now_ns=t)['status']=='IDLE';m.begin([.4,0],0,now_ns=t)
    e,t=joint_visual(1,previous=initial);p=e['current_pose'];c=e['continuity_evidence']
    if fault=='missing':e['current_pose']=None
    if fault=='stale':p['measured_ns']-=1
    if fault=='future':p['available_ns']+=1
    if fault=='episode':e['identity']=(1,0,0)
    if fault=='reflection':p['rotation_initial_body_from_current_body'][0][0]=-1
    if fault=='label':p['gyro_role']='rotation_estimator'
    if fault=='witness':c['rotation_measurement_witnesses']=[]
    if fault=='composition':c['rotation_measurement_witnesses'][0]['composed_rotation_initial_body_from_current_body']=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    if fault=='selected_position':p['position_initial_body_m']=[.1,0.,0.]
    if fault=='gyro_gate':c['rotation_measurement_witnesses'][0]['gyro_disagreement_rad']=.11
    if fault=='gap':p['frame']=2
    if fault=='reset':p['frame']=0
    if fault in ('promotion','bridge_budget'):
        c.update(status='MEASURED_INCREMENT_BRIDGE',anchor_available=False,selected_anchor_rotation_witness=None,
            bridge_frames=11 if fault=='bridge_budget' else 1,disagreement_m=None,disagreement_rad=None)
        p['promoted_keyframe']=fault=='promotion'
    r=m.observe(e,now_ns=t)
    assert r['status']=='FAILED' and r['requested_command']==[0.,0.,0.]
    e,t=joint_visual(2,previous=initial)
    assert m.observe(e,now_ns=t)['status']=='FAILED'
    with pytest.raises(SensorContractError):m.begin([0,0],0,now_ns=t)


@pytest.mark.parametrize('sign',[-1,1])
def test_full_synthetic_feedback_return_preserves_goal_holds_winding_and_memory(sign):
    tab=load_fixed_table();m=JointInnerGoalRoomReturn(sign,tab)
    state=np.zeros(3);increment=np.zeros(3);previous=None
    for tick in range(3601):
        e,t=joint_visual(tick,state,previous);previous=deepcopy(e)
        ex=m.runtime.executor.observe(e,now_ns=t)
        r=m.advance(dict(evidence=e,execution=ex,requested_command=ex['requested_command']),now_ns=t)
        if r['terminal']:break
        local=ex['local_decision']
        if local and 'new_pulse' in local['diagnostic']:
            effect=tab.effects[local['diagnostic']['new_pulse']['action_index']]
            increment=(compose(state,effect.delta_xy_yaw)-state)/effect.ticks
        if any(r['requested_command']):state+=increment
    assert r['terminal']=='ROOM_RETURN_CANDIDATE',r
    assert len(m.completed)==7 and np.linalg.norm(state[:2])<=.04
    for leg in m.runtime.executor.legs:
        d=leg['final_decision'];assert d['diagnostic']['position_error_m']<=.04
        assert abs(d['diagnostic']['yaw_error_rad'])<=.05 and d['quiet_intervals']>=10
        assert leg['goal']['pose_mode']=='joint' and d['controller']=='joint_inner_goal_empirical_pulse_feedback_v1'
    assert m.runtime.executor.start_ns==1_600_000_000 and not m.snapshot()['home_verified']


def test_real_rendered_rgbd_enters_execution_without_mode_conversion():
    runtime=RawJointInnerGoalExecution(table())
    for frame,item in enumerate(sequence(2.,1,rate=(.01,-.02,.03),steps=5)):
        p,d,f,now=item[:4];r=runtime.observe(p,d,f,now_ns=now)
        assert r['execution']['status']!='FAILED',r
        assert r['evidence']['current_pose']['mode']=='joint'
        np.testing.assert_allclose(r['evidence']['current_pose']['position_initial_body_m'],item[4],atol=.006,rtol=0)
        if frame==0:runtime.executor.begin([.4,0],0,now_ns=now)
    assert runtime.executor.last_frame==4


def test_real_missing_rgb_latches_runtime_and_mission():
    runtime=RawJointInnerGoalExecution(table());items=list(packets([texture()]*3))
    p,d,f,now=items[0];runtime.observe(p,d,f,now_ns=now);runtime.executor.begin([.4,0],0,now_ns=now)
    p,d,f,now=deepcopy(items[1]);p['image']['rgb'][:]=128
    r=runtime.observe(p,d,f,now_ns=now)
    assert r['execution']['status']=='FAILED' and not any(r['requested_command'])
    p,d,f,now=items[2];r=runtime.observe(p,d,f,now_ns=now)
    assert r['execution']['status']=='FAILED' and r['evidence']['terminal_failure'] is not None
