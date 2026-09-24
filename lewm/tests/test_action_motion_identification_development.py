from dataclasses import replace

import numpy as np
import pytest

from lewm.action_motion_identification_development import (
    MotionState, MotionIdentificationController, motion_priors, command_schedule)
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_continuous_startup_handoff_development import frames
from lewm.tests.test_startup_observation_turn_development import components
from scripts.audit_go2_action_motion_identification_development_v1 import (
    primitive_poses, primitive_displacement_bounds, prediction_errors, audit_command_tape)
from lewm.relative_gyro_turn_development import rotation_increment


def controller():
    geometry,kwargs=components()
    kwargs['region_prior']=replace(kwargs['region_prior'],lower_initial_body_m=(-1.25,)*3,
        upper_initial_body_m=(1.25,)*3,valid_until_ns=8_000_000_000)
    owner=MotionState(geometry,**kwargs)
    return MotionIdentificationController(owner)


def test_new_condition_and_distinct_fixed_schedules_do_not_modify_old_defaults():
    velocity,region=motion_priors(1_500_000_000,'a'*64)
    assert region.valid_until_ns==8_000_000_000 and region.lower_initial_body_m==(-1.25,)*3
    assert velocity.anchor_ns==1_500_000_000
    a,b=command_schedule('identification'),command_schedule('validation')
    assert len(a)==len(b)==28 and a!=b
    assert a[:6]==[[.12,0.,0.]]*6 and a[10:14]==[[0.,0.,.35]]*4
    assert a[-4:]==[[0.,0.,0.]]*4 and b[10:14]==[[0.,0.,-.35]]*4
    with pytest.raises(SensorContractError): motion_priors(1_600_000_000,'a'*64)


def test_full_synthetic_schedule_retains_same_owner_and_every_prediction():
    model=controller(); owner=model.owner; memory=owner._memory; rows=[]
    for p,d,f,t in frames(45):
        row=model.observe(p,d,f,now_ns=t); rows.append(row)
        if row['terminal']: break
    assert model.status=='COMPLETE_IDENTIFICATION_SCHEDULE'
    motion=[r for r in rows if r['prediction'] is not None]
    assert len(motion)==28 and [r['requested_command'] for r in motion]==command_schedule('identification')
    assert owner._memory is memory and owner._startup.last_ns==2_000_000_000
    assert all(r['envelope']['setup_contains'] and not r['prediction']['execution_error_validated'] for r in motion)
    assert owner.factored_guard is not None
    with pytest.raises(SensorContractError): model.observe(p,d,f,now_ns=t)


@pytest.mark.parametrize('fault',['wall','penetration','incompatible'])
def test_factored_continuation_rejects_conflicts_after_ready(monkeypatch,fault):
    import lewm.action_motion_identification_development as module
    model=controller(); packets=list(frames(9))
    for p,d,f,t in packets[:8]: assert not model.observe(p,d,f,now_ns=t)['terminal']
    original=module.query_factored_configuration
    def bad(*args,**kwargs):
        result=original(*args,**kwargs); row=result['primitives'][0]
        if fault=='wall': row['nonfloor_conflict_sources']=[kwargs['now_ns']]
        if fault=='penetration': row['ground']['observed_penetration_sources']=[kwargs['now_ns']]
        if fault=='incompatible': row['ground']['incompatible_covered_plane_pairs']=[[1,2]]
        return result
    monkeypatch.setattr(module,'query_factored_configuration',bad)
    p,d,f,t=packets[8]; row=model.observe(p,d,f,now_ns=t)
    assert row['terminal'] and row['requested_command']==[0.,0.,0.]
    assert model.owner.status.startswith('FAILED_')


def test_stale_motion_owner_fails_without_reinitialization():
    model=controller(); packets=list(frames(9))
    for p,d,f,t in packets[:8]: model.observe(p,d,f,now_ns=t)
    p,d,f,t=packets[8]; row=model.observe(p,d,f,now_ns=t+1)
    assert row['terminal'] and model.owner._count==8


def test_primitive_displacement_bound_encloses_material_points():
    geometry,_=components(); q=np.repeat([0.,.8,-1.5],4)
    before=primitive_poses(geometry,np.zeros(3),np.eye(3),q)
    after=primitive_poses(geometry,np.array([.02,-.01,.03]),rotation_increment([.03,-.02,.1]),q+.03)
    bounds=primitive_displacement_bounds(before,after); rng=np.random.default_rng(2026090603)
    for a,b,bound in zip(before,after,bounds,strict=True):
        points=rng.normal(size=(100,3)); points*=a[3]/np.linalg.norm(points,axis=1)[:,None]
        actual=np.linalg.norm((points@b[2].T+b[1])-(points@a[2].T+a[1]),axis=1)
        assert np.max(actual)<=bound+1e-12
    assert primitive_displacement_bounds(before,before)==[0.]*27


def prediction_fixture(n=201):
    geometry,_=components(); q=np.repeat([0.,.8,-1.5],4)
    raw=dict(timestamp_s=np.arange(n)*.002,base_pose_world=np.tile([0.,0.,.3,0.,0.,0.,1.],(n,1)),
        joint_position=np.tile(q,(n,1)),applied_command=np.zeros((n,3)))
    raw['base_pose_world'][:,0]=.1*raw['timestamp_s']
    forecast=dict(expected_applied_commands=[[0.,0.,0.]]*4,positions_current_body_m=[[0.,0.,0.]]*5,
        rotations_current_body=[np.eye(3).tolist()]*5,joints_rad=[q.tolist()]*5)
    decisions=[dict(observation_index=0,decision=dict(prediction=forecast,motion_index=0))]
    return geometry,raw,decisions,[dict(physical_sample_index=0)]


def test_prediction_auditor_scores_actual_recorded_endpoints_and_material_bounds():
    geometry,raw,decisions,cameras=prediction_fixture()
    rows=prediction_errors(raw,geometry,decisions,cameras)
    assert len(rows)==4 and all(r['status']=='SCORED_EXECUTED_FUTURE' for r in rows)
    np.testing.assert_allclose([r['body_translation_error_m'] for r in rows],[.01,.02,.03,.04],atol=1e-12)
    np.testing.assert_allclose([r['maximum_primitive_point_error_upper_bound_m'] for r in rows],[.01,.02,.03,.04],atol=1e-12)


def test_truncated_and_unexecuted_forecasts_are_not_model_error_samples():
    geometry,raw,decisions,cameras=prediction_fixture(100)
    rows=prediction_errors(raw,geometry,decisions,cameras)
    assert [r['status'] for r in rows]==['SCORED_EXECUTED_FUTURE']+['TRUNCATED_HORIZON']*3
    raw['applied_command'][1:51,0]=.1
    rows=prediction_errors(raw,geometry,decisions,cameras)
    assert rows[0]['status']=='PREDICTED_COMMAND_SEQUENCE_NOT_EXECUTED'
    assert not any('body_translation_error_m' in r for r in rows)


def test_raw_tape_reconstruction_detects_unaccounted_or_mismatched_motion():
    raw=dict(timestamp_s=np.arange(800)*.002,requested_command=np.zeros((800,3)),
        applied_command=np.zeros((800,3)),post_slew_applied_command=np.zeros((800,3)),
        phase=np.r_[np.zeros(750),np.full(50,3)])
    decision=dict(terminal=False,requested_command=[0.,0.,0.],phase=3)
    tape=[dict(tick=0,pre_sample_index=749,post_sample_index=799,phase=3,decision_index=0,
        requested_command=[0.,0.,0.],execution_wall_ms=1.)]
    result=dict(requested_ticks=1,stopping_tail_ticks=0,physical_stop_reason=None)
    audit_command_tape(raw,tape,[dict(decision=decision)],result)
    raw['applied_command'][-1,0]=.1
    with pytest.raises((ValueError,AssertionError)):
        audit_command_tape(raw,tape,[dict(decision=decision)],result)
