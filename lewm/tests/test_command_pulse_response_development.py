from copy import deepcopy
import numpy as np
import pytest

from lewm.command_pulse_response_development import (COMMANDS,TRIALS,MAXIMUM_TICKS,
    events,schedule,validate_command,specification,pack,PulseSequence)
from lewm.tests.test_bounded_visual_servo_development import evidence
from scripts.audit_go2_command_pulse_response_v1 import relative_change,event_reports


def test_design_counterbalances_exact_events_and_keeps_braking_tail():
    a=events('a'); b=events('b')
    assert len(a)==len(b)==16
    assert [{k:v for k,v in e.items() if k!='event_index'} for e in a]==list(reversed([{k:v for k,v in e.items() if k!='event_index'} for e in b]))
    for order in ('a','b'):
        tape=schedule(order); assert len(tape)==MAXIMUM_TICKS==386
        assert all(r['requested_command']==[0.,0.,0.] for r in tape[:10])
        for event in events(order):
            rows=[r for r in tape if r['event_index']==event['event_index']]
            assert len(rows)==event['pulse_ticks']+20
            assert all(r['requested_command']==event['requested_command'] for r in rows[:event['pulse_ticks']])
            assert all(r['requested_command']==[0.,0.,0.] for r in rows[event['pulse_ticks']:])


@pytest.mark.parametrize('command',[[.21,0,0],[0,0,.46],[0,.01,0],[np.nan,0,0],[0,0],[-.2,0,0]])
def test_command_cap_and_schema_fail_closed(command):
    with pytest.raises(ValueError):validate_command(command)


def test_supported_bank_and_small_commands_are_in_declared_new_domain():
    for _,c in COMMANDS: assert validate_command(c)==list(c)
    assert validate_command([.2,0,.45])==[.2,0,.45]


def test_sequence_is_fixed_and_completes_only_after_all_commands():
    driver=PulseSequence('b'); planned=schedule('b')
    for i in range(MAXIMUM_TICKS+1):
        t=1_500_000_000+i*100_000_000
        r=driver.step(evidence(t,x=.01),now_ns=t)
        if i<MAXIMUM_TICKS:
            assert r['terminal'] is None and r['requested_command']==planned[i]['requested_command']
        else: assert r['terminal']=='PULSE_SCHEDULE_COMPLETE'
    assert not r['navigation_qualified'] and not r['native_pose_used']


@pytest.mark.parametrize('fault',['missing','stale','future','episode','mode','nan','reflection','excursion','clock'])
def test_observation_failure_stops_schedule_and_latches(fault):
    driver=PulseSequence('a'); t=1_500_000_000
    driver.step(evidence(t),now_ns=t);t+=100_000_000;e=evidence(t)
    if fault=='missing':e['current_pose']=None
    elif fault=='stale':e['current_pose']['measured_ns']-=1
    elif fault=='future':e['current_pose']['available_ns']+=1
    elif fault=='episode':e['identity']=(1,0,0)
    elif fault=='mode':e['current_pose']['mode']='joint'
    elif fault=='nan':e['current_pose']['position_initial_body_m'][0]=np.nan
    elif fault=='reflection':e['current_pose']['rotation_initial_body_from_current_body']=(-np.eye(3)).tolist()
    elif fault=='excursion':e['current_pose']['position_initial_body_m'][0]=1.01
    else:t+=100_000_000;e=evidence(t)
    r=driver.step(e,now_ns=t)
    assert r['terminal']=='PULSE_SCHEDULE_FAILED' and r['requested_command']==[0.,0.,0.]
    assert driver.step(evidence(t+100_000_000),now_ns=t+100_000_000)['terminal']=='PULSE_SCHEDULE_FAILED'


def test_fresh_actual_starts_pair_friction_and_exact_spec():
    for trial in TRIALS:
        spec=specification(trial); p=pack(spec); x,y,yaw=spec['geometry']['spawn_se2_world']
        assert p.robot.spawn_xyz_m==(x,y,.375)
        assert p.robot.spawn_quat_wxyz[3]==pytest.approx(np.sin(yaw/2))
        assert p.physics_randomization.floor_friction_mu==(1. if trial.startswith('nominal') else .15)
        bad=deepcopy(spec);bad['geometry']['spawn_se2_world'][0]=0.
        with pytest.raises(ValueError):pack(bad)
    assert pack(specification('nominal_a')).robot==pack(specification('lower_friction_a')).robot
    assert pack(specification('nominal_a')).robot!=pack(specification('nominal_b')).robot


def test_relative_response_uses_initial_body_frame_not_world_axes():
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    r=relative_change((np.array([1.,2.,0.]),R),(np.array([1.,2.1,0.]),R))
    np.testing.assert_allclose(r['displacement_body_m'],[.1,0,0],atol=1e-14)
    assert r['yaw_change_rad']==0.


def test_event_labels_align_actual_command_end_and_twosecond_tail():
    n=20050; t=np.arange(1,n+1)*.002; p=np.zeros((n,7));p[:,0]=(t-1.5)*.01
    raw=dict(timestamp_s=t,base_pose_world=p,base_twist_world=np.zeros((n,6)))
    rotations=np.tile(np.eye(3),(n,1,1)); decisions=[]
    planned=schedule('a')
    for tick in range(387):
        now=1_500_000_000+tick*100_000_000
        decision=(planned[tick]|dict(terminal=None)) if tick<386 else dict(terminal='PULSE_SCHEDULE_COMPLETE',stage='terminal',event_index=None)
        decisions.append(dict(tick=tick,evidence=evidence(now,x=.001*tick),decision=decision))
    reports=event_reports('a',decisions,raw,rotations); e=reports[0]
    assert e['start_tick']==10 and e['endpoints']['pulse_end']['tick']==12
    assert e['endpoints']['brake_20']['tick']==32
    assert e['endpoints']['pulse_end']['visual']['displacement_body_m'][0]==pytest.approx(.002)
    assert e['endpoints']['brake_20']['native_braking_change']['displacement_body_m'][0]==pytest.approx(.020)
    assert all(e['endpoints']['brake_20']['visual'] is not None for e in reports)
    truncated={k:v[:1349] for k,v in raw.items()}
    incomplete=event_reports('a',decisions[:12],truncated,rotations[:1349])
    assert incomplete[0]['endpoints']['pulse_end']['native'] is None
    assert incomplete[0]['endpoints']['pulse_end']['visual'] is None
    assert len(incomplete)==16  # missing events retained, not success-only filtering
    # A failed planned pulse followed by a zero-command drain is not that pulse.
    stopped=deepcopy(decisions)
    stopped[11]['decision']['terminal']='PULSE_SCHEDULE_FAILED'
    missing=event_reports('a',stopped[:12],raw,rotations)
    assert not missing[0]['pulse_complete'] and missing[0]['issued_pulse_ticks']==1
    assert missing[0]['endpoints']['brake_20']['native'] is None
