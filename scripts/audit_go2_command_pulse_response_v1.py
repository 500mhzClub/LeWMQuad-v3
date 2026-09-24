"""Raw acquisition and sensor-command replay, followed by native target scoring."""
import json
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.command_pulse_response_development import PulseSequence,schedule,events,MAXIMUM_TICKS
from lewm.bounded_visual_servo_development import wrapped
from lewm.command_pulse_response_development import specification
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.visual_led_motion_development import VisualLedMotion
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_raw_audit_development import audit_sensors,contact_packet
from scripts.run_go2_command_pulse_response_v1 import OUTPUT,TRIALS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources


def relative_change(first,second):
    """Actual displacement and orientation change in the first body frame."""
    p,R=first; q,S=second
    delta=np.asarray(R).T@(np.asarray(q)-p); turn=np.asarray(R).T@S
    return dict(displacement_body_m=delta.tolist(),yaw_change_rad=float(np.arctan2(turn[1,0],turn[0,0])))


def event_reports(order,decisions,raw,rotations):
    planned=schedule(order); pose_by_tick={}; decision_by_tick={r['tick']:r['decision'] for r in decisions}
    for row in decisions:
        p=row['evidence']['current_pose']
        if p is not None:
            pose_by_tick[row['tick']]=(np.asarray(p['position_initial_body_m']),np.asarray(p['rotation_initial_body_from_current_body']))
    reports=[]; n=len(raw['timestamp_s'])
    def native_pose(tick):
        i=749+tick*50
        return (raw['base_pose_world'][i,:3],rotations[i]) if i<n else None
    for event in events(order):
        begin=next(i for i,r in enumerate(planned) if r['event_index']==event['event_index'] and r['event_offset']==0)
        pulse_end=begin+event['pulse_ticks']
        issued=sum(t in decision_by_tick and decision_by_tick[t]['terminal'] is None
                   and decision_by_tick[t]['event_index']==event['event_index']
                   and decision_by_tick[t]['stage']=='pulse' for t in range(begin,pulse_end))
        complete_pulse=issued==event['pulse_ticks'] and native_pose(pulse_end) is not None
        native0=native_pose(begin) if complete_pulse else None
        visual0=pose_by_tick.get(begin) if complete_pulse else None
        endpoints={}
        for name,tick in [('pulse_end',pulse_end)]+[('brake_'+str(h),pulse_end+h) for h in (1,5,10,20)]:
            native=native_pose(tick); visual=pose_by_tick.get(tick); stop=native_pose(pulse_end); vstop=pose_by_tick.get(pulse_end)
            row=dict(tick=tick,native=None,visual=None,native_braking_change=None,visual_braking_change=None)
            if native0 is not None and native is not None:
                row['native']=relative_change(native0,native)
                i=749+tick*50; a=i-49
                row['native_last100ms_maximum_speed_m_s']=float(np.linalg.norm(raw['base_twist_world'][a:i+1,:3],axis=1).max())
                S=rotations[a-1:i+1]; rel=np.einsum('ij,njk->nik',rotations[a-1].T,S)
                ys=np.unwrap(np.arctan2(rel[:,1,0],rel[:,0,0]))
                row['native_last100ms_maximum_yaw_rate_rad_s']=float(np.abs(np.diff(ys)/.002).max())
                row['native_last100ms_quiet']=row['native_last100ms_maximum_speed_m_s']<=.02 and row['native_last100ms_maximum_yaw_rate_rad_s']<=.05
                if stop is not None: row['native_braking_change']=relative_change(stop,native)
            if visual0 is not None and visual is not None:
                row['visual']=relative_change(visual0,visual)
                if vstop is not None: row['visual_braking_change']=relative_change(vstop,visual)
            endpoints[name]=row
        reports.append(event|dict(start_tick=begin,issued_pulse_ticks=issued,pulse_complete=complete_pulse,endpoints=endpoints))
    return reports


def audit_condition(condition,result):
    directory=OUTPUT/condition; spec=specification(condition)
    if result['physics_samples']<750:
        raw=read_npz(directory,'physics_trace.npz'); n=len(raw['timestamp_s'])
        assert result['physical_stop'] is not None and n==result['physics_samples']
        np.testing.assert_allclose(raw['timestamp_s'],np.arange(1,n+1)*.002,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(raw['requested_command'],np.zeros((n,3)))
        return dict(raw_sensor_audit_pass=False,status='PARTIAL_SETUP_ONLY_NOT_FULL_AUDIT',
                    native_stop=result['physical_stop'],physics_samples=n,schedule_complete=False)
    raw,contacts,topology,roles,cameras,relatives,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s']); assert n<=20550
    # Changed actual physics prefix, not just a different random seed label.
    prior=ROOT/'.generated/go2_longer_observed_floor_motion_development_v1_attempt_001/fit'
    old=read_npz(prior,'physics_trace.npz')
    prefix_equal=bool(np.array_equal(raw['base_pose_world'][:750],old['base_pose_world'][:750]))
    assert not prefix_equal
    previous_servo=read_npz(ROOT/'.generated/go2_goal_hold_visual_servo_v1_attempt_001'/spec['condition'],'physics_trace.npz')
    old_servo_prefix_equal=bool(np.array_equal(raw['base_pose_world'][:750],previous_servo['base_pose_world'][:750]))
    assert not old_servo_prefix_equal
    np.testing.assert_allclose(raw['base_pose_world'][0,:2],spec['geometry']['spawn_se2_world'][:2],rtol=0,atol=.002)
    initial_yaw=float(Rotation.from_quat(raw['base_pose_world'][0,3:]).as_euler('xyz')[2])
    assert abs(wrapped(initial_yaw-spec['geometry']['spawn_se2_world'][2]))<.002
    friction=read_json(directory,'friction_checks.json')
    for row in friction:
        np.testing.assert_allclose(row['solver_friction'],spec['friction_mu'],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(row['solver_ratio'],np.ones((1,28)))
    assert friction[0]['physics_steps']==0 and friction[-1]['physics_steps']==n
    assert read_json(directory,'actuator_identity.json')['effective']==read_json(directory,'terminal_actuator_gains.json')
    decisions=read_json(directory,'pulse_decisions.json'); tape=read_json(directory,'command_tape.json')
    assert len(decisions)==result['decisions'] and len(tape)==result['command_ticks']
    motion=VisualLedMotion('gyro',identity=(0,0,0)); controller=PulseSequence(spec['order']); desired=[]
    for tick,row in enumerate(decisions):
        frame=row['observation_index']; assert row['tick']==tick and frame==tick
        p,d=load_rgbd_observation(directory,frame); fast=load_fast_packet(directory,frame); now=p['sensor_state']['decision_ns']
        observed=motion.observe(p,d,fast,now_ns=now); command=controller.step(observed,now_ns=now)
        assert json.loads(json.dumps(observed))==row['evidence'] and command==row['decision']
        assert row['pre_sample_index']==cameras[frame]['physical_sample_index']==749+tick*50
        if command['terminal'] is None: desired.append((command['requested_command'],command['phase'],'fixed_pulse_schedule'))
        elif command['terminal']=='PULSE_SCHEDULE_FAILED': desired.extend([([0.,0.,0.],9,'terminal_zero_command_drain')]*10)
    if result['physical_stop'] is None: assert len(desired)==len(tape)
    else: assert len(tape)<=len(desired)
    for i,(item,(requested,phase,role)) in enumerate(zip(tape,desired)):
        assert item['tick']==i and item['requested_command']==requested and item['phase']==phase and item['role']==role
        a,b=item['pre_sample_index'],item['post_sample_index']; assert a==749+i*50 and b<=a+50
        if item['completed']: assert b==a+50
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],np.tile(requested,(b-a,1)))
        applied=raw['applied_command'][a]+np.clip(np.asarray(requested,np.float32)-raw['applied_command'][a],[-.25,0,-.35],[.25,0,.35])
        np.testing.assert_allclose(raw['applied_command'][a+1:b+1],np.tile(applied,(b-a,1)),rtol=0,atol=1e-7)
        np.testing.assert_array_equal(raw['phase'][a+1:b+1],np.full(b-a,phase))
    np.testing.assert_array_equal(raw['requested_command'][:750],np.zeros((750,3)))
    assert n==750+sum(t['post_sample_index']-t['pre_sample_index'] for t in tape)
    setup=read_json(directory,'setup_checks.json'); assert setup['setup']['velocity_and_nonfloor_setup_checks_pass']
    guard=dict(robot_geom_ids=friction[0]['robot_geom_ids'],foot_geom_ids=[int(k) for k in setup['feet']['native_foot_geom_to_shape']],
               ground_geom_ids=roles['physical_ground_geom_ids'])
    guards=[]
    for i in range(750,n):
        indices=nonfoot_ground_contact_indices(contact_packet(contacts,i),**guard)
        speed=float(np.linalg.norm(raw['base_twist_world'][i,:3])); inside=bool((np.abs(raw['base_pose_world'][i,:2])<8).all())
        if result['physical_stop'] is None: assert not indices and speed<=.3 and inside
        guards.append(dict(sample_index=i,nonfoot_ground_contact_indices=indices,base_speed_m_s=speed,in_domain=inside,evaluator_only=True))
    logged_guards=read_json(directory,'native_guard_rows.json')
    assert logged_guards==guards[:len(logged_guards)]
    if result['physical_stop'] is None: assert logged_guards==guards
    poses=raw['base_pose_world']; rotations=Rotation.from_quat(poses[:,3:]).as_matrix(); R0=rotations[749]; p0=poses[749,:3]
    relative=(poses[:,:3]-p0)@R0; relative_rotation=np.einsum('ij,njk->nik',R0.T,rotations)
    yaw=np.arctan2(relative_rotation[:,1,0],relative_rotation[:,0,0])
    reports=event_reports(spec['order'],decisions,raw,rotations)
    errors=[]
    for row in decisions:
        pose=row['evidence']['current_pose']
        if pose is not None:
            errors.append(float(np.linalg.norm(np.asarray(pose['position_initial_body_m'])-relative[row['pre_sample_index']])))
    terminal=result['schedule_terminal']
    complete=terminal is not None and terminal['terminal']=='PULSE_SCHEDULE_COMPLETE'
    return dict(raw_sensor_audit_pass=True,raw_depth_checks_within1mm=all(r['within1mm'] for r in sensors['depth_checks']),
        raw_depth_checks=len(sensors['depth_checks']),replayed_sensor_decisions=len(decisions),command_ticks=len(tape),
        physics_samples=n,duration_s=float(raw['timestamp_s'][-1]),native_stop=result['physical_stop'],
        actual_new_spawn_verified=True,old_fitting_prefix_exact=prefix_equal,old_servo_prefix_exact=old_servo_prefix_equal,
        initial_yaw_rad=initial_yaw,schedule_complete=complete,terminal_schedule=terminal,
        events=reports,complete_two_second_events=sum(e['endpoints']['brake_20']['native'] is not None for e in reports),
        maximum_visual_position_error_m=max(errors,default=None),
        model_fitting=False,independent_validation=False,real_time_qualified=False,navigation_qualified=False,goal_achieved=False)


def main():
    target=OUTPUT/'raw_pulse_audit_launch.json'
    if target.exists(): raise ValueError('exclusive servo audit')
    cv2.setNumThreads(1); launch=read_json(OUTPUT,'launch.json'); verify(launch); result=read_json(OUTPUT,'result.json')
    if result['absent_expected_artifacts']: raise ValueError('partial artifact absence requires explicit partial audit')
    inputs=launch['input_sha256']|{str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    sources=discover_sources(('scripts/audit_go2_command_pulse_response_v1.py',),launch['source_sha256'])
    verify_bindings(inputs|sources); write_json(target,dict(source_sha256=sources,input_sha256=inputs)); reports={}
    try:
        for c in TRIALS:
            reports[c]=audit_condition(c,result['conditions'][c]); write_json(OUTPUT/(c+'_pulse_evaluation.json'),reports[c])
            print('PULSE_AUDIT',c,{k:v for k,v in reports[c].items() if k not in ('events','terminal_schedule')},flush=True)
        verify(launch); verify_bindings(inputs|sources)
        write_json(OUTPUT/'raw_pulse_audit.json',dict(status='RAW_PULSE_AUDIT_PASS' if all(r['raw_sensor_audit_pass'] for r in reports.values()) else 'RAW_PULSE_AUDIT_PARTIAL',conditions=reports,
            audit_launch_sha256=digest(target),evaluation_sha256={c:digest(OUTPUT/(c+'_pulse_evaluation.json')) for c in TRIALS},
            goal_achieved=False,navigation_qualified=False))
    except Exception as error:
        write_json(OUTPUT/'raw_pulse_audit_failure.json',dict(status='TERMINAL_RAW_PULSE_AUDIT_FAILURE',reason=repr(error),completed_conditions=reports)); raise


if __name__=='__main__': main()
