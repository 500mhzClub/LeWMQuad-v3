"""Raw acquisition and sensor-command replay, followed by native target scoring."""
import json
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.goal_hold_visual_servo_development import GoalHoldVisualServo
from lewm.bounded_visual_servo_development import wrapped
from lewm.goal_hold_visual_servo_scene_development import specification
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.visual_led_motion_development import VisualLedMotion
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_raw_audit_development import audit_sensors,contact_packet
from scripts.run_go2_goal_hold_visual_servo_v1 import OUTPUT,CONDITIONS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources


def score_native_hold(relative,yaw,twist,start,end):
    if start<0 or end-start!=500: raise ValueError('exact one-second native interval required')
    p=np.asarray(relative)[start:end+1]; angle=np.asarray(yaw)[start:end+1]
    speed=np.linalg.norm(np.asarray(twist)[start:end+1,:3],axis=1)
    yaw_rate=np.abs(np.diff(np.unwrap(angle))/.002)
    pe=np.linalg.norm(p[:,:2]-[.4,0.],axis=1)
    ye=np.abs(np.arctan2(np.sin(.3-angle),np.cos(.3-angle)))
    if len(p)!=501 or not all(np.isfinite(x).all() for x in (pe,ye,speed,yaw_rate)):
        raise ValueError('complete finite native hold required')
    return dict(start_sample=start,end_sample=end,physics_pose_samples=501,
        maximum_planar_error_m=float(pe.max()),maximum_yaw_error_rad=float(ye.max()),
        maximum_speed_m_s=float(speed.max()),maximum_yaw_rate_rad_s=float(yaw_rate.max()),
        passed=bool(pe.max()<=.06 and ye.max()<=.05 and speed.max()<=.02 and yaw_rate.max()<=.05))


def audit_condition(condition,result):
    directory=OUTPUT/condition; spec=specification(condition)
    raw,contacts,topology,roles,cameras,relatives,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s']); assert n<=18750
    # Changed actual physics prefix, not just a different random seed label.
    prior=ROOT/'.generated/go2_longer_observed_floor_motion_development_v1_attempt_001/fit'
    old=read_npz(prior,'physics_trace.npz')
    prefix_equal=bool(np.array_equal(raw['base_pose_world'][:750],old['base_pose_world'][:750]))
    assert not prefix_equal
    previous_servo=read_npz(ROOT/'.generated/go2_course_aware_visual_servo_v1_attempt_001'/condition,'physics_trace.npz')
    old_servo_prefix_equal=bool(np.array_equal(raw['base_pose_world'][:750],previous_servo['base_pose_world'][:750]))
    assert not old_servo_prefix_equal
    np.testing.assert_allclose(raw['base_pose_world'][0,:2],[-.60,-.10],rtol=0,atol=.002)
    initial_yaw=float(Rotation.from_quat(raw['base_pose_world'][0,3:]).as_euler('xyz')[2])
    assert abs(wrapped(initial_yaw-.06))<.002
    friction=read_json(directory,'friction_checks.json')
    for row in friction:
        np.testing.assert_allclose(row['solver_friction'],spec['friction_mu'],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(row['solver_ratio'],np.ones((1,28)))
    assert friction[0]['physics_steps']==0 and friction[-1]['physics_steps']==n
    assert read_json(directory,'actuator_identity.json')['effective']==read_json(directory,'terminal_actuator_gains.json')
    decisions=read_json(directory,'servo_decisions.json'); tape=read_json(directory,'command_tape.json')
    assert len(decisions)==result['decisions'] and len(tape)==result['command_ticks']
    motion=VisualLedMotion('gyro',identity=(0,0,0)); controller=GoalHoldVisualServo(); desired=[]
    for tick,row in enumerate(decisions):
        frame=row['observation_index']; assert row['tick']==tick and frame==tick
        p,d=load_rgbd_observation(directory,frame); fast=load_fast_packet(directory,frame); now=p['sensor_state']['decision_ns']
        observed=motion.observe(p,d,fast,now_ns=now); command=controller.step(observed,now_ns=now)
        assert json.loads(json.dumps(observed))==row['evidence'] and command==row['decision']
        assert row['pre_sample_index']==cameras[frame]['physical_sample_index']==749+tick*50
        if command['terminal'] is None: desired.append((command['requested_command'],command['phase'],'visual_feedback'))
        elif command['terminal']=='VISUAL_SERVO_FAILED': desired.extend([([0.,0.,0.],9,'terminal_zero_command_drain')]*10)
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
    position_error=float(np.linalg.norm(relative[-1,:2]-[.4,0.])); yaw_error=abs(wrapped(.3-float(yaw[-1])))
    final_speed=float(np.max(np.linalg.norm(raw['base_twist_world'][-100:,:3],axis=1)))
    final_angular=float(np.max(np.linalg.norm(raw['base_twist_world'][-100:,3:],axis=1)))
    brake=[]
    for phase in (2,4,9):
        indices=np.flatnonzero(raw['phase']==phase)
        if not len(indices): continue
        for segment in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            a,b=int(segment[0])-1,int(segment[-1])
            brake.append(dict(phase=phase,start_s=float(raw['timestamp_s'][a]),end_s=float(raw['timestamp_s'][b]),
                displacement_m=float(np.linalg.norm(poses[b,:3]-poses[a,:3])),yaw_drift_rad=wrapped(float(yaw[b]-yaw[a]))))
    errors=[]
    for row in decisions:
        pose=row['evidence']['current_pose']
        if pose is None: continue
        i=row['pre_sample_index']; errors.append(float(np.linalg.norm(np.asarray(pose['position_initial_body_m'])-relative[i])))
    terminal=result['controller_terminal']; completed=terminal is not None and terminal['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'
    hold=None
    if completed:
        assert result['physical_stop'] is None and terminal==decisions[-1]['decision']
        end=decisions[-1]['pre_sample_index']; start=end-500
        np.testing.assert_array_equal(raw['requested_command'][start+1:end+1],np.zeros((500,3)))
        np.testing.assert_array_equal(raw['phase'][start+1:end+1],np.full(500,4))
        for row in decisions[-11:]:
            d=row['decision']['diagnostic']
            assert d['position_error_m']<=.06 and abs(d['yaw_error_rad'])<=.05
        hold=score_native_hold(relative,yaw,raw['base_twist_world'],start,end)
    return dict(raw_sensor_audit_pass=True,raw_depth_checks_within1mm=all(r['within1mm'] for r in sensors['depth_checks']),
        raw_depth_checks=len(sensors['depth_checks']),replayed_sensor_decisions=len(decisions),command_ticks=len(tape),
        physics_samples=n,duration_s=float(raw['timestamp_s'][-1]),native_stop=result['physical_stop'],
        actual_new_spawn_verified=True,old_fitting_prefix_exact=prefix_equal,
        old_servo_prefix_exact=old_servo_prefix_equal,initial_yaw_rad=initial_yaw,
        visual_controller_complete=completed,terminal_controller=terminal,native_final_hold=hold,
        complete_local_task_success=completed and hold is not None and hold['passed'],
        native_final_position_initial_body_m=relative[-1].tolist(),native_final_yaw_rad=float(yaw[-1]),
        native_final_planar_error_m=position_error,native_final_yaw_error_rad=yaw_error,
        native_final_200ms_maximum_speed_m_s=final_speed,native_final_200ms_maximum_angular_speed_rad_s=final_angular,
        native_planar_yaw_targets_met=position_error<=.06 and yaw_error<=.05,
        native_final_speed_limit_met=final_speed<=.02,
        maximum_visual_position_error_m=max(errors,default=None),brake_segments=brake,
        observed_maximum_forward_x_m=float(relative[749:,0].max()),
        model_fitting=False,real_time_qualified=False,navigation_qualified=False,goal_achieved=False)


def main():
    target=OUTPUT/'raw_servo_audit_launch.json'
    if target.exists(): raise ValueError('exclusive servo audit')
    cv2.setNumThreads(1); launch=read_json(OUTPUT,'launch.json'); verify(launch); result=read_json(OUTPUT,'result.json')
    if result['absent_expected_artifacts']: raise ValueError('partial artifact absence requires explicit partial audit')
    inputs=launch['input_sha256']|{str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    sources=discover_sources(('scripts/audit_go2_goal_hold_visual_servo_v1.py',),launch['source_sha256'])
    verify_bindings(inputs|sources); write_json(target,dict(source_sha256=sources,input_sha256=inputs)); reports={}
    try:
        for c in CONDITIONS:
            reports[c]=audit_condition(c,result['conditions'][c]); write_json(OUTPUT/(c+'_servo_evaluation.json'),reports[c])
            print('BOUNDED_VISUAL_SERVO_AUDIT',c,json.dumps(reports[c]),flush=True)
        verify(launch); verify_bindings(inputs|sources)
        write_json(OUTPUT/'raw_servo_audit.json',dict(status='RAW_SERVO_AUDIT_PASS',conditions=reports,
            audit_launch_sha256=digest(target),evaluation_sha256={c:digest(OUTPUT/(c+'_servo_evaluation.json')) for c in CONDITIONS},
            goal_achieved=False,navigation_qualified=False))
    except Exception as error:
        write_json(OUTPUT/'raw_servo_audit_failure.json',dict(status='TERMINAL_RAW_SERVO_AUDIT_FAILURE',reason=repr(error),completed_conditions=reports)); raise


if __name__=='__main__': main()
