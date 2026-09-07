"""Raw sensor/controller replay plus evaluator-only motion prediction errors."""
from dataclasses import asdict

import numpy as np

from lewm.action_motion_identification_development import MotionState, motion_priors
from lewm.action_response_validation_development import MotionValidationController
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same, require
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same, padded_body_inside_setup, native_materials
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.run_go2_action_motion_validation_development_v1 import OUTPUT, PROTOCOL, artifact_names, specification, frozen_model, MODEL_IDENTITY
from scripts.audit_go2_action_motion_identification_development_v1 import primitive_poses, primitive_displacement_bounds, prediction_errors, audit_command_tape
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json, audit_sensors, contact_packet, classify


MODES={'velocity_persistence':'prediction',
       'position_persistence':'position_persistence_prediction',
       'sensor_response':'response_prediction'}


def compare_predictions(raw,geometry,decisions,cameras):
    results={}
    keys=('body_translation_error_m','body_rotation_error_rad','maximum_joint_error_rad',
          'maximum_primitive_point_error_upper_bound_m','maximum_primitive_centre_error_m')
    for mode,field in MODES.items():
        derived=[item | {'decision':item['decision'] | {'prediction':item['decision'].get(field)}} for item in decisions]
        rows=prediction_errors(raw,geometry,derived,cameras)
        by_frame={item['observation_index']:item['decision']['prediction'] for item in derived}
        for row in rows:
            if row['status']!='SCORED_EXECUTED_FUTURE': continue
            frame=row['observation_index']; pred=by_frame[frame]; h=int(round(row['horizon_s']*10))
            start=cameras[frame]['physical_sample_index']; end=start+50*h
            anchor=raw['base_pose_world'][start]; Ra=rotation_xyzw(anchor[3:])
            pose=raw['base_pose_world'][end]
            predicted=primitive_poses(geometry,pred['positions_current_body_m'][h],
                np.asarray(pred['rotations_current_body'][h]),pred['joints_rad'][h])
            actual=primitive_poses(geometry,Ra.T@(pose[:3]-anchor[:3]),Ra.T@rotation_xyzw(pose[3:]),raw['joint_position'][end])
            errors=[float(np.linalg.norm(a[1]-b[1])) for a,b in zip(predicted,actual,strict=True)]
            row['maximum_primitive_centre_error_m']=max(errors)
            row['worst_primitive_centre_shape_id']=predicted[int(np.argmax(errors))][0]
        summary=[]
        for h in range(1,5):
            selected=[r for r in rows if r['status']=='SCORED_EXECUTED_FUTURE' and abs(r['horizon_s']-.1*h)<1e-12]
            summary.append(dict(horizon_s=.1*h,scored=len(selected),
                metrics={k:dict(mean=float(np.mean([r[k] for r in selected])),
                                maximum=max(r[k] for r in selected)) for k in keys} if selected else {},
                centres_exceeding_illustrative_50mm=sum(r['maximum_primitive_centre_error_m']>.05 for r in selected)))
        results[mode]=dict(rows=rows,summary=summary)
    signatures=[[(r['observation_index'],r['horizon_s'],r['status']) for r in results[m]['rows']] for m in MODES]
    require(all(x==signatures[0] for x in signatures),'identical forecast populations and executed commands for all models')
    return results


def audit_outer_timings(timings,decisions,tape):
    outer=timings['outer']; require(len(decisions)<=len(outer)<=len(decisions)+1,'every attempted loop retained')
    previous=None; wall=[]
    for i,row in enumerate(outer):
        same(row['decision_index'],i)
        start,end=row['start_perf_counter_ns'],row['end_perf_counter_ns']
        require(type(start) is int and type(end) is int and 0<=start<=end,'monotonic outer timestamps')
        require(previous is None or start>=previous,'sequential nonoverlapping outer loops'); previous=end
        require(np.isfinite(row['outer_wall_ms']) and abs(row['outer_wall_ms']-(end-start)/1e6)<1e-9,'outer duration identity')
        if i>=len(decisions):
            require(not row['decision_recorded'] and not row['completed_without_exception'],'partial exceptional loop retained')
            continue
        same(row['observation_index'],decisions[i]['observation_index'])
        require(row['decision_recorded'],'every saved decision belongs to outer loop')
        fresh=row['fresh_capture_inside_loop']; require(type(fresh) is bool,'explicit fresh capture flag')
        same(row['captures_before'],row['observation_index']+(0 if fresh else 1))
        same(fresh,i>0)  # Initial packet was captured for admission before this timer.
        entries=[e for e in tape if e['decision_index']==i]
        same(row['command_tick_attempted'],bool(entries))
        component=timings['controller'][i]['controller_wall_ms']+sum(e['execution_wall_ms'] for e in entries)
        if fresh: component+=timings['captures'][row['observation_index']]['acquisition_and_depth_observer_ms']
        require(np.isfinite(component) and component>=0 and row['outer_wall_ms']+1e-6>=component,
                'outer interval encloses all timed components actually inside it')
        if fresh and row['completed_without_exception']: wall.append(row['outer_wall_ms'])
    return wall


def audit():
    launch=read_json(OUTPUT,'launch.json'); result=read_json(OUTPUT,'result.json')
    require(result['status']=='ACQUISITION_COMPLETE_AUDIT_REQUIRED','auditable acquisition required')
    same(launch['specification'],specification())
    expected=set(artifact_names(result['rgbd_frames']))
    require(set(result['artifact_sha256'])<=expected,'only explicit nonsealed acquisition artifacts')
    same(set(result['absent_expected_artifacts']),expected-set(result['artifact_sha256']))
    bindings=launch['source_sha256'] | launch['input_sha256'] | {
        str((OUTPUT/name).relative_to(ROOT)):sha for name,sha in result['artifact_sha256'].items()}
    bindings |= {str((OUTPUT/name).relative_to(ROOT)):digest(OUTPUT/name) for name in ('launch.json','result.json')}
    def verify():
        verify_bindings(bindings); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    verify()
    raw,contacts,topology,roles,cameras,relatives,geometry,sensors=audit_sensors(OUTPUT,specification(),result)
    initial=raw['base_pose_world'][749]; R0=rotation_xyzw(initial[3:])
    velocity,region=motion_priors(1_500_000_000,launch['source_sha256'][PROTOCOL])
    static=read_json(OUTPUT,'static_objects.json'); original=read_json(OUTPUT,'startup_admission.json')
    require(original['checks_sha256']==digest(OUTPUT/'startup_checks.json'),'initial native-support witness bound')
    native=read_json(OUTPUT,'startup_native_robot_geometry.json')
    initial_feet=match_native_foot_geometries(native,geometry,raw['joint_position'][749],initial)
    json_same(initial_feet,read_json(OUTPUT,'startup_checks.json')['feet'])
    support=initial_ground_support_witness(classify(contact_packet(contacts,749),topology),
        expected_support_groups=['FL_calf','FR_calf','RL_calf','RR_calf'],ground_link_ids=topology['ground_link_ids'],
        geometry=geometry,joint_position=raw['joint_position'][749],position_world_m=initial[:3],rotation_world_from_body=R0)
    json_same(support,read_json(OUTPUT,'startup_checks.json')['support'])
    require(support['initial_native_support_witness_present'],'raw initial support independently reconstructed')
    check=check_setup_snapshot(velocity,region,identity=(0,0,0),measured_ns=1_500_000_000,
        position_world_m=initial[:3],rotation_world_from_initial_body=R0,velocity_world_m_s=raw['base_twist_world'][749,:3],
        native_static_boxes=static,expected_nonfloor_names=tuple(r['native_name'] for r in static),
        geometry=geometry,joint_position=raw['joint_position'][749])
    require(check['velocity_and_nonfloor_setup_checks_pass'],'new full calibration region independently checked')
    json_same(dict(new_region=asdict(region),check=check,inherited_initial_checks_sha256=original['checks_sha256'],
        static_scene_through_expiry_is_declared_condition=True,scope='new calibration-arena assumption; not sensor data or maze prior'),
        read_json(OUTPUT,'motion_setup_checks.json'))
    admission=original | {'checks_sha256':digest(OUTPUT/'motion_setup_checks.json')}
    json_same(admission,read_json(OUTPUT,'motion_admission.json')); admission['identity']=tuple(admission['identity'])
    owner=MotionState(geometry,velocity_prior=velocity,region_prior=region,admission=admission)
    controller=MotionValidationController(owner,frozen_model(),MODEL_IDENTITY); decisions=read_json(OUTPUT,'motion_decisions.json')
    same(launch['model_canonical_sha256'],MODEL_IDENTITY); same(result['model_canonical_sha256'],MODEL_IDENTITY)
    require(len(decisions)==result['controller_decisions']<=61,'bounded decision population')
    for frame,item in enumerate(decisions):
        same(item['observation_index'],frame)
        policy,depth=load_rgbd_observation(OUTPUT,frame); now=policy['sensor_state']['decision_ns']
        actual=controller.observe(policy,depth,load_fast_packet(OUTPUT,frame),now_ns=now)
        json_same(actual,item['decision'])
        if not actual['state']['terminal']: json_same(owner.relative_observation(now_ns=now),relatives[frame]['observer'])
    for item in read_json(OUTPUT,'motion_tail_states.json'):
        frame=item['observation_index']; policy,depth=load_rgbd_observation(OUTPUT,frame); now=policy['sensor_state']['decision_ns']
        json_same(owner.observe(policy,depth,load_fast_packet(OUTPUT,frame),now_ns=now),item['state'])
    same(controller.status,result['controller_status']); same(owner.status,result['owner_status'])
    tape=read_json(OUTPUT,'motion_command_tape.json'); audit_command_tape(raw,tape,decisions,result)
    guard=dict(robot_geom_ids=[r['geom_id'] for r in native],foot_geom_ids=sorted(initial_feet['native_foot_geom_to_shape']),
        ground_geom_ids=roles['physical_ground_geom_ids'])
    violations=[]; pose_bounds=[]; rates=[]; previous=None; guard_rows=[]; region_rows=[]
    for i in range(750,len(raw['timestamp_s'])):
        pose=raw['base_pose_world'][i]; R=rotation_xyzw(pose[3:])
        bad=nonfoot_ground_contact_indices(contact_packet(contacts,i),**guard)
        inside=padded_body_inside_setup(geometry,region,initial,pose,raw['joint_position'][i])
        speed=float(np.linalg.norm(raw['base_twist_world'][i,:3]))
        body_bad=pose[2]<.15 or max(abs(np.arctan2(R[2,1],R[2,2])),abs(np.arcsin(np.clip(R[2,0],-1,1))))>.70
        if bad or not inside or speed>.3 or raw['physics_contact'][i] or body_bad:
            violations.append(i)
        if not (body_bad or raw['physics_contact'][i]):
            guard_rows.append(dict(sample_index=i,measured_ns=int(round(raw['timestamp_s'][i]*1e9)),
                base_speed_m_s=speed,nonfoot_ground_contact_indices=bad))
            if not bad and speed<=.3: region_rows.append(dict(sample_index=i,inside=bool(inside)))
        current=primitive_poses(geometry,pose[:3],R,raw['joint_position'][i])
        if previous is not None: rates.append(max(primitive_displacement_bounds(previous,current))/.002)
        previous=current
    same(guard_rows,read_json(OUTPUT,'startup_guard_rows.json')); same(region_rows,read_json(OUTPUT,'motion_region_rows.json'))
    require(not violations or (violations[0]==len(raw['timestamp_s'])-1 and result['physical_stop_reason'] is not None),
        'no physics after first physical guard failure')
    for item in decisions:
        envelope=item['decision']['envelope']
        if envelope is None: continue
        start=cameras[item['observation_index']]['physical_sample_index']; end=min(start+200,len(raw['timestamp_s'])-1)
        centre=np.asarray(envelope['centre_initial_body_m']); maximum=-np.inf
        for i in range(start,end+1):
            pose=raw['base_pose_world'][i]; p=R0.T@(pose[:3]-initial[:3]); R=R0.T@rotation_xyzw(pose[3:])
            for s in geometry.supports(raw['joint_position'][i],R)['shapes']:
                maximum=max(maximum,float(np.max(np.maximum(np.asarray(s['upper'])+p-centre,
                    centre-np.asarray(s['lower'])-p)+.04-envelope['half_extent_m'])))
        pose_bounds.append(dict(observation_index=item['observation_index'],full_horizon_recorded=end==start+200,
            maximum_sampled_padded_primitive_envelope_excess_m=float(maximum)))
    gains=read_json(OUTPUT,'actuator_identity.json'); same(gains['effective'],read_json(OUTPUT,'terminal_actuator_gains.json'))
    terminal=read_json(OUTPUT,'terminal_native_robot_geometry.json'); same(native_materials(native),native_materials(terminal))
    json_same(match_native_foot_geometries(terminal,geometry,raw['joint_position'][-1],raw['base_pose_world'][-1]),
              read_json(OUTPUT,'terminal_foot_identity.json'))
    timings=read_json(OUTPUT,'motion_timings.json')
    same([r['observation_index'] for r in timings['captures']],list(range(len(cameras))))
    same([r['observation_index'] for r in timings['controller']],[r['observation_index'] for r in decisions])
    wall=[]
    for i,row in enumerate(timings['controller']):
        capture=timings['captures'][row['observation_index']]['acquisition_and_depth_observer_ms']
        execution=next((e['execution_wall_ms'] for e in tape if e['decision_index']==i),0.)
        wall.append(float(capture+row['controller_wall_ms']+execution))
    require(all(np.isfinite(v) and v>=0 for v in wall),'finite sequential loop timing')
    speed=float(np.max(np.linalg.norm(raw['base_twist_world'][-50:,:3],axis=1)))
    angular=float(np.max(np.linalg.norm(raw['base_twist_world'][-50:,3:],axis=1)))
    quiet=speed<=.05 and angular<=.1
    predictions=compare_predictions(raw,geometry,decisions,cameras)
    outer_wall=audit_outer_timings(timings,decisions,tape)
    success=bool(controller.status=='COMPLETE_VALIDATION_SCHEDULE' and result['physical_stop_reason'] is None
        and result['sensor_stop_reason'] is None and owner.status=='READY_FOR_NAVIGATION_CONSUMER'
        and not violations and result['stopping_tail_ticks']==3 and quiet and not result['absent_expected_artifacts']
        and raw['timestamp_s'][-1]*1e9<=region.valid_until_ns and all(r['within1mm'] for r in sensors['depth_checks'])
        and all(r['maximum_sampled_padded_primitive_envelope_excess_m']<=1e-12 for r in pose_bounds))
    verify()
    return dict(status='RAW_MOTION_VALIDATION_REPLAY_COMPLETE',bounded_validation_execution_complete=success,
        physics_samples=len(raw['timestamp_s']),rgbd_frames=len(cameras),controller_reconstruction_exact=True,
        controller_status=controller.status,owner_status=owner.status,physical_stop_sample_indices=violations,
        physical_stop_reason=result['physical_stop_reason'],sensor_stop_reason=result['sensor_stop_reason'],
        sensor_report=sensors,prediction_errors=predictions,envelope_diagnostics=pose_bounds,
        maximum_sampled_material_point_displacement_rate_upper_bound_m_s=max(rates) if rates else None,
        continuous_point_speed_bound_established=False,final_window_max_speed_m_s=speed,
        final_window_max_angular_speed_rad_s=angular,final_quiet=quiet,component_sum_wall_ms=wall,
        complete_capture_decision_execution_outer_wall_ms=outer_wall,
        outer_loop_records=timings['outer'],first_loop_capture_outside_timer=True,
        model_canonical_sha256=MODEL_IDENTITY,
        empirical_model_fitted=True,fixed_B_comparison_complete=success,
        independent_maze_validation_complete=False,contact_model_validated=False,
        real_time_qualified=False,navigation_qualified=False)


def main():
    target=OUTPUT/'raw_artifact_audit.json'; require(not target.exists(),'fresh independent audit output; no overwrite')
    report=audit(); write_json(target,report)
    print({k:v for k,v in report.items() if k not in ('sensor_report','prediction_errors','envelope_diagnostics','component_sum_wall_ms',
        'complete_capture_decision_execution_outer_wall_ms','outer_loop_records')},flush=True)


if __name__=='__main__': main()
