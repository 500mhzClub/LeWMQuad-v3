"""Raw acquisition audit and explicitly non-permissive coverage diagnostics."""
from dataclasses import asdict
import json

import cv2
import numpy as np
import trimesh

from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgbd_shadow_motion_development import ShadowObserver
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from lewm.longer_motion_collection_development import TRIALS, specification, schedule, TAIL_TICKS
from lewm_genesis.appearance_surface_development import triangle_identity
from lewm_genesis.rgbd_motion_scene_development import independently_seeded_surfaces
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same, native_materials
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.fresh_maze_session_development import priors
from scripts.probe_go2_bounded_depth_surface_development_v1 import HYPOTHESES
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_raw_audit_development import audit_sensors, contact_packet, classify
from scripts.run_go2_longer_observed_floor_motion_development_v1 import OUTPUT, PROTOCOL, artifact_names
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def audit_tape(raw, tape):
    """Reconstruct every requested/applied command and clock, not just endpoints."""
    expected = schedule()+[dict(segment='terminal_zero_tail', phase=10,
        requested_command=[0., 0., 0.]) for _ in range(TAIL_TICKS)]
    if len(tape) != len(expected) or len(raw['timestamp_s']) != 30000:
        raise ValueError('incomplete tape is retained, not certified complete')
    np.testing.assert_array_equal(raw['requested_command'][:750], np.zeros((750, 3)))
    np.testing.assert_array_equal(raw['applied_command'][:750], np.zeros((750, 3)))
    np.testing.assert_array_equal(raw['phase'][:750], np.zeros(750))
    np.testing.assert_array_equal(raw['post_slew_applied_command'], raw['applied_command'])
    np.testing.assert_array_equal(np.rint(raw['timestamp_s']*1e9), np.arange(1,30001)*2_000_000)
    previous = None
    for i, (item, wanted) in enumerate(zip(tape, expected, strict=True)):
        pre, end = 749+50*i, 799+50*i
        if (item['tick'] != i or not item['completed'] or item['pre_sample_index'] != pre
                or item['post_sample_index'] != end or any(item[k] != v for k,v in wanted.items())):
            raise ValueError('fixed tape identity or completion mismatch')
        np.testing.assert_array_equal(raw['requested_command'][pre+1:end+1], np.tile(wanted['requested_command'], (50,1)))
        np.testing.assert_array_equal(raw['phase'][pre+1:end+1], np.full(50, wanted['phase']))
        prior = raw['applied_command'][pre]
        projected = prior+np.clip(np.asarray(wanted['requested_command'], np.float32)-prior, [-.25,0.,-.35], [.25,0.,.35])
        np.testing.assert_allclose(raw['applied_command'][pre+1:end+1], np.tile(projected,(50,1)), atol=1e-7, rtol=0)
        start, finish = item['start_perf_counter_ns'], item['end_perf_counter_ns']
        if (not start <= item['command_finished_perf_counter_ns'] <= finish
                or (previous is not None and start < previous)
                or item['outer_wall_ms'] != (finish-start)/1e6):
            raise ValueError('command timing mismatch')
        previous = finish
    return dict(complete=True, motion_ticks=580, zero_tail_ticks=5, all_commands_and_clocks_reconstructed=True)


def segment_motion(raw, tape):
    result = []
    for name in dict.fromkeys(r['segment'] for r in tape):
        rows = [r for r in tape if r['segment']==name]
        lo, hi = rows[0]['pre_sample_index'], rows[-1]['post_sample_index']
        poses = raw['base_pose_world'][lo:hi+1]
        yaw = np.unwrap([np.arctan2(rotation_xyzw(p[3:])[1,0], rotation_xyzw(p[3:])[0,0]) for p in poses])
        window = raw['base_twist_world'][max(lo+1,hi-99):hi+1]
        result.append(dict(segment=name, first_sample=lo, last_sample=hi,
            net_translation_m=float(np.linalg.norm(poses[-1,:3]-poses[0,:3])),
            path_length_m=float(np.linalg.norm(np.diff(poses[:,:3], axis=0),axis=1).sum()),
            yaw_change_rad=float(yaw[-1]-yaw[0]),
            final200ms_max_speed_m_s=float(np.linalg.norm(window[:,:3],axis=1).max()),
            final200ms_max_angular_speed_rad_s=float(np.linalg.norm(window[:,3:],axis=1).max())))
    return result


def audit_trial(trial, result, definition):
    directory = OUTPUT/trial; spec = specification(trial)
    json_same(spec, read_json(directory,'specification.json'))
    if result['status'] != 'PHYSICAL_TAPE_COMPLETE_AUDIT_REQUIRED' or result['physical_stop'] is not None:
        raise ValueError('incomplete physical outcome requires a dedicated partial-trace audit')
    raw, contacts, topology, roles, cameras, relatives, geometry, sensors = audit_sensors(directory, spec, result)
    assert all(r['within1mm'] for r in sensors['depth_checks'])
    assert [c['physical_sample_index'] for c in cameras] == list(range(749,30000,50))
    initial = raw['base_pose_world'][749]; R0 = rotation_xyzw(initial[3:]); q0 = raw['joint_position'][749]
    velocity, region = priors(definition)
    native = read_json(directory,'startup_native_robot_geometry.json')
    feet = match_native_foot_geometries(native, geometry, q0, initial)
    setup = check_setup_snapshot(velocity, region, identity=(0,0,0), measured_ns=1_500_000_000,
        position_world_m=initial[:3], rotation_world_from_initial_body=R0,
        velocity_world_m_s=raw['base_twist_world'][749,:3], native_static_boxes=read_json(directory,'static_objects.json'),
        expected_nonfloor_names=('wide_front',), geometry=geometry, joint_position=q0)
    support = initial_ground_support_witness(classify(contact_packet(contacts,749),topology),
        expected_support_groups=['FL_calf','FR_calf','RL_calf','RR_calf'], ground_link_ids=topology['ground_link_ids'],
        geometry=geometry, joint_position=q0, position_world_m=initial[:3], rotation_world_from_body=R0)
    json_same(dict(velocity_prior=asdict(velocity), region_prior=asdict(region), setup=setup, support=support, feet=feet,
        sample_index=749, definition_sha256=definition, setup_region_used_for_navigation=False,
        evidence_role='EVALUATOR_ONLY_INITIAL_SETUP_NOT_FUTURE_GAIT_QUALIFICATION'), read_json(directory,'setup_checks.json'))
    assert setup['velocity_and_nonfloor_setup_checks_pass'] and support['initial_native_support_witness_present']
    guard = dict(robot_geom_ids=[r['geom_id'] for r in native], foot_geom_ids=sorted(feet['native_foot_geom_to_shape']),
                 ground_geom_ids=roles['physical_ground_geom_ids'])
    guards = []
    for i in range(750,len(raw['timestamp_s'])):
        pose = raw['base_pose_world'][i]; R = rotation_xyzw(pose[3:])
        assert pose[2]>=.15 and max(abs(np.arctan2(R[2,1],R[2,2])), abs(np.arcsin(np.clip(R[2,0],-1,1))))<=.70
        guards.append(dict(sample_index=i, measured_ns=int(round(raw['timestamp_s'][i]*1e9)),
            nonfoot_ground_contact_indices=nonfoot_ground_contact_indices(contact_packet(contacts,i), **guard),
            base_speed_m_s=float(np.linalg.norm(raw['base_twist_world'][i,:3])),
            in_declared_capture_domain=bool((np.abs(pose[:2])<8.).all()), evaluator_supervision_not_policy_input=True))
    json_same(guards,read_json(directory,'native_guard_rows.json'))
    assert not raw['physics_contact'].any()
    assert all(not r['nonfoot_ground_contact_indices'] and r['base_speed_m_s']<=.3 and r['in_declared_capture_domain'] for r in guards)
    tape = read_json(directory,'command_tape.json'); accounting = audit_tape(raw,tape)
    assert result['completed_motion_ticks']==580 and result['completed_zero_tail_ticks']==5
    json_same(read_json(directory,'actuator_identity.json')['effective'],read_json(directory,'terminal_actuator_gains.json'))
    json_same(native_materials(native),native_materials(read_json(directory,'terminal_native_robot_geometry.json')))
    environment = read_json(directory,'floor_visual_collision_identity.json')
    for name, expected in independently_seeded_surfaces(spec['geometry']['wall_boxes'],spec['appearance_arm'],spec['appearance_seed']):
        mesh = trimesh.load_mesh(directory/'visual_meshes'/(name+'.ply'),file_type='ply',process=False)
        np.testing.assert_array_equal(mesh.vertices,expected.vertices.astype(np.float32))
        np.testing.assert_array_equal(mesh.faces,expected.faces)
        np.testing.assert_array_equal(mesh.visual.vertex_colors,expected.visual.vertex_colors)
        assert triangle_identity(mesh.vertices,mesh.faces)==next(r['geometry'] for r in environment['visual_surfaces'] if r['name']==name)
    shadow = ShadowObserver(velocity); saved = read_json(directory,'shadow_observations.json')
    assert len(saved)==len(cameras)==586
    causal_queries = []; surface = None; shape_errors = dict.fromkeys([s['shape_id'] for s in geometry.supports(q0,np.eye(3))['shapes']],0.)
    for frame,item in enumerate(saved):
        policy,depth = load_rgbd_observation(directory,frame); now = policy['sensor_state']['decision_ns']
        actual = shadow.observe(policy,depth,load_fast_packet(directory,frame),now_ns=now)
        assert item['observation_index']==frame; json_same(actual,item['shadow'])
        if actual['state'] is None: continue
        state = actual['state']; json_same(state['depth_state'],relatives[frame]['observer'])
        if frame == 0:
            surface = BoundedDepthSurface(depth['depth_m'],depth['valid'],shadow.model.state.integrator.gravity/9.81,**HYPOTHESES)
        if surface.status == 'BOUNDED_MEASURED_SURFACE_AVAILABLE':
            query = surface.query(geometry, policy['sensor_state']['sensed']['joints']['values'][-1,:12],
                rotation_observation_from_body=state['depth_state']['relative_orientation']['rotation_initial_body_from_current_body'],
                translation_observation_from_body=state['fusion']['position_initial_body_m'],point_error_by_shape=shape_errors)
            causal_queries.append(dict(frame=frame, measured_ns=now, floor_coverage=query['floor_coverage']))
    json_same(shadow.failure,result['shadow_failure']); assert shadow.successes==result['shadow_successful_frames']
    # Estimator replay is complete before native scoring. Native poses below are
    # evaluator-only and NEVER recover the failed observer or authorize commands.
    evaluator_queries = []; position_errors = []; ranks = {}
    for frame,camera in enumerate(cameras):
        i = camera['physical_sample_index']; pose = raw['base_pose_world'][i]
        R, t = R0.T@rotation_xyzw(pose[3:]), R0.T@(pose[:3]-initial[:3])
        state = saved[frame]['shadow']['state']
        if state is not None:
            position_errors.append(float(np.linalg.norm(np.asarray(state['fusion']['position_initial_body_m'])-t)))
        motion = relatives[frame]['observer']['motion']
        if motion is not None: ranks[str(motion['rank'])]=ranks.get(str(motion['rank']),0)+1
        if surface is not None and surface.status=='BOUNDED_MEASURED_SURFACE_AVAILABLE':
            query = surface.query(geometry,raw['joint_position'][i],rotation_observation_from_body=R,
                translation_observation_from_body=t,point_error_by_shape=shape_errors)
            evaluator_queries.append(dict(frame=frame, measured_ns=int(round(raw['timestamp_s'][i]*1e9)),floor_coverage=query['floor_coverage']))
    def coverage_summary(rows):
        full = [r['frame'] for r in rows if all(r['floor_coverage'].values())]
        return dict(queries=len(rows),maximum_covered_shapes=max((sum(r['floor_coverage'].values()) for r in rows),default=0),
            first_full_coverage_frame=full[0] if full else None,full_coverage_frames=len(full))
    path = raw['base_pose_world'][749:,:3]
    details = dict(sensors=sensors, causal_initial_surface_queries=causal_queries,
        evaluator_only_initial_surface_queries=evaluator_queries,relative_depth_observations=relatives)
    summary = dict(accounting=accounting, physics_samples=len(raw['timestamp_s']),rgbd_frames=len(cameras),
        native_guard_violations=0, segment_motion=segment_motion(raw,tape),
        path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),net_translation_m=float(np.linalg.norm(path[-1]-path[0])),
        maximum_active_speed_m_s=max(r['base_speed_m_s'] for r in guards),
        first_observed_depth_rank=relatives[1]['observer']['motion']['rank'],all_motion_depth_rank_counts=ranks,
        shadow_successful_frames=shadow.successes,shadow_failure=shadow.failure,
        maximum_admitted_position_error_m=max(position_errors,default=None),
        initial_surface_status=None if surface is None else surface.status,
        causal_coverage=coverage_summary(causal_queries),evaluator_only_coverage=coverage_summary(evaluator_queries),
        maximum_interior_depth_error_m=max(r['maximum_error_m'] for r in sensors['depth_checks']),
        zero_additional_point_error_is_diagnostic_not_calibrated=True,validation_used_for_fitting=False,
        prospective_motion_qualified=False,navigation_qualified=False)
    return plain(summary),plain(details)


def main():
    cv2.setNumThreads(1)
    target = OUTPUT/'raw_acquisition_audit_launch.json'
    if target.exists(): raise ValueError('fresh audit only; no silent overwrite or retry')
    launch = read_json(OUTPUT,'launch.json'); result = read_json(OUTPUT,'result.json'); verify(launch)
    expected = {f'{t}/{n}' for t in TRIALS for n in artifact_names(result['trials'][t]['rgbd_frames'])}
    assert set(result['artifact_sha256'])==expected and not result['absent_expected_artifacts']
    inputs = {str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs |= {str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    sources = discover_sources(('scripts/audit_go2_longer_observed_floor_motion_development_v1.py',
        'lewm/tests/test_longer_motion_raw_audit_development.py'),launch['source_sha256'])
    verify_bindings(inputs|sources)
    write_json(target,dict(source_sha256=sources,input_sha256=inputs,scope='acquisition audit, no model fitting or navigation qualification'))
    summaries = {}; artifacts = {}
    try:
        for trial in TRIALS:
            summaries[trial],details = audit_trial(trial,result['trials'][trial],launch['source_sha256'][PROTOCOL])
            name = trial+'_raw_acquisition_audit_details.json'; write_json(OUTPUT/name,details); artifacts[name]=digest(OUTPUT/name)
            print('RAW_ACQUISITION_AUDIT_PASS '+trial+' '+json.dumps(summaries[trial]),flush=True)
        verify(launch); verify_bindings(inputs|sources)
        write_json(OUTPUT/'raw_acquisition_audit.json',dict(status='RAW_ACQUISITION_AUDIT_PASS',
            trials=summaries,artifact_sha256=artifacts,audit_launch_sha256=digest(target),
            physical_trials=2,independent_geometries=1,validation_used_for_fitting=False,
            error_bounds_calibrated=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'raw_acquisition_audit_failure.json',dict(status='TERMINAL_AUDIT_FAILURE',
            reason=str(error),completed_trials=summaries,artifact_sha256=artifacts)); raise


if __name__=='__main__': main()
