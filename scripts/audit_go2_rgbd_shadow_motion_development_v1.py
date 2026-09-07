"""Independent raw reconstruction, immutable shadow replay, then error scoring."""
from dataclasses import asdict
import json

import cv2
import numpy as np
import trimesh

from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries,nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgbd_shadow_motion_development import schedule,priors,ShadowObserver
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot,initial_ground_support_witness
from lewm_genesis.appearance_surface_development import triangle_identity
from lewm_genesis.rgbd_motion_scene_development import independently_seeded_surfaces
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same,padded_body_inside_setup,native_materials,stopping_metrics
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_raw_audit_development import audit_sensors,read_npz,contact_packet,classify
from scripts.run_go2_rgbd_shadow_motion_development_v1 import OUTPUT,PROTOCOL,ARMS,APPEARANCE_SEED,specification,artifact_names
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

IDENTITIES={'launch.json':'628168df54a4e8d96575f939609057f610de35d60fd579cd58d76ca58dd964dc',
    'result.json':'4a5b97bb50f28ca631bdb1bb1e3ee953658105613742333c638429b0c482bfd1'}


def audit_arm(arm,result,definition):
    directory=OUTPUT/arm;spec=specification(arm)
    json_same(spec,read_json(directory,'specification.json'))
    raw,contacts,topology,roles,cameras,relative,geometry,sensors=audit_sensors(directory,spec,result)
    assert all(r['within1mm'] for r in sensors['depth_checks'])
    initial=raw['base_pose_world'][749];R0=rotation_xyzw(initial[3:]);velocity,region=priors(definition)
    native=read_json(directory,'startup_native_robot_geometry.json')
    feet=match_native_foot_geometries(native,geometry,raw['joint_position'][749],initial)
    check=check_setup_snapshot(velocity,region,identity=(0,0,0),measured_ns=1_500_000_000,
        position_world_m=initial[:3],rotation_world_from_initial_body=R0,velocity_world_m_s=raw['base_twist_world'][749,:3],
        native_static_boxes=read_json(directory,'static_objects.json'),expected_nonfloor_names=tuple(r['wall_id'] for r in spec['geometry']['wall_boxes']),
        geometry=geometry,joint_position=raw['joint_position'][749])
    support=initial_ground_support_witness(classify(contact_packet(contacts,749),topology),
        expected_support_groups=['FL_calf','FR_calf','RL_calf','RR_calf'],ground_link_ids=topology['ground_link_ids'],
        geometry=geometry,joint_position=raw['joint_position'][749],position_world_m=initial[:3],rotation_world_from_body=R0)
    json_same(dict(velocity_prior=asdict(velocity),region_prior=asdict(region),setup=check,support=support,feet=feet,
        sample_index=749,definition_sha256=definition,evidence_role='EVALUATOR_ONLY_BOUNDED_ACQUISITION_NOT_DEPLOYMENT_SENSOR'),
        read_json(directory,'setup_checks.json'))
    assert check['velocity_and_nonfloor_setup_checks_pass'] and support['initial_native_support_witness_present']
    guard=dict(robot_geom_ids=[r['geom_id'] for r in native],foot_geom_ids=sorted(feet['native_foot_geom_to_shape']),
        ground_geom_ids=roles['physical_ground_geom_ids'])
    guards=[]
    for i in range(750,len(raw['timestamp_s'])):
        now=int(round(raw['timestamp_s'][i]*1e9))
        row=dict(sample_index=i,measured_ns=now,nonfoot_ground_contact_indices=nonfoot_ground_contact_indices(contact_packet(contacts,i),**guard),
            base_speed_m_s=float(np.linalg.norm(raw['base_twist_world'][i,:3])),
            padded_body_inside_region=bool(padded_body_inside_setup(geometry,region,initial,raw['base_pose_world'][i],raw['joint_position'][i])),
            region_active=bool(region.anchor_ns<=now<=region.valid_until_ns))
        guards.append(row)
    json_same(guards,read_json(directory,'native_guard_rows.json'))
    violations=[r for r in guards if r['nonfoot_ground_contact_indices'] or r['base_speed_m_s']>.3 or not r['padded_body_inside_region'] or not r['region_active']]
    tape=read_json(directory,'command_tape.json');wanted=schedule()+[[0.,0.,0.]]*3
    assert len(tape)==53 and len(raw['timestamp_s'])==3400 and result['status']=='PHYSICAL_TAPE_COMPLETE'
    assert not violations and not raw['physics_contact'].any() and result['physical_stop'] is None
    np.testing.assert_array_equal(raw['requested_command'][:750],np.zeros((750,3)))
    np.testing.assert_array_equal(raw['applied_command'][:750],np.zeros((750,3)))
    np.testing.assert_array_equal(raw['post_slew_applied_command'],raw['applied_command'])
    for pose in raw['base_pose_world']:
        R=rotation_xyzw(pose[3:])
        assert pose[2]>=.15 and max(abs(np.arctan2(R[2,1],R[2,2])),abs(np.arcsin(np.clip(R[2,0],-1,1))))<=.70
    previous_end=None
    for i,item in enumerate(tape):
        pre=749+50*i;end=pre+50
        assert item['tick']==i and item['phase']==(1 if i<50 else 2) and item['completed']
        assert item['pre_sample_index']==pre and item['post_sample_index']==end and item['requested_command']==wanted[i]
        np.testing.assert_array_equal(raw['requested_command'][pre+1:end+1],np.tile(wanted[i],(50,1)))
        np.testing.assert_array_equal(raw['phase'][pre+1:end+1],np.full(50,item['phase']))
        # Reconstruct the deployed clip/slew contract from the prior actual tick.
        prior_applied=raw['applied_command'][pre]
        projected=prior_applied+np.clip(np.asarray(wanted[i],np.float32)-prior_applied,[-.25,0.,-.35],[.25,0.,.35])
        np.testing.assert_allclose(raw['applied_command'][pre+1:end+1],np.tile(projected,(50,1)),atol=1e-7,rtol=0)
        start,finish=item['start_perf_counter_ns'],item['end_perf_counter_ns']
        assert start<=item['command_finished_perf_counter_ns']<=finish
        assert previous_end is None or start>=previous_end
        assert item['outer_wall_ms']==(finish-start)/1e6;previous_end=finish
    assert read_json(directory,'actuator_identity.json')['effective']==read_json(directory,'terminal_actuator_gains.json')
    assert native_materials(native)==native_materials(read_json(directory,'terminal_native_robot_geometry.json'))
    # Serialized surfaces match the fixed fresh generator and native geometry.
    environment=read_json(directory,'floor_visual_collision_identity.json')
    for name,expected in independently_seeded_surfaces(spec['geometry']['wall_boxes'],arm,APPEARANCE_SEED):
        mesh=trimesh.load_mesh(directory/'visual_meshes'/(name+'.ply'),file_type='ply',process=False)
        np.testing.assert_array_equal(mesh.vertices,expected.vertices.astype(np.float32));np.testing.assert_array_equal(mesh.faces,expected.faces)
        np.testing.assert_array_equal(mesh.visual.vertex_colors,expected.visual.vertex_colors)
        wanted_geometry=next(r['geometry'] for r in environment['visual_surfaces'] if r['name']==name)
        assert triangle_identity(mesh.vertices,mesh.faces)==wanted_geometry
    shadow=ShadowObserver(velocity);saved=read_json(directory,'shadow_observations.json');failure_diagnostic=None
    assert len(saved)==len(cameras)==54
    for frame,item in enumerate(saved):
        assert item['observation_index']==frame
        policy,depth=load_rgbd_observation(directory,frame);now=policy['sensor_state']['decision_ns']
        actual=shadow.observe(policy,depth,load_fast_packet(directory,frame),now_ns=now)
        json_same(actual,item['shadow'])
        if actual['state'] is not None:json_same(actual['state']['depth_state'],relative[frame]['observer'])
        if actual['status']=='TERMINAL_SHADOW_FAILURE':
            integrator=shadow.model.state.integrator
            sensitivity=integrator.prior_sensitivity.snapshot(velocity.radius_m_s)
            inherited=float(3*np.sqrt(integrator.position_proxy))
            failure_diagnostic=dict(measured_ns=now,consecutive_weak_seconds=integrator.weak_seconds,
                inherited_position_scale_m=inherited,point_position_scale_m=integrator.point_position_scale,
                initial_velocity_position_radius_m=sensitivity['position_radius_m'],
                combined_position_scale_m=inherited+integrator.point_position_scale+sensitivity['position_radius_m'],
                failed_frame_admitted_to_memory=False)
    json_same(shadow.failure,result['shadow_failure']);assert shadow.successes==result['shadow_successful_frames']
    # All saved shadow states/failures have now been reconstructed. Truth below
    # is scoring only and cannot recover a failed estimator or fill missing rows.
    errors=[];point_errors=[];complemented=0;ranks={};common=[]
    for frame,item in enumerate(saved):
        state=item['shadow']['state']
        if state is None:continue
        index=cameras[frame]['physical_sample_index'];pose=raw['base_pose_world'][index]
        truth=R0.T@(pose[:3]-initial[:3]);fusion=state['fusion']
        error=float(np.linalg.norm(np.asarray(fusion['position_initial_body_m'])-truth))
        errors.append(error)
        if frame<28:common.append(error)
        if frame:
            before=raw['base_pose_world'][cameras[frame-1]['physical_sample_index']]
            delta=rotation_xyzw(before[3:]).T@(pose[:3]-before[:3])
            point=state['point_state']['motion']['translation_previous_body_m']
            if point is not None:point_errors.append(float(np.linalg.norm(np.asarray(point)-delta)))
            rank=state['depth_state']['motion']['rank'];ranks[str(rank)]=ranks.get(str(rank),0)+1
            complemented+=int(fusion['constraints']['point_used_for_weak_directions'])
    last=next(item['shadow']['state']['fusion'] for item in saved[::-1] if item['shadow']['state'] is not None)
    wall=[r['outer_wall_ms'] for r in tape]
    return raw,dict(physical_tape_complete=True,physics_samples=3400,rgbd_frames=54,
        motion_ticks=50,tail_ticks=3,native_guard_violations=0,stopping=stopping_metrics(raw,tape),
        maximum_body_speed_m_s=max(r['base_speed_m_s'] for r in guards),
        net_motion_displacement_m=float(np.linalg.norm(raw['base_pose_world'][-1,:3]-initial[:3])),
        shadow_successful_frames=shadow.successes,shadow_failure=shadow.failure,failure_diagnostic=failure_diagnostic,
        accepted_point_pairs=len(point_errors),point_complemented_pairs=complemented,depth_rank_counts_in_admitted_pairs=ranks,
        maximum_point_translation_error_m=max(point_errors,default=None),mean_point_translation_error_m=float(np.mean(point_errors)) if point_errors else None,
        maximum_admitted_position_error_m=max(errors),last_admitted_position_error_m=errors[-1],
        common_first28_maximum_position_error_m=max(common),last_admitted_position_scale_m=last['position_error_scale_m'],
        minimum_outer_tick_ms=min(wall),median_outer_tick_ms=float(np.median(wall)),maximum_outer_tick_ms=max(wall),
        ticks_exceeding100ms=sum(t>100 for t in wall),maximum_interior_depth_error_m=max(r['maximum_error_m'] for r in sensors['depth_checks']),
        native_pose_used_for_shadow_replay=False,uncertainty_model_validated=False,navigation_qualified=False)


def audit():
    cv2.setNumThreads(1)
    bindings={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(bindings)
    launch=read_json(OUTPUT,'launch.json');result=read_json(OUTPUT,'result.json');verify(launch)
    assert result['status']=='SHADOW_MOTION_ACQUISITION_COMPLETE_AUDIT_REQUIRED'
    expected={f'{arm}/{n}' for arm,r in result['arms'].items() for n in artifact_names(r)}
    assert set(result['artifact_sha256'])==expected and not result['absent_expected_artifacts']
    bindings|={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    own=('scripts/audit_go2_rgbd_shadow_motion_development_v1.py','scripts/rgbd_shadow_motion_raw_audit_development.py')
    sources={n:digest(ROOT/n) for n in own};verify_bindings(bindings|sources)
    statistics={};reference=None
    for arm in ARMS:
        raw,statistics[arm]=audit_arm(arm,result['arms'][arm],launch['source_sha256'][PROTOCOL])
        if reference is None:reference=raw
        else:
            assert set(raw)==set(reference)
            for key in raw:np.testing.assert_array_equal(raw[key],reference[key])
        print('RAW_SENSOR_SHADOW_AUDIT_PASS '+arm,flush=True)
    verify(launch);verify_bindings(bindings|sources)
    return dict(status='FRESH_PHYSICAL_SHADOW_MOTION_RAW_AUDIT_PASS',statistics=statistics,
        all_matched_arm_physics_arrays_identical=True,artifact_count=len(expected),identities=IDENTITIES,
        auditor_source_sha256=sources,independent_layout_trials=1,estimator_selects_commands=False,navigation_qualified=False)


if __name__=='__main__':
    target=OUTPUT/'raw_artifact_audit.json'
    if target.exists():raise ValueError('fresh independent audit only')
    result=audit();write_json(target,result);print(json.dumps(result),flush=True)
