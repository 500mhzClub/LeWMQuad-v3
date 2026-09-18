"""Robot-visible raw sensor characterization on a fixed recorded turn prefix."""
import time
import cv2
import numpy as np
import torch
from lewm.geometry_progress_layout_family_development import specification
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.support_friction_challenge_development import native_friction
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.auxiliary_tilted_depth_geometry_development import reference_pose,body_from_optical
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.auxiliary_depth_visible_robot_session_development import VisibleRobotFamilySession
from scripts.auxiliary_tilted_depth_capture_development import capture
from scripts.geometry_progress_family_runtime_development import preflight,verify
from scripts.geometry_progress_family_episode_development import artifacts
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.characterize_go2_auxiliary_tilted_depth_geometry_v1 import OUTPUT as GEOMETRY
from scripts.run_go2_training_bias_goal_probe_v1 import OUTPUT as INPUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_auxiliary_tilted_depth_prefix_v1_attempt_001'
PROTOCOL='docs/go2_auxiliary_tilted_depth_prefix_v1_2026-09-08.md'
GEOMETRY_SHA='8059e48ee6a7092fcaea51950191c2aa587aa208955889b1759b154d1be01265'
INPUT_SHA='5e48672a074d2086d734d02844af499d49d0a57571b9148b433b65ad2bedcfb5'
TRIAL='family_episode_039'
CASE='full_jepa_family_episode_039'
COMMAND_TICKS=19


def collect(launch,tape):
    directory=OUTPUT/'sensor_prefix';directory.mkdir();spec=specification(TRIAL)
    write_json(directory/'specification.json',spec);session=None;aux=[];commands=[];friction=[];gains=None
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=VisibleRobotFamilySession(spec,directory);session.install_contact_identity();build=session.ctx.build
        gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(directory/'actuator_identity.json',gains)
        write_json(directory/'floor_roles.json',dict(physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle',**native_friction(build,spec['friction_mu'])))
        session.settle_recorded();session.capture_current();admit_context_setup(session,launch['source_sha256'][PROTOCOL])
        for tick in range(COMMAND_TICKS+1):
            if hardware()['artifact_free_bytes']<40*1024**3+64*1024**2:raise ValueError('capture storage reserve')
            friction.append(dict(stage='before_capture',tick=tick,**native_friction(build,spec['friction_mu'])))
            assert len(session.samples)==750+50*tick and len(session.model_manifest)==tick+1
            aux.append(capture(session,directory,tick))
            if tick==COMMAND_TICKS:break
            item=dict(tape[tick]);assert item['tick']==tick and item['requested_command'][:2]==[0.,0.]
            session.phase=item['phase'];item['role']='fixed_recorded_turn_for_sensor_characterization'
            item['completed']=False;commands.append(item)
            session.command_tick(item['requested_command']);item['completed']=True
            assert len(session.samples)-1==item['post_sample_index']
        terminal_gains=read_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist())
        assert terminal_gains==gains['effective']
        write_json(directory/'terminal_actuator_gains.json',terminal_gains)
        write_json(directory/'terminal_native_robot_geometry.json',capture_native_robot_geometry(build.robot))
        write_json(directory/'terminal_environment_identity.json',appearance_environment_identity(session))
        result=dict(status='AUXILIARY_SENSOR_PREFIX_COLLECTED',trial=TRIAL,physics_samples=len(session.samples),
            rgbd_frames=len(session.model_manifest),auxiliary_frames=len(aux),command_ticks=len(commands),
            setup_checked=True,setup_admitted=True,physical_stop=None,navigation_qualified=False,
            fixed_command_replay=True,model_inference=False,robot_visualization_enabled=True)
        write_json(directory/'result.json',result)
        return result
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
                write_json(directory/'context_decisions.json',[]);write_json(directory/'command_tape.json',commands)
                write_json(directory/'native_guard_rows.json',session.guard_rows);write_json(directory/'friction_checks.json',friction)
                write_json(directory/'auxiliary_camera_audit.json',aux)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


def audit(collection):
    directory=OUTPUT/'sensor_prefix';matches={}
    for name in ('physics_trace.npz','ideal_sensor_samples.npz','fast_gyro_samples.npz','policy_histories.npz','fast_gyro_histories.npz'):
        with np.load(directory/name,allow_pickle=False) as current,np.load(INPUT/CASE/name,allow_pickle=False) as old:
            assert set(current.files)==set(old.files)
            for key in current.files:
                a=current[key];b=old[key][:len(a)] if a.ndim else old[key]
                assert np.array_equal(a,b),(name,key,'physical/public prefix changed')
            matches[name]=fingerprint({k:current[k] for k in current.files})
    auxiliary=read_json(directory,'auxiliary_camera_audit.json');rows=read_json(INPUT/CASE,'context_decisions.json')
    target=read_json(GEOMETRY,'projection.json');patches=RetainedFloorPatches();coverage_rows=[]
    assert len(auxiliary)==COMMAND_TICKS+1 and collection['physics_samples']==750+50*COMMAND_TICKS
    for i,row in enumerate(auxiliary):
        assert row['frame']==i and row['physical_sample_index']==749+50*i and row['measured_ns']==1_500_000_000+100_000_000*i
        with np.load(directory/f'auxiliary_depth_{i:04d}.npz',allow_pickle=False) as z:
            native=z['native_optical_depth_m'];depth=z['depth_m'];valid=z['valid'];seg=z['diagnostic_segmentation']
            import hashlib
            assert hashlib.sha256(native.tobytes()).hexdigest()==row['native_depth_sha256']
            assert hashlib.sha256(seg.tobytes()).hexdigest()==row['diagnostic_segmentation_sha256']
            mask=np.isfinite(native)&(native>=.2)&(native<=5.)
            assert np.array_equal(mask,valid) and np.array_equal(depth,np.where(mask,native,np.float32(0.)))
            receipt=rows[i]['decision']['memory_receipt'];pose=rows[i]['decision']['evidence']['current_pose']
            B=np.asarray(receipt['map_from_initial']);R=B@np.asarray(pose['rotation_initial_body_from_current_body'])
            p=B@np.asarray(pose['position_initial_body_m']);Q,q=reference_pose(R,p)
            patches.append(depth,valid,Q,q,receipt['floor_height_map_m'],dict(frame=i,measured_ns=row['measured_ns'],
                depth_sha256=row['native_depth_sha256'],rgb_sha256=row['rgb_sha256']))
        covered=patches.coverage([c['centre_map_xy_m'] for c in target['candidates']])
        coverage_rows.append(dict(frame=i,candidates=[dict(action=c['action'],**w) for c,w in zip(target['candidates'],covered,strict=True)]))
    new_rgb=read_json(directory,'camera_audit.json');old_rgb=read_json(INPUT/CASE,'camera_audit.json')
    return dict(exact_physical_and_public_prefix_sha256=matches,auxiliary_frames_reconstructed=len(auxiliary),
        primary_rgb_exact_by_frame=[a['rgb_sha256']==b['rgb_sha256'] for a,b in zip(new_rgb,old_rgb[:len(new_rgb)],strict=True)],
        robot_pixels_by_frame=[r['robot_pixels'] for r in auxiliary],coverage_by_frame=coverage_rows,
        primary_rgb_used_by_new_observer=False,original_observed_poses_used_retrospectively=True,
        terminal_foot_regions_used_retrospectively=True,segmentation_used_for_floor_coverage=False,
        model_input_evaluated=False,controller_evaluated=False,navigation_qualified=False)


def main():
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    validate_root(OUTPUT,must_exist=False)
    verify_artifacts(INPUT,{'result.json':INPUT_SHA});prior=read_json(INPUT,'result.json')
    input_ids={'result.json':INPUT_SHA,**prior['artifact_sha256']};verify_artifacts(INPUT,input_ids)
    verify_artifacts(GEOMETRY,{'result.json':GEOMETRY_SHA});geometry=read_json(GEOMETRY,'result.json')
    assert geometry['status']=='AUXILIARY_TILTED_DEPTH_GEOMETRY_COMPLETE'
    geometry_ids={'result.json':GEOMETRY_SHA,**geometry['artifact_sha256']};verify_artifacts(GEOMETRY,geometry_ids)
    seeds=('scripts/capture_go2_auxiliary_tilted_depth_prefix_v1.py','lewm/tests/test_visible_robot_sensor_scope_development.py')
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,seed_paths=seeds,planned_trials=[TRIAL],workers=1,storage_bytes=1024**3)
    for name,sha in geometry['source_sha256'].items():
        if name in launch['source_sha256']:assert launch['source_sha256'][name]==sha
    sources=discover_sources((PROTOCOL,*seeds),launch['source_sha256']|geometry['source_sha256'])
    tape=read_json(INPUT/CASE,'command_tape.json')[:COMMAND_TICKS]
    assert len(tape)==COMMAND_TICKS and all(t['requested_command'][:2]==[0.,0.] for t in tape)
    launch.update(source_sha256=sources,input_artifact_sha256=input_ids,geometry_artifact_sha256=geometry_ids,
        exact_command_prefix=tape,auxiliary_body_from_optical=body_from_optical().tolist(),
        robot_visualization_enabled=True,model_inference=False,fixed_command_replay=True,
        native_execution=True,sensor_characterization_only=True,planned_command_ticks=COMMAND_TICKS)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('AUXILIARY_TILTED_DEPTH_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);start=time.perf_counter()
    try:
        result=collect(launch,tape);report=audit(result);write_json(OUTPUT/'raw_audit.json',report)
        names=['sensor_prefix/'+n for n in artifacts('',result)]
        names+=['sensor_prefix/auxiliary_camera_audit.json']
        names+=[f'sensor_prefix/auxiliary_{kind}_{i:04d}.{suffix}' for i in range(COMMAND_TICKS+1) for kind,suffix in (('depth','npz'),('rgb','png'))]
        ids={n:digest(OUTPUT/n) for n in ['launch.json','raw_audit.json',*names]};verify_artifacts(OUTPUT,ids)
        verify(launch);verify_artifacts(INPUT,input_ids);verify_artifacts(GEOMETRY,geometry_ids)
        write_json(OUTPUT/'result.json',dict(status='AUXILIARY_TILTED_DEPTH_PREFIX_COMPLETE',source_sha256=sources,
            artifact_sha256=ids,collection=result,raw_audit_pass=True,wall_s=time.perf_counter()-start,
            hardware_after=hardware(),robot_visualization_enabled=True,model_inference=False,
            fixed_command_replay=True,controller_evaluated=False,hardware_mount_validated=False,
            realistic_acquisition_latency_validated=False,navigation_qualified=False,goal_achieved=False))
        print('AUXILIARY_TILTED_DEPTH_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUXILIARY_DEPTH_PREFIX_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
