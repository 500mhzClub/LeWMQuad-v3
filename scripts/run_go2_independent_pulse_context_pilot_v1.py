"""Twelve fixed training-context episodes; tracker-independent simulation excitation."""
import shutil
import time
import cv2
from lewm.independent_pulse_context_development import TRIALS,specification,schedule,decision,WARMUP_TICKS
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.support_friction_challenge_development import native_friction
from lewm.visual_led_motion_development import VisualLedMotion
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.independent_pulse_context_session_development import PulseContextSession
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts

PREVIOUS=BASE/'go2_inner_arrival_room_return_v1_attempt_001'
OUTPUT=BASE/'go2_independent_pulse_context_pilot_v1_attempt_001'
PROTOCOL='docs/go2_independent_pulse_context_pilot_v1_2026-09-06.md'
RESERVE=40*1024**3


def preflight():
    verify_artifacts(PREVIOUS,{'launch.json':'53ec5a2a04831ddbc8ae3932038ce90a5f12c797d38087bb6fb95a3e8778667a'})
    old=read_json(PREVIOUS,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/run_go2_independent_pulse_context_pilot_v1.py',
        'scripts/audit_go2_independent_pulse_context_pilot_v1.py',
        'lewm/tests/test_independent_pulse_context_development.py'),old['source_sha256'])
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch.update(source_sha256=sources,input_sha256=old['input_sha256'],conditions={c:specification(c) for c in TRIALS},
        protocol=PROTOCOL,output_root=str(OUTPUT),maximum_command_ticks=33,minimum_free_bytes=RESERVE,
        planned_storage_bytes=2*1024**3,tracker_required_for_commands=False,model_training=False,
        data_role='train',independent_layouts=1,evaluation_layouts=0,real_time_qualified=False,
        hidden_robot_ideal_camera=True,controlled_level_floor=True,navigation_qualified=False,goal_achieved=False)
    verify(launch);validate_root(OUTPUT,must_exist=False)
    if shutil.disk_usage(BASE.parent).free<RESERVE+launch['planned_storage_bytes']:raise ValueError('pilot budget plus40GiB reserve required')
    return launch


def collect(trial,definition):
    directory=OUTPUT/trial;directory.mkdir();spec=specification(trial);write_json(directory/'specification.json',spec)
    session=None;rows=[];tape=[];friction=[];stop=None;acquisition_stop=None;terminal=None;admitted=False
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=PulseContextSession(spec,directory);session.install_contact_identity();build=session.ctx.build
        gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(directory/'actuator_identity.json',gains)
        write_json(directory/'floor_roles.json',dict(physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle',**native_friction(build,spec['friction_mu'])))
        tracker=VisualLedMotion('gyro',identity=(0,0,0))
        try:
            session.settle_recorded();session.capture_current();admit_context_setup(session,definition);admitted=True
            for tick in range(len(schedule(spec['action_index']))+1):
                free=shutil.disk_usage(BASE.parent).free
                if free<RESERVE:acquisition_stop='STORAGE_RESERVE_STOP';break
                friction.append(dict(stage='before_decision',tick=tick,**native_friction(build,spec['friction_mu'])))
                start=time.perf_counter_ns()
                try:
                    p,d,f,now=session.sensor_packets();selected=decision(spec['action_index'],tick,p)
                except (ValueError,TypeError,KeyError) as error:
                    acquisition_stop='PACKET_CONTRACT_STOP: '+str(error);break
                shadow=tracker.observe(p,d,f,now_ns=now)
                rows.append(dict(tick=tick,observation_index=len(session.model_manifest)-1,
                    pre_sample_index=len(session.samples)-1,decision=selected,shadow=shadow,
                    observation_and_control_wall_ms=(time.perf_counter_ns()-start)/1e6,resource_free_bytes=free))
                if selected['terminal']:terminal='FIXED_CONTEXT_PULSE_COMPLETE';break
                session.phase=selected['phase']
                item=dict(tick=tick,requested_command=selected['requested_command'],phase=selected['phase'],role=selected['role'],
                    pre_sample_index=len(session.samples)-1,post_sample_index=None,completed=False)
                tape.append(item)
                try:session.command_tick(item['requested_command']);item['completed']=True
                finally:item['post_sample_index']=len(session.samples)-1
        except PhysicalStop as error:stop=str(error)
        friction.append(dict(stage='terminal',**native_friction(build,spec['friction_mu'])))
        terminal_gains=read_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains!=gains['effective']:raise ValueError('frozen gait gains changed')
        write_json(directory/'terminal_actuator_gains.json',terminal_gains)
        write_json(directory/'terminal_native_robot_geometry.json',capture_native_robot_geometry(build.robot))
        write_json(directory/'terminal_environment_identity.json',appearance_environment_identity(session))
        result=dict(status='CONTEXT_TERMINAL_AUDIT_REQUIRED',trial=trial,physical_stop=stop,acquisition_stop=acquisition_stop,
            schedule_terminal=terminal,setup_admitted=admitted,setup_checked=(directory/'setup_checks.json').is_file(),
            departure_present=any(r['tick']==WARMUP_TICKS for r in rows),command_ticks=len(tape),completed_ticks=sum(t['completed'] for t in tape),
            physics_samples=len(session.samples),rgbd_frames=len(session.model_manifest),decisions=len(rows),
            tracker_required_for_commands=False,native_state_used_for_commands=False,navigation_qualified=False)
        write_json(directory/'result.json',result)
        print('CONTEXT_PILOT',trial,result,flush=True);return result
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
                write_json(directory/'context_decisions.json',rows);write_json(directory/'command_tape.json',tape)
                write_json(directory/'native_guard_rows.json',session.guard_rows);write_json(directory/'friction_checks.json',friction)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


def artifacts(trial,result):
    names=['specification.json','actuator_identity.json','floor_roles.json','terminal_actuator_gains.json',
        'terminal_native_robot_geometry.json','terminal_environment_identity.json','result.json',
        'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json',
        'ideal_sensor_samples.npz','policy_histories.npz','policy_observations.json','camera_audit.json',
        'depth_observations.json','depth_camera_audit.json','fast_gyro_samples.npz','fast_gyro_histories.npz',
        'floor_visual_collision_identity.json','context_decisions.json','command_tape.json','native_guard_rows.json','friction_checks.json',
        'visual_meshes/ground_visual.ply']
    if result['setup_checked']:names+=['static_objects.json','startup_native_robot_geometry.json','setup_checks.json']
    names+=['visual_meshes/'+b['wall_id']+'_visual.ply' for b in specification(trial)['geometry']['wall_boxes']]
    return names+[f'{prefix}_{i:04d}.{suffix}' for i in range(result['rgbd_frames'])
        for prefix,suffix in (('rgb','png'),('depth','npz'),('native_depth','npz'))]


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive fixed context pilot; no retry/resume')
    cv2.setNumThreads(1);launch=preflight();create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);results={}
    try:
        for c in TRIALS:
            verify(launch)
            if shutil.disk_usage(BASE.parent).free<RESERVE+200*1024**2:raise ValueError('next episode storage reserve')
            results[c]=collect(c,launch['source_sha256'][PROTOCOL])
        verify(launch);names=[c+'/'+n for c,r in results.items() for n in artifacts(c,r)]
        present=[n for n in names if (OUTPUT/n).is_file()];bindings={n:digest(OUTPUT/n) for n in present}
        verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='CONTEXT_COLLECTION_TERMINAL',conditions=results,
            planned_trials=list(TRIALS),absent_expected_artifacts=sorted(set(names)-set(present)),artifact_sha256=bindings,
            independent_layouts=1,data_role='train',evaluation_layouts=0,model_trained=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_INFRASTRUCTURE_FAILURE',reason=repr(error),completed_conditions=list(results)));raise


if __name__=='__main__':main()
