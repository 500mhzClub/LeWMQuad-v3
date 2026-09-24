"""Fresh multi-reference + persistent-intent simulated return assay."""
from dataclasses import asdict
import json
import shutil
import time

import cv2

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.room_return_pulse_development import STAGES
from lewm.intent_room_return_development import IntentRoomReturn
from scripts.fixed_nominal_pulse_table_development import load_fixed_table,FITTING,AUDIT_SHA256
from lewm.intent_room_return_scene_development import specification,TRIALS
from lewm.support_friction_challenge_development import native_friction
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.intent_room_return_session_development import IntentRoomReturnSession
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.fresh_maze_session_development import admit_setup
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts

PREVIOUS=ROOT/'.generated/go2_coupled_room_return_v1_attempt_001'
OUTPUT=BASE/'go2_intent_room_return_v1_attempt_001'
PROTOCOL='docs/go2_intent_room_return_v1_2026-09-06.md'
IDENTITIES={'launch.json':'be4267ab90f122e18ca8ef8f260cacc2150cdb6aed57f568b6170863c41e0fbf',
    'result.json':'b5ac72bf6ad35df56d99f0dc9be00ae19208d091e805aeaea4501e8d0152cf98',
    'raw_return_audit.json':'2d25a7ce77d4c7c5cf3688f50c6a5f17d872243b10d4bf613554ebca6c97c099'}


def preflight():
    ids={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(ids)
    old=read_json(PREVIOUS,'launch.json'); verify(old); result=read_json(PREVIOUS,'result.json')
    inputs=old['input_sha256']|ids|{str((PREVIOUS/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    model=load_fixed_table()
    inputs|={str((FITTING/n).relative_to(ROOT)):digest(FITTING/n) for n in
             ('raw_pulse_audit.json','nominal_a_pulse_evaluation.json','nominal_b_pulse_evaluation.json')}
    witness=ROOT/'.generated/go2_coupled_pulse_rollout_diagnostic_v1_attempt_001/result.json'
    inputs[str(witness.relative_to(ROOT))]='500e0d973ce4b6d2f119af888557af0780dab84475b323f45df87ed275a4af47'
    replay=ROOT/'.generated/go2_multi_reference_recorded_replay_v1_attempt_001'
    replay_ids={str((replay/n).relative_to(ROOT)):h for n,h in {
        'launch.json':'7ed32677c9f90871294f020aee2f3d8b8bcc04182aaedabaa224da82fe1a2dce',
        'result.json':'efd4aa06967d78988e5b9f94a621ced26122c55ca75c0d2a27e02e819d519cb8'}.items()}
    verify_bindings(replay_ids)
    replay_launch=read_json(replay,'launch.json')
    verify_bindings(replay_launch['source_sha256']|replay_launch['input_sha256'])
    inputs|=replay_ids
    sources=discover_sources((PROTOCOL,'scripts/run_go2_intent_room_return_v1.py',
        'scripts/audit_go2_intent_room_return_v1.py',
        'lewm/tests/test_room_return_pulse_development.py',
        'lewm/tests/test_continuous_pulse_execution_development.py',
        'lewm/tests/test_coupled_pulse_rollout_development.py',
        'lewm/tests/test_coupled_pulse_feedback_development.py',
        'lewm/tests/test_multi_reference_return_development.py',
        'lewm/tests/test_navigation_artifact_root_development.py',
        'lewm/tests/test_intent_room_return_physical_development.py'),replay_launch['source_sha256'])
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch|=dict(source_sha256=sources,input_sha256=inputs,protocol=PROTOCOL,controller_stages=STAGES,
        conditions={c:specification(c) for c in TRIALS},output_root=str(OUTPUT),minimum_free_bytes=40*1024**3,
        planned_storage_bytes=15*1024**3,minimum_trial_start_free_bytes=45*1024**3,
        fixed_empirical_pulse_table=asdict(model),model_fitting_episodes=['nominal_a','nominal_b'],
        model_fitting_audit_sha256=AUDIT_SHA256,online_model_adaptation=False,
        maximum_control_ticks=3600,maximum_terminal_zero_ticks=10,model='gyro',
        controlled_level_floor_assumption=True,hidden_robot_ideal_camera_assumption=True,
        physics_paused_during_compute=True,real_time_qualified=False,navigation_qualified=False)
    verify(launch); validate_root(OUTPUT,must_exist=False); return launch


def collect(condition,definition):
    directory=OUTPUT/condition; directory.mkdir(); spec=specification(condition); write_json(directory/'specification.json',spec)
    session=None; decisions=[]; tape=[]; friction=[]; physical_stop=None; final_control=None
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=IntentRoomReturnSession(spec,directory); session.install_contact_identity(); build=session.ctx.build
        gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(directory/'actuator_identity.json',gains)
        write_json(directory/'floor_roles.json',dict(physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle',**native_friction(build,spec['friction_mu'])))
        controller=IntentRoomReturn(spec['turn_sign'],load_fixed_table())
        def execute(command,phase,role):
            session.phase=phase
            item=dict(tick=len(tape),requested_command=command,phase=phase,role=role,
                pre_sample_index=len(session.samples)-1,post_sample_index=None,completed=False)
            tape.append(item)
            try: session.command_tick(command); item['completed']=True
            finally: item['post_sample_index']=len(session.samples)-1
        try:
            session.settle_recorded(); session.capture_current(); admit_setup(session,definition)
            for tick in range(3601):
                friction.append(dict(stage='before_decision',tick=tick,**native_friction(build,spec['friction_mu'])))
                start=time.perf_counter_ns(); p,d,f,now=session.sensor_packets()
                free=shutil.disk_usage(BASE.parent).free
                if free<40*1024**3:controller.runtime.executor.fail('STORAGE_RESERVE_STOP',now_ns=now)
                decision=controller.observe(p,d,f,now_ns=now); evidence=decision['evidence']
                decisions.append(dict(tick=tick,observation_index=len(session.model_manifest)-1,evidence=evidence,decision=decision,
                    resource_free_bytes=free,
                    observation_and_control_wall_ms=(time.perf_counter_ns()-start)/1e6,
                    pre_sample_index=len(session.samples)-1))
                if tick%50==0 or decision['terminal']:
                    local=decision['execution']['local_decision'];diag=local['diagnostic'] if local else {}
                    print('INTENT_SERVO',condition,tick,decision['stage'],decision['terminal'],
                          dict(completed_stages=decision['completed_stages'],reason=decision['reason'],
                               local={k:v for k,v in diag.items() if k!='rollout'}),flush=True)
                if decision['terminal']:
                    final_control=decision
                    if decision['terminal']!='ROOM_RETURN_CANDIDATE':
                        for _ in range(10): execute([0.,0.,0.],9,'terminal_zero_command_drain')
                        session.capture_current()
                    break
                local=decision['execution']['local_decision']
                execute(decision['requested_command'],local['phase'] if local else 2,'visual_feedback')
            if final_control is None: raise ValueError('controller must be terminal by fixed maximum')
        except PhysicalStop as error:
            physical_stop=str(error)
            controller.finish_physical_stop(physical_stop,now_ns=int(round(session.samples[-1]['timestamp_s']*1e9)))
        friction.append(dict(stage='terminal',**native_friction(build,spec['friction_mu'])))
        terminal=read_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=gains['effective']: raise ValueError('gait gains changed')
        write_json(directory/'terminal_actuator_gains.json',terminal)
        write_json(directory/'terminal_native_robot_geometry.json',capture_native_robot_geometry(build.robot))
        write_json(directory/'terminal_environment_identity.json',appearance_environment_identity(session))
        result=dict(status='SERVO_TERMINAL_AUDIT_REQUIRED',condition=condition,physical_stop=physical_stop,
            controller_terminal=final_control,command_ticks=len(tape),completed_ticks=sum(t['completed'] for t in tape),
            physics_samples=len(session.samples),rgbd_frames=len(session.model_manifest),decisions=len(decisions),
            native_state_used_for_commands=False,controlled_level_floor=True,ideal_camera=True,navigation_qualified=False)
        write_json(directory/'return_memory.json',controller.snapshot())
        write_json(directory/'result.json',result); return result
    finally:
        if session is not None:
            try:
                session.persist(directory); session.persist_observations(directory)
                write_json(directory/'servo_decisions.json',decisions); write_json(directory/'command_tape.json',tape)
                write_json(directory/'native_guard_rows.json',session.guard_rows); write_json(directory/'friction_checks.json',friction)
            finally: session.ctx.build.scene.destroy()
        shutdown_genesis()


def artifacts(result):
    names=['specification.json','actuator_identity.json','floor_roles.json','terminal_actuator_gains.json',
        'terminal_native_robot_geometry.json','terminal_environment_identity.json','result.json',
        'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json',
        'ideal_sensor_samples.npz','policy_histories.npz','policy_observations.json','camera_audit.json',
        'depth_observations.json','depth_camera_audit.json','fast_gyro_samples.npz','fast_gyro_histories.npz',
        'floor_visual_collision_identity.json','static_objects.json','startup_native_robot_geometry.json','setup_checks.json',
        'servo_decisions.json','command_tape.json','native_guard_rows.json','friction_checks.json',
        'return_memory.json','visual_meshes/ground_visual.ply',
        'visual_meshes/east_visual.ply','visual_meshes/west_visual.ply','visual_meshes/north_visual.ply','visual_meshes/south_visual.ply']
    return names+[f'{prefix}_{i:04d}.{suffix}' for i in range(result['rgbd_frames'])
        for prefix,suffix in (('rgb','png'),('depth','npz'),('native_depth','npz'))]


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive bounded visual servo')
    validate_root(OUTPUT,must_exist=False)
    cv2.setNumThreads(1); launch=preflight()
    if shutil.disk_usage(BASE.parent).free<launch['minimum_free_bytes']+launch['planned_storage_bytes']:
        raise ValueError('whole three-trial storage budget plus reserve required before launch')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch); results={}
    try:
        for c in TRIALS:
            verify(launch)
            if shutil.disk_usage(BASE.parent).free<launch['minimum_trial_start_free_bytes']: raise ValueError('next trial storage budget plus reserve exhausted')
            results[c]=collect(c,launch['source_sha256'][PROTOCOL])
        verify(launch); names=[c+'/'+n for c,r in results.items() for n in artifacts(r)]
        present=[n for n in names if (OUTPUT/n).is_file()]
        verify_artifacts(OUTPUT,{n:digest(OUTPUT/n) for n in present})
        write_json(OUTPUT/'result.json',dict(status='ROOM_RETURN_PULSE_COLLECTION_TERMINAL',conditions=results,
            absent_expected_artifacts=sorted(set(names)-set(present)),artifact_sha256={n:digest(OUTPUT/n) for n in present},
            controlled_level_floor=True,ideal_camera=True,real_time_qualified=False,navigation_qualified=False,goal_achieved=False))
        print(json.dumps(results),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_VISUAL_SERVO_INFRASTRUCTURE_FAILURE',reason=repr(error),completed_conditions=results)); raise


if __name__=='__main__': main()
