"""Paired inner-arrival controller assay; unchanged scene, observer and score."""
from dataclasses import asdict
import json
import shutil
import time

import cv2

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.room_return_pulse_development import STAGES
from lewm.inner_goal_room_return_development import InnerGoalRoomReturn
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

PREVIOUS=BASE/'go2_intent_room_return_v1_attempt_001'
OUTPUT=BASE/'go2_inner_arrival_room_return_v1_attempt_001'
PROTOCOL='docs/go2_inner_arrival_room_return_v1_2026-09-06.md'
IDENTITIES={'launch.json':'7a5c427ca521de367a301376fafd262aeda5f6e16b7ed876f2b40e87ce0b1a91',
    'result.json':'27e2f91eaece8667e48fb98d75ce3ee8cfcc3a1c9d0052a97033cad5acc321d3',
    'raw_return_audit.json':'a350c74d4f5851a7486bf01a8276be7f9eb420a83cd8198ad684159b8385923d'}


def preflight():
    verify_artifacts(PREVIOUS,IDENTITIES)
    old=read_json(PREVIOUS,'launch.json');verify(old)
    result=read_json(PREVIOUS,'result.json')
    if result['absent_expected_artifacts']:raise ValueError('complete paired baseline required')
    external=IDENTITIES|result['artifact_sha256'];verify_artifacts(PREVIOUS,external)
    inputs=dict(old['input_sha256'])
    witness_ids={
        'docs/go2_recorded_inner_arrival_diagnostic_2026-09-06.json':'69cce3bd01527816dca8585571dd6bb4116c46c02259598a662172e48fb39003',
        '.generated/go2_balanced_feature_replay_v1_attempt_001/launch.json':'c33cea0bcd34476680c686b13553964b4659357bf16dda9dfd74e7ac3b756021',
        '.generated/go2_balanced_feature_replay_v1_attempt_001/result.json':'4710a28bab24babe71b6f48a288c9e2f9795937e11638996bbd38e50a3791ed3'}
    verify_bindings(witness_ids);inputs|=witness_ids
    sources=discover_sources((PROTOCOL,'scripts/run_go2_inner_arrival_room_return_v1.py',
        'scripts/audit_go2_inner_arrival_room_return_v1.py',
        'lewm/tests/test_inner_arrival_physical_development.py',
        'lewm/tests/test_inner_goal_pulse_development.py'),old['source_sha256'])
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch|=dict(source_sha256=sources,input_sha256=inputs,protocol=PROTOCOL,controller_stages=STAGES,
        conditions={c:specification(c) for c in TRIALS},output_root=str(OUTPUT),minimum_free_bytes=40*1024**3,
        planned_storage_bytes=15*1024**3,minimum_trial_start_free_bytes=45*1024**3,
        fixed_empirical_pulse_table=asdict(load_fixed_table()),model_fitting_episodes=['nominal_a','nominal_b'],
        model_fitting_audit_sha256=AUDIT_SHA256,online_model_adaptation=False,
        paired_baseline_root=str(PREVIOUS),paired_baseline_artifact_sha256=external,
        paired_factor='consistent_internal_position_arrival_region_0.06_to_0.04_m',
        internal_position_tolerance_m=.04,external_position_tolerance_m=.06,
        maximum_control_ticks=3600,maximum_terminal_zero_ticks=10,model='gyro',
        controlled_level_floor_assumption=True,hidden_robot_ideal_camera_assumption=True,
        physics_paused_during_compute=True,real_time_qualified=False,navigation_qualified=False)
    if json.loads(json.dumps(launch['conditions']))!=old['conditions']:raise ValueError('exact baseline conditions required')
    verify(launch);validate_root(OUTPUT,must_exist=False);return launch


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
        controller=InnerGoalRoomReturn(spec['turn_sign'],load_fixed_table())
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
                    print('INNER_ARRIVAL_SERVO',condition,tick,decision['stage'],decision['terminal'],
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
        verify_artifacts(PREVIOUS,launch['paired_baseline_artifact_sha256'])
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
