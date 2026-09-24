"""Paired controlled-floor visual feedback trials, not a maze qualification."""
import json
import shutil
import time

import cv2

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.pulse_feedback_servo_development import PulseFeedbackServo,RULES
from lewm.pulse_feedback_scene_development import specification,TRIALS
from lewm.support_friction_challenge_development import native_friction
from lewm.visual_led_motion_development import VisualLedMotion
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.pulse_feedback_session_development import PulseFeedbackSession
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.fresh_maze_session_development import admit_setup
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

PREVIOUS=ROOT/'.generated/go2_command_pulse_response_v1_attempt_001'
OUTPUT=ROOT/'.generated/go2_pulse_feedback_servo_v1_attempt_001'
PROTOCOL='docs/go2_pulse_feedback_servo_v1_2026-09-06.md'
IDENTITIES={'launch.json':'5fcd856e45d7580b4b9a069ca63f71afc86df18cae7b328085b93673fcbdb0de',
    'result.json':'f60e46072e1a33767ed221dae26e39ca382cfb96c07aa9be89023521c591dc9d'}


def preflight():
    ids={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(ids)
    old=read_json(PREVIOUS,'launch.json'); verify(old); result=read_json(PREVIOUS,'result.json')
    inputs=old['input_sha256']|ids|{str((PREVIOUS/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources((PROTOCOL,'scripts/run_go2_pulse_feedback_servo_v1.py',
        'scripts/audit_go2_pulse_feedback_servo_v1.py',
        'lewm/tests/test_pulse_feedback_servo_development.py'),old['source_sha256'])
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch|=dict(source_sha256=sources,input_sha256=inputs,protocol=PROTOCOL,controller_rules=RULES,
        conditions={c:specification(c) for c in TRIALS},minimum_free_bytes=10*1024**3,
        maximum_control_ticks=1000,maximum_terminal_zero_ticks=10,model='gyro',
        controlled_level_floor_assumption=True,hidden_robot_ideal_camera_assumption=True,
        physics_paused_during_compute=True,real_time_qualified=False,navigation_qualified=False)
    verify(launch); return launch


def collect(condition,definition):
    directory=OUTPUT/condition; directory.mkdir(); spec=specification(condition); write_json(directory/'specification.json',spec)
    session=None; decisions=[]; tape=[]; friction=[]; physical_stop=None; final_control=None
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=PulseFeedbackSession(spec,directory); session.install_contact_identity(); build=session.ctx.build
        gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(directory/'actuator_identity.json',gains)
        write_json(directory/'floor_roles.json',dict(physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle',**native_friction(build,spec['friction_mu'])))
        motion=VisualLedMotion('gyro',identity=(0,0,0)); controller=PulseFeedbackServo()
        def execute(command,phase,role):
            session.phase=phase
            item=dict(tick=len(tape),requested_command=command,phase=phase,role=role,
                pre_sample_index=len(session.samples)-1,post_sample_index=None,completed=False)
            tape.append(item)
            try: session.command_tick(command); item['completed']=True
            finally: item['post_sample_index']=len(session.samples)-1
        try:
            session.settle_recorded(); session.capture_current(); admit_setup(session,definition)
            for tick in range(1001):
                friction.append(dict(stage='before_decision',tick=tick,**native_friction(build,spec['friction_mu'])))
                start=time.perf_counter_ns(); p,d,f,now=session.sensor_packets()
                evidence=motion.observe(p,d,f,now_ns=now); decision=controller.step(evidence,now_ns=now)
                decisions.append(dict(tick=tick,observation_index=len(session.model_manifest)-1,evidence=evidence,decision=decision,
                    observation_and_control_wall_ms=(time.perf_counter_ns()-start)/1e6,
                    pre_sample_index=len(session.samples)-1))
                if tick%50==0 or decision['terminal']:
                    print('VISUAL_SERVO',condition,tick,decision['stage'],decision['terminal'],decision['diagnostic'],flush=True)
                if decision['terminal']:
                    final_control=decision
                    if decision['terminal']!='VISUAL_TARGET_SEQUENCE_COMPLETE':
                        for _ in range(10): execute([0.,0.,0.],9,'terminal_zero_command_drain')
                        session.capture_current()
                    break
                execute(decision['requested_command'],decision['phase'],'visual_feedback')
            if final_control is None: raise ValueError('controller must be terminal by fixed maximum')
        except PhysicalStop as error: physical_stop=str(error)
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
        'visual_meshes/ground_visual.ply','visual_meshes/wide_front_visual.ply']
    return names+[f'{prefix}_{i:04d}.{suffix}' for i in range(result['rgbd_frames'])
        for prefix,suffix in (('rgb','png'),('depth','npz'),('native_depth','npz'))]


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive bounded visual servo')
    cv2.setNumThreads(1); launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch); results={}
    try:
        for c in TRIALS:
            verify(launch)
            if shutil.disk_usage(ROOT).free<launch['minimum_free_bytes']: raise ValueError('disk reserve exhausted')
            results[c]=collect(c,launch['source_sha256'][PROTOCOL])
        verify(launch); names=[c+'/'+n for c,r in results.items() for n in artifacts(r)]
        present=[n for n in names if (OUTPUT/n).is_file()]
        write_json(OUTPUT/'result.json',dict(status='PULSE_FEEDBACK_SERVO_COLLECTION_TERMINAL',conditions=results,
            absent_expected_artifacts=sorted(set(names)-set(present)),artifact_sha256={n:digest(OUTPUT/n) for n in present},
            controlled_level_floor=True,ideal_camera=True,real_time_qualified=False,navigation_qualified=False,goal_achieved=False))
        print(json.dumps(results),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_VISUAL_SERVO_INFRASTRUCTURE_FAILURE',reason=repr(error),completed_conditions=results)); raise


if __name__=='__main__': main()
