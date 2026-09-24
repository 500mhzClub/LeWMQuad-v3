"""Fresh matched physical stimuli; RGBD fusion runs in shadow and never acts."""
from dataclasses import asdict
import json
import time

import cv2
import numpy as np

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.rgbd_shadow_motion_development import schedule,ShadowObserver,POINT_HYPOTHESES
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.probe_go2_rgbd_motion_scene_development_v1 import OUTPUT as PREVIOUS,pack,ARMS,APPEARANCE_SEED
from scripts.rgbd_shadow_motion_session_development import ShadowMotionSession,admit_shadow_setup,appearance_environment_identity
from scripts.run_go2_causal_rgb_body_capture_development_v1 import probe_spec
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_observation_turn_session_development import latest_policy
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_rgbd_shadow_motion_development_v1_attempt_001'
PROTOCOL='docs/go2_rgbd_shadow_motion_development_v1_2026-09-06.md'
SEEDS=('scripts/run_go2_rgbd_shadow_motion_development_v1.py',
       'lewm/tests/test_rgbd_shadow_motion_development.py',PROTOCOL)
IDENTITIES={'launch.json':'76fd6be7675427fd4bbb275a5b0de5dba116f613948aff09eb98c36888c86744',
    'result.json':'d2a088a415a59f3cf3b7963d1bc62aa7941a547f9be30017d5711af50011ed30',
    'raw_artifact_audit.json':'cb3af02c4872b5663d6084595b7d10721c739e242650fa9eb859502928007cbf'}


def specification(arm):
    if arm not in ARMS:raise ValueError('fixed declared appearance arm required')
    definition=pack();spec=probe_spec(0)
    walls=[dict(wall_id=o.object_id,centre_xyz=list(o.center_xyz_m),size_xyz=list(o.size_xyz_m),
                yaw_rad=o.yaw_rad,material_id=o.material_id) for o in definition.static_objects]
    spec=spec|dict(scene_id='fresh-rgbd-shadow-motion-v1-'+arm,family='DEVELOPMENT_SHADOW_MOTION',
        procedural_seed=definition.physics_seed,appearance_arm=arm,appearance_seed=APPEARANCE_SEED)
    spec['geometry']=spec['geometry']|dict(wall_boxes=walls,spawn_se2_world=[-.25,-.2,.27])
    return spec


def preflight():
    bindings={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(bindings)
    old=read_json(PREVIOUS,'launch.json');verify(old);result=read_json(PREVIOUS,'result.json')
    inputs=old['input_sha256']|bindings|{str((PREVIOUS/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources(SEEDS,old['source_sha256'])
    launch=old|dict(source_sha256=sources,input_sha256=inputs,specifications={a:specification(a) for a in ARMS},
        motion_schedule=schedule(),terminal_zero_ticks=3,point_hypotheses=asdict(POINT_HYPOTHESES),
        scope='three fresh matched supervised physical tapes; causal fusion/memory shadow only; no navigation qualification')
    verify(launch);return launch


def collect(arm,definition_sha256):
    output=OUTPUT/arm;output.mkdir();spec=specification(arm);write_json(output/'specification.json',spec)
    session=None;shadow=None;rows=[];tape=[];physical_stop=None;completed=0;tail=0
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=ShadowMotionSession(spec,output);session.install_contact_identity();build=session.ctx.build
        gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(output/'actuator_identity.json',gains)
        write_json(output/'floor_roles.json',dict(physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        def observe_current():
            session.capture_current();frame=len(session.model_manifest)-1;policy=latest_policy(session)
            now=policy['sensor_state']['decision_ns'];start=time.perf_counter()
            result=shadow.observe(policy,session.latest_depth,session.fast_buffer.packet(now_ns=now),now_ns=now)
            rows.append(dict(observation_index=frame,shadow=result,observer_wall_ms=1000*(time.perf_counter()-start)))
        try:
            session.settle_recorded();session.capture_current()
            velocity=admit_shadow_setup(session,definition_sha256);shadow=ShadowObserver(velocity);observe_current()
            for index,command in enumerate(schedule()+[[0.,0.,0.]]*3):
                session.phase=1 if index<50 else 2
                item=dict(tick=index,phase=session.phase,requested_command=command,
                    pre_sample_index=len(session.samples)-1,post_sample_index=None,completed=False)
                tape.append(item);start=time.perf_counter_ns();item['start_perf_counter_ns']=start
                try:
                    session.command_tick(command);item['command_finished_perf_counter_ns']=time.perf_counter_ns()
                    if index<50:completed+=1
                    else:tail+=1
                    observe_current();item['completed']=True
                finally:
                    item['post_sample_index']=len(session.samples)-1
                    item['end_perf_counter_ns']=time.perf_counter_ns()
                    item['outer_wall_ms']=(item['end_perf_counter_ns']-start)/1e6
                print(json.dumps(dict(arm=arm,tick=index,time_s=session.samples[-1]['timestamp_s'],
                    shadow_status=rows[-1]['shadow']['status'])),flush=True)
        except PhysicalStop as error:
            physical_stop=str(error)
        terminal=read_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=gains['effective']:raise ValueError('actuator gains changed')
        write_json(output/'terminal_actuator_gains.json',terminal)
        write_json(output/'terminal_native_robot_geometry.json',capture_native_robot_geometry(build.robot))
        write_json(output/'terminal_environment_identity.json',appearance_environment_identity(session))
        result=dict(status='PHYSICAL_TAPE_COMPLETE' if completed==50 and tail==3 and physical_stop is None else 'PHYSICAL_TAPE_INCOMPLETE',
            completed_motion_ticks=completed,completed_zero_tail_ticks=tail,physical_stop=physical_stop,
            shadow_failure=None if shadow is None else shadow.failure,shadow_successful_frames=0 if shadow is None else shadow.successes,
            physics_samples=len(session.samples),rgbd_frames=len(session.model_manifest),
            estimator_selects_commands=False,navigation_qualified=False,uncertainty_model_validated=False)
        write_json(output/'result.json',result);return result
    finally:
        if session is not None:
            try:
                session.persist(output);session.persist_observations(output)
                write_json(output/'shadow_observations.json',rows);write_json(output/'command_tape.json',tape)
                write_json(output/'native_guard_rows.json',session.guard_rows)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


def artifact_names(result):
    fixed=('specification.json','actuator_identity.json','floor_roles.json','terminal_actuator_gains.json',
        'terminal_native_robot_geometry.json','terminal_environment_identity.json','result.json',
        'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json',
        'ideal_sensor_samples.npz','policy_histories.npz','policy_observations.json','camera_audit.json',
        'depth_observations.json','depth_camera_audit.json','fast_gyro_samples.npz','fast_gyro_histories.npz',
        'floor_visual_collision_identity.json','static_objects.json','startup_native_robot_geometry.json','setup_checks.json',
        'shadow_observations.json','command_tape.json','native_guard_rows.json')
    return list(fixed)+[f'{prefix}_{i:04d}.{suffix}' for i in range(result['rgbd_frames'])
        for prefix,suffix in (('rgb','png'),('depth','npz'),('native_depth','npz'))]+[
        f'visual_meshes/{name}_visual.ply' for name in ['ground',*[o.object_id for o in pack().static_objects]]]


def main():
    if OUTPUT.exists():raise ValueError('fresh fixed shadow-motion acquisition only; no retry')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        results={arm:collect(arm,launch['source_sha256'][PROTOCOL]) for arm in ARMS}
        verify(launch);names=[f'{arm}/{n}' for arm,r in results.items() for n in artifact_names(r)]
        present=[n for n in names if (OUTPUT/n).is_file()]
        result=dict(status='SHADOW_MOTION_ACQUISITION_COMPLETE_AUDIT_REQUIRED',arms=results,
            artifact_sha256={n:digest(OUTPUT/n) for n in present},absent_expected_artifacts=sorted(set(names)-set(present)),
            independent_layout_trials=1,estimator_selects_commands=False,navigation_qualified=False)
        write_json(OUTPUT/'result.json',result);print(json.dumps(results),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SHADOW_MOTION_ACQUISITION_FAILURE',error=repr(error)));raise


if __name__=='__main__':main()
