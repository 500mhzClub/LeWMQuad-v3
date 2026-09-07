"""One fixed paired physical challenge with live shadow load observations."""
import json
import shutil
import time

import cv2

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.support_friction_challenge_development import CONDITIONS,specification,schedule,native_friction
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.fresh_maze_session_development import admit_setup
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.support_friction_session_development import SupportFrictionSession

OUTPUT=ROOT/'.generated/go2_support_friction_collection_v1_attempt_001'
PREVIOUS=ROOT/'.generated/go2_support_friction_native_preflight_v1_attempt_001'
PROTOCOL='docs/go2_support_friction_collection_v1_2026-09-06.md'
IDENTITIES={'launch.json':'e23e076e8e2ea105642d1f7f7764c960b6b200e06a9cf256fb78f319c012f529',
 'result.json':'9ffaa7b39107980371cfb5378e516484829459c8fffbb0bc4418794bdb0d9aae'}


def preflight():
    ids={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(ids)
    old=read_json(PREVIOUS,'launch.json');verify(old);result=read_json(PREVIOUS,'result.json')
    if result['status']!='ZERO_STEP_FRICTION_SENSOR_PREFLIGHT_PASS':raise ValueError('successful native preflight required')
    inputs=old['input_sha256']|ids|{str((PREVIOUS/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources((PROTOCOL,'scripts/run_go2_support_friction_collection_v1.py'),old['source_sha256'])
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch|=dict(source_sha256=sources,input_sha256=inputs,protocol=PROTOCOL,
        conditions={c:specification(c) for c in CONDITIONS},commands=schedule(),physics_execution_planned=True,
        minimum_free_bytes=10*1024**3,scope='fresh matched controlled friction intervention; no navigation or model fitting')
    verify(launch);return launch


def collect(condition,definition):
    directory=OUTPUT/condition;directory.mkdir();spec=specification(condition);write_json(directory/'specification.json',spec)
    session=None;tape=[];friction_rows=[];physical_stop=None;completed=0
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=SupportFrictionSession(spec,directory);session.install_contact_identity();session.install_sensor_identity()
        build=session.ctx.build;gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(directory/'actuator_identity.json',gains);write_json(directory/'acquisition_foot_identity.json',session.foot_identity)
        write_json(directory/'floor_roles.json',dict(physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction_rows.append(dict(stage='before_settle',**native_friction(build,spec['friction_mu'])))
        try:
            session.settle_recorded();session.capture_current();admit_setup(session,definition)
            friction_rows.append(dict(stage='after_settle',**native_friction(build,spec['friction_mu'])))
            for tick,command in enumerate(schedule()):
                friction_rows.append(dict(stage='before_command',tick=tick,**native_friction(build,spec['friction_mu'])))
                session.phase=command['phase'];item=command|dict(tick=tick,pre_sample_index=len(session.samples)-1,
                    post_sample_index=None,completed=False,start_perf_counter_ns=time.perf_counter_ns());tape.append(item)
                try:
                    session.command_tick(command['requested_command']);completed+=1
                    session.capture_current();item['completed']=True
                finally:
                    item['post_sample_index']=len(session.samples)-1;item['end_perf_counter_ns']=time.perf_counter_ns()
                if tick%20==0 or tick==224:
                    print('FRICTION_COLLECTION',condition,tick,session.samples[-1]['timestamp_s'],session.support_failure,flush=True)
        except PhysicalStop as error:physical_stop=str(error)
        friction_rows.append(dict(stage='terminal',**native_friction(build,spec['friction_mu'])))
        terminal=read_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=gains['effective']:raise ValueError('gait actuator gain changed')
        write_json(directory/'terminal_actuator_gains.json',terminal)
        write_json(directory/'terminal_native_robot_geometry.json',capture_native_robot_geometry(build.robot))
        write_json(directory/'terminal_environment_identity.json',appearance_environment_identity(session))
        result=dict(status='PHYSICAL_CHALLENGE_COMPLETE_AUDIT_REQUIRED' if completed==225 and physical_stop is None else 'PHYSICAL_CHALLENGE_INCOMPLETE_AUDIT_REQUIRED',
            condition=condition,completed_ticks=completed,physical_stop=physical_stop,physics_samples=len(session.samples),
            rgbd_frames=len(session.model_manifest),foot_sensor_samples=len(session.load_samples),support_predictions=len(session.support_rows),
            support_failure=session.support_failure,estimator_selects_commands=False,navigation_qualified=False)
        write_json(directory/'result.json',result);return result
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory);session.persist_loads(directory)
                write_json(directory/'support_predictions.json',dict(rows=session.support_rows,failure=session.support_failure,
                    native_pose_loaded_by_consumer=False,ground_labels_loaded_by_consumer=False))
                write_json(directory/'command_tape.json',tape);write_json(directory/'native_guard_rows.json',session.guard_rows)
                write_json(directory/'friction_checks.json',friction_rows)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


def artifacts(result):
    fixed=('specification.json','actuator_identity.json','acquisition_foot_identity.json','floor_roles.json',
        'terminal_actuator_gains.json','terminal_native_robot_geometry.json','terminal_environment_identity.json','result.json',
        'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
        'policy_histories.npz','policy_observations.json','camera_audit.json','depth_observations.json','depth_camera_audit.json',
        'fast_gyro_samples.npz','fast_gyro_histories.npz','floor_visual_collision_identity.json','static_objects.json',
        'startup_native_robot_geometry.json','setup_checks.json','live_foot_sensor.npz','support_predictions.json',
        'command_tape.json','native_guard_rows.json','friction_checks.json',
        'visual_meshes/ground_visual.ply','visual_meshes/wide_front_visual.ply')
    return list(fixed)+[f'{prefix}_{i:04d}.{suffix}' for i in range(result['rgbd_frames'])
        for prefix,suffix in (('rgb','png'),('depth','npz'),('native_depth','npz'))]


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive physical friction challenge')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);results={}
    try:
        for condition in CONDITIONS:
            verify(launch)
            if shutil.disk_usage(ROOT).free<launch['minimum_free_bytes']:raise ValueError('disk reserve exhausted')
            results[condition]=collect(condition,launch['source_sha256'][PROTOCOL])
        verify(launch);names=[c+'/'+n for c,r in results.items() for n in artifacts(r)]
        present=[n for n in names if (OUTPUT/n).is_file()]
        write_json(OUTPUT/'result.json',dict(status='FRICTION_COLLECTION_TERMINAL_RAW_AUDIT_REQUIRED',conditions=results,
            absent_expected_artifacts=sorted(set(names)-set(present)),artifact_sha256={n:digest(OUTPUT/n) for n in present},
            paired_geometry_count=1,paired_seed_count=1,model_fitting=False,navigation_qualified=False,goal_achieved=False))
        print(json.dumps(results),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FRICTION_COLLECTION_INFRASTRUCTURE_FAILURE',reason=repr(error),completed_conditions=results));raise


if __name__=='__main__':main()
