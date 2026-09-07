"""One-shot actual zero-step friction/transducer/gait preparation."""
import json
import shutil

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.support_friction_challenge_development import CONDITIONS,specification,schedule,native_friction
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.support_friction_session_development import SupportFrictionSession

OUTPUT=ROOT/'.generated/go2_support_friction_native_preflight_v1_attempt_001'
PREVIOUS=ROOT/'.generated/go2_causal_support_kinematics_development_v1_attempt_001'
PROTOCOL='docs/go2_support_friction_native_preflight_v1_2026-09-06.md'
IDENTITIES={'launch.json':'88b6cdbf38cf20b6d2af3537474fe2e1e1f75b7221e7cbcb10d66b558b8f4132',
 'result.json':'957dbafe351941c4e78f8be12a07778381526e607d99bc1fa52a444d484763f1',
 'derivative_up_audit_launch.json':'968fcc8b51619e8a54a5ee77290c3dc811de72b932460a444115895851f57fb0',
 'derivative_up_audit.json':'98a32092377497091fd9afb9c8bdded3d984b947f73b42e4f32862683f0cb639'}


def preflight():
    ids={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(ids)
    old=read_json(PREVIOUS,'launch.json');verify(old);audit=read_json(PREVIOUS,'derivative_up_audit_launch.json')
    verify_bindings(audit['source_sha256']|audit['input_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_support_friction_native_preflight_v1.py',
        'lewm/tests/test_support_friction_challenge_development.py'),old['source_sha256']|audit['source_sha256'])
    launch=old|dict(source_sha256=sources,input_sha256=old['input_sha256']|audit['input_sha256']|ids,
        protocol=PROTOCOL,specifications={c:specification(c) for c in CONDITIONS},future_schedule=schedule(),
        scope='actual zero-step friction/gait/sensor identity only; no physical collection')
    if shutil.disk_usage(ROOT).free<10*1024**3: raise ValueError('10GiB reserve required')
    verify(launch);return launch


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive zero-step preflight')
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);results={};artifacts=[]
    try:
        for condition in CONDITIONS:
            directory=OUTPUT/condition;directory.mkdir();session=None;spec=specification(condition)
            initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
            try:
                session=SupportFrictionSession(spec,directory);session.install_contact_identity();session.install_sensor_identity()
                b=session.ctx.build;gains=configure_gains(b.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
                assert read_gains(b.robot,session.ctx.runner._leg_dof_idx.tolist())==gains['effective']
                friction=native_friction(b,spec['friction_mu'])
                if int(b.scene.t)!=0 or session.samples or session.load_samples or session.support_rows or session.model_manifest:
                    raise ValueError('zero-step no-observation preflight required')
                row=dict(friction=friction,foot_identity=session.foot_identity,gains=gains,
                    native_robot_geometry=capture_native_robot_geometry(b.robot),
                    physics_steps=0,renders=0,loads=0,support_predictions=0)
                write_json(directory/'native_preflight.json',row);results[condition]=row
                artifacts.extend([condition+'/native_preflight.json',condition+'/visual_meshes/ground_visual.ply',condition+'/visual_meshes/wide_front_visual.ply'])
                print('ZERO_STEP_FRICTION_PREFLIGHT_PASS',condition,friction['requested_pair_coefficient'],flush=True)
            finally:
                if session is not None:session.ctx.build.scene.destroy()
                shutdown_genesis()
        for name in ('ground_visual.ply','wide_front_visual.ply'):
            assert digest(OUTPUT/CONDITIONS[0]/'visual_meshes'/name)==digest(OUTPUT/CONDITIONS[1]/'visual_meshes'/name)
        verify(launch);write_json(OUTPUT/'result.json',dict(status='ZERO_STEP_FRICTION_SENSOR_PREFLIGHT_PASS',
            conditions=CONDITIONS,physics_steps=0,physical_collection_executed=False,navigation_qualified=False,goal_achieved=False,
            artifact_sha256={n:digest(OUTPUT/n) for n in artifacts}))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FRICTION_PREFLIGHT_FAILURE',reason=repr(error),completed_conditions=list(results)));raise


if __name__=='__main__':main()
