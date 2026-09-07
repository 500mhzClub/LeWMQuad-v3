"""Zero-step native robot/collision identity preflight for fresh motion collection."""
import math
import json
import hashlib
from dataclasses import asdict

from lewm_genesis.bounded_scene_builder_development import build_scene_from_pack as reference_build
from lewm_genesis.rgbd_motion_scene_development import build_scene_from_pack
from lewm_genesis.go2_adapter import resolve_go2_urdf
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from lewm_genesis.scene_loader import (ScenePack,StaticObject,RobotSpec,PhysicsRandomization,VisualRandomization,
    MaterialOverride,LightingSpec,load_platform_manifest,camera_mount_from_platform,physics_timing_from_platform,
    DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER)
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_appearance_information_meshset_development_v2 import OUTPUT as PREVIOUS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_rgbd_motion_scene_preflight_v1_attempt_001'
PROTOCOL='docs/go2_rgbd_motion_scene_preflight_v1_2026-09-06.md'
SEEDS=('scripts/probe_go2_rgbd_motion_scene_development_v1.py',PROTOCOL,
    'lewm/tests/test_rgbd_motion_scene_development.py','lewm/tests/test_rgbd_inertial_fusion_development.py')
IDENTITIES={'launch.json':'4e7b4bf77c3449d77bae09d7426a40e3b320409bc937afc2eace475f5ad5efb7',
    'result.json':'b0689df36aaf202058cd5d85ba88c0cee34391def6496b3259d2e72072e43b4f',
    'raw_artifact_audit.json':'4b8315e32a51eb79bdd632f083ea6535c9bac7cfe432d46ef07145650c83edb4'}
ARMS=('neutral','repeated','distinctive')
APPEARANCE_SEED=2026090607


def pack():
    platform=load_platform_manifest(ROOT/'config/go2_platform_manifest.yaml'); yaw=.27
    objects=[StaticObject(object_id=name,kind='wall',center_xyz_m=pos,size_xyz_m=size,yaw_rad=angle,material_id='NEUTRAL_WALL')
        for name,pos,size,angle in (
            ('front',(3.04,0,.3),(.08,6.16,.6),0.),('back',(-3.04,0,.3),(.08,6.16,.6),0.),
            ('left',(0,3.04,.3),(6.,.08,.6),0.),('right',(0,-3.04,.3),(6.,.08,.6),0.),
            ('offset_partition',(1.2,1.,.3),(.08,1.8,.6),.27))]
    return ScenePack(scene_id='fresh-rgbd-motion-scene-v1',family='DEVELOPMENT_MOTION_COLLECTION',
        split='DEVELOPMENT_UNASSIGNED',difficulty_tier='NATIVE_INTERFACE_PREFLIGHT',
        manifest_sha256=hashlib.sha256(json.dumps(dict(objects=[asdict(o) for o in objects],spawn=[-.25,-.2,.375],yaw=yaw),sort_keys=True).encode()).hexdigest(),
        physics_seed=2026090606,topology_seed=2026090606,visual_seed=APPEARANCE_SEED,
        world_bounds_xy_m=((-4.,-4.),(4.,4.)),static_objects=tuple(objects),
        robot=RobotSpec(urdf_path=resolve_go2_urdf(platform,ROOT),spawn_xyz_m=(-.25,-.2,.375),
            spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2)),foot_links_in_lewm_order=DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER),
        camera=camera_mount_from_platform(platform),timing=physics_timing_from_platform(platform),
        camera_constraints={'min_camera_clearance_m':.10,'min_wall_thickness_m':.08},source_dir=ROOT,
        visual_randomization=VisualRandomization(material_overrides=(MaterialOverride('NEUTRAL_FLOOR',(.5,.5,.5,1.)),
            MaterialOverride('NEUTRAL_WALL',(.35,.35,.35,1.))),
            lighting=LightingSpec(direction=(0.,0.,-1.),diffuse_rgb=(.8,.8,.8),specular_rgb=(.2,.2,.2),ambient_rgb=(.25,.25,.25))),
        physics_randomization=PhysicsRandomization(1.,0.,1.,0.))


def preflight():
    bindings={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(bindings)
    old=read_json(PREVIOUS,'launch.json');verify(old);result=read_json(PREVIOUS,'result.json')
    inputs=old['input_sha256']|bindings|{str((PREVIOUS/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources(SEEDS,old['source_sha256'])
    launch=old|dict(source_sha256=sources,input_sha256=inputs,
        scope='zero-step native scene preflight; no rendering, policy, physical motion, navigation or calibration')
    verify(launch);return launch


def physical_identity(build, *, reference=False):
    # Entity/link indices may shift when visual-only entities are separated;
    # compare full physical shape, local link name, world pose, friction/solver.
    clean=lambda rows:[{k:v for k,v in r.items() if k not in ('geom_id','link_id')} for r in rows]
    physical=([build.collision_floor]+[e for e in build.scene.entities if e.name in {o.object_id for o in build.pack.static_objects}]
              if reference else build.physical_environment)
    environment=clean([r for e in physical for r in capture_native_robot_geometry(e)])
    return dict(environment=environment,robot=clean(capture_native_robot_geometry(build.robot)),
        physics_steps=int(build.scene.t))


def main():
    if OUTPUT.exists():raise ValueError('fresh fixed native preflight only')
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    initialize_genesis(backend='cpu',seed=2026090606,logging_level='warning')
    try:
        definition=pack(); reference=reference_build(definition,n_envs=1,backend='cpu',render_robot=False,apply_textures=False)
        try: baseline=physical_identity(reference,reference=True)
        finally: reference.scene.destroy()
        write_json(OUTPUT/'reference_identity.json',baseline);names=['reference_identity.json'];identities={};robot_identities={}
        for arm in ARMS:
            directory=OUTPUT/arm;directory.mkdir()
            build=build_scene_from_pack(definition,output=directory,appearance_arm=arm,appearance_seed=APPEARANCE_SEED)
            try:
                actual=physical_identity(build)
                assert actual==baseline and actual['physics_steps']==0
                assert len(actual['environment'])==6 and len(actual['robot'])>0
                native=build.native_environment_identity
                assert native['robot_present'] and native['physics_stepped'] is False
                identities[arm]=native
                assert identities[arm]==identities[ARMS[0]]
                robot_identities[arm]=build.native_robot_geometry
                assert robot_identities[arm]==robot_identities[ARMS[0]]
                write_json(directory/'physical_identity.json',actual)
                write_json(directory/'visual_identity.json',native)
                write_json(directory/'robot_native_identity.json',build.native_robot_geometry)
                names.extend(f'{arm}/{name}' for name in ('physical_identity.json','visual_identity.json','robot_native_identity.json'))
                names.extend(f'{arm}/{name}_visual.ply' for name in ['ground',*[o.object_id for o in definition.static_objects]])
                print('NATIVE_ROBOT_SCENE_PASS '+arm,flush=True)
            finally:build.scene.destroy()
        verify(launch)
        result=dict(status='FRESH_ROBOT_APPEARANCE_SCENE_NATIVE_PREFLIGHT_PASS',
            robot_collision_geometries=len(baseline['robot']),environment_collision_geometries=len(baseline['environment']),
            all_physical_geometry_identical_to_reference=True,all_arm_native_visual_geometry_identical=True,
            physics_steps=0,rendered_frames=0,gait_gains_validated=False,physical_execution_validated=False,
            navigation_qualified=False,artifact_sha256={n:digest(OUTPUT/n) for n in names})
        write_json(OUTPUT/'result.json',result);print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_NATIVE_SCENE_PREFLIGHT_FAILURE',error=repr(error)));raise
    finally:shutdown_genesis()


if __name__=='__main__':main()
