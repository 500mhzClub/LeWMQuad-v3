"""Zero-step current-scene geometry-cache identity inspection; no cache retirement."""
import contextlib
import os
from pathlib import Path
import re
import stat
import time
import cv2
import torch
from lewm.novel_maze_round_trip_scene_development import specification, pack
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from lewm_genesis.visible_robot_union_rgbd_scene_development import build_scene_from_pack
from lewm_genesis.scene_builder import shutdown_genesis
from scripts.scene_geometry_cache_identity_development import capture
from scripts.run_go2_executed_waypoint_maze_pilot_v1 import OUTPUT as INPUT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE/'go2_maze_geometry_cache_identity_v1_attempt_001'
PROTOCOL = 'docs/go2_maze_geometry_cache_identity_v1_2026-09-08.md'
INPUT_LAUNCH_SHA = 'f7fec194358f3821c036df413c360f355cd758ec7d1ca5c1b7466b99ceea9f4e'
CACHE = Path('/home/andrewknowles/.cache/genesis/gsd')


def inventory():
    if CACHE.is_symlink() or CACHE.resolve() != CACHE: raise ValueError('exact ordinary GSD cache root required')
    result = {}
    # Flat metadata only. Protected names are excluded before any stat or access.
    for path in CACHE.iterdir():
        if path.name in ('sealed', 'sealed_test.json') or path.name.startswith('sealed_'): continue
        if not re.fullmatch(r'[0-9a-f]{64}\.gsd', path.name) or path.is_symlink():
            raise ValueError('unexpected non-flat or nonordinary cache entry')
        s = path.stat()
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid(): raise ValueError('owned regular cache metadata required')
        result[path.name] = dict(inode=s.st_ino, byte_count=s.st_size, allocated_bytes=s.st_blocks*512,
            mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, owner_uid=s.st_uid, link_count=s.st_nlink)
    return result


def main():
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if any(os.environ.get(k) is not None for k in ('GS_CACHE_FILE_PATH', 'XDG_CACHE_HOME')):
        raise ValueError('same observed default-cache environment as active native worker required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive cache identity inspection required')
    verify_artifacts(INPUT, {'launch.json': INPUT_LAUNCH_SHA}); old = read_json(INPUT, 'launch.json'); verify(old)
    spec = specification(0); assert old['scene_specification'] == spec
    sources = discover_sources((PROTOCOL, 'scripts/inspect_go2_maze_geometry_cache_v1.py',
        'lewm/tests/test_scene_geometry_cache_identity_development.py'), old['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+64*1024**2:
        raise ValueError('bounded zero-step scene inspection resources unavailable')
    from genesis.utils.mesh import get_gsd_path
    from genesis.utils.misc import get_gsd_cache_dir
    import genesis
    assert Path(get_gsd_cache_dir()) == CACHE
    native_root = Path(genesis.__file__).parent
    native_files = {str(native_root/name): digest(native_root/name) for name in (
        'utils/mesh.py', 'utils/misc.py', 'engine/entities/rigid_entity/rigid_geom.py')}
    launch = old | dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        source_native_launch_sha256=INPUT_LAUNCH_SHA, cache_inspection_source_sha256=native_files,
        hardware=resources, scene_builds=1, physics_steps=0, rgbd_frames=0, high_level_model_loaded=False,
        actuator_policy_loaded=False, cache_root=str(CACHE), cache_environment={'GS_CACHE_FILE_PATH': None, 'XDG_CACHE_HOME': None},
        cache_retirement_authorized=False, scientific_artifact_mutation=False,
        concurrency_reason='one zero-step current-geometry scene alongside independent raw audit; no second navigation collection')
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    started = time.perf_counter(); build = None
    print('MAZE_CACHE_IDENTITY_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        before = inventory(); write_json(OUTPUT/'cache_metadata_before.json', before)
        visual = OUTPUT/'visual_meshes'; visual.mkdir()
        with (OUTPUT/'inspection.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                build = build_scene_from_pack(pack(spec), output=visual, appearance_arm=spec['appearance_arm'],
                    appearance_seed=spec['appearance_seed'], n_envs=1, backend='cpu', show_viewer=False, render_robot=True)
                assert int(build.scene.t) == 0
                def path_for(geom):
                    material = geom._material
                    return get_gsd_path(geom._init_verts, geom._init_faces,
                        material.sdf_cell_size, material.sdf_min_res, material.sdf_max_res)
                report = capture([*build.physical_environment, build.robot, *build.visual_surfaces], CACHE, path_for)
                assert int(build.scene.t) == 0
                write_json(OUTPUT/'scene_identity.json', dict(environment=build.native_environment_identity,
                    robot=build.native_robot_geometry, scene_specification=spec, physics_steps=0))
            finally:
                if build is not None: build.scene.destroy()
                shutdown_genesis()
        after = inventory(); write_json(OUTPUT/'cache_metadata_after.json', after)
        assert all(after.get(k) == v for k, v in before.items()), 'preexisting cache metadata changed'
        new = sorted(set(after)-set(before))
        assert set(new) <= set(report['cache_bindings']), 'unrelated new cache identity'
        for name, binding in report['cache_bindings'].items():
            assert digest(CACHE/name) == binding['sha256'] and (CACHE/name).stat().st_size == binding['byte_count']
        for name, sha in native_files.items(): assert digest(Path(name)) == sha
        verify(launch); verify_artifacts(INPUT, {'launch.json': INPUT_LAUNCH_SHA})
        names = ('launch.json', 'cache_metadata_before.json', 'cache_metadata_after.json', 'inspection.log',
            'scene_identity.json', 'visual_meshes/ground_visual.ply', 'visual_meshes/wall_union_visual.ply')
        ids = {n: digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='MAZE_GEOMETRY_CACHE_IDENTITY_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, new_current_scene_cache_names=new,
            preexisting_cache_metadata_unchanged=True, physics_steps=0, navigation_execution=False,
            high_level_model_loaded=False, actuator_policy_loaded=False, cache_retirement_performed=False,
            wall_s=time.perf_counter()-started, hardware_after=hardware(), goal_achieved=False))
        print('MAZE_CACHE_IDENTITY_COMPLETE', digest(OUTPUT/'result.json'), report['unique_existing_cache_count'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MAZE_CACHE_IDENTITY_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
