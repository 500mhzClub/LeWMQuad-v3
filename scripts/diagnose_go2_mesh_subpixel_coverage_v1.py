"""Actual float32 mesh input coverage under hypothetical subpixel snapping."""
import numpy as np
import hashlib
import trimesh
from lewm.mesh_subpixel_coverage_diagnosis_development import diagnose_mesh
from lewm.physical_first_surface_depth_development import evaluate_visibility
from scripts.run_go2_view_reentry_maze_pilot_v1 import OUTPUT as INPUT, CASE
from scripts.diagnose_go2_view_reentry_raster_edge_v1 import OUTPUT as EDGE
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_mesh_subpixel_coverage_v1_attempt_001'
PROTOCOL = 'docs/go2_mesh_subpixel_coverage_diagnosis_v1_2026-09-08.md'
INPUT_SHA = '0f40eb01e5d5feaf004d0c0e98a9b6d712791dcd676b6013ac965fbb1603ffb8'
EDGE_SHA = '73d6077b81430ebe78518634568a2fe4f5295966d551ca258c0c53f0a1f0b9a1'


def diagnose(edge):
    directory = INPUT/CASE[0]; spec = read_json(directory, 'specification.json')
    T = read_json(directory, 'camera_audit.json')[909]['world_from_optical']
    raster = read_json(directory, 'raster_0909.json')
    mesh = trimesh.load(directory/'visual_meshes/wall_union_visual.ply', process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32); faces = np.asarray(mesh.faces)
    vertex_sha = hashlib.sha256(vertices.tobytes()).hexdigest()
    assert vertex_sha == raster['order']['surfaces']['walls']['positions_sha256']
    assert len(vertices) == raster['order']['surfaces']['walls']['vertices']
    with np.load(directory/'native_depth_0909.npz', allow_pickle=False) as z:
        assert z.files == ['optical_depth_m']; native = z['optical_depth_m']
    witness = edge['report']['frames'][0]; pixel = witness['failing_sampled_pixels'][0]['pixel_row_column']
    assert witness['frame'] == 909 and pixel == [260, 428]
    strict = evaluate_visibility(native, spec['geometry']['wall_boxes'], T, render_near_m=.005)
    assert strict == witness['original_strict_score'] and not strict['passes_sampled_physical_visibility']
    report = diagnose_mesh(vertices, faces, T, pixel, subpixel_bits=raster['precision']['subpixel_bits'], near_m=.005)
    value = float(native[tuple(pixel)])
    assert value == witness['failing_sampled_pixels'][0]['native_optical_depth_m']
    return dict(mesh_report=report, native_optical_depth_m=value, native_input_positions_sha256=vertex_sha,
        native_float32_position_array_exact=True, original_strict_score=strict,
        original_strict_failure_unchanged=True, navigation_qualified=False, goal_achieved=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive mesh subpixel coverage diagnosis required')
    bindings = []
    for root, sha, status in ((INPUT, INPUT_SHA, 'VIEW_REENTRY_MAZE_PILOT_COMPLETE'),
            (EDGE, EDGE_SHA, 'VIEW_REENTRY_RASTER_EDGE_DIAGNOSIS_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json')
        assert result['status'] == status
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bindings.append((root, ids, result))
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_mesh_subpixel_coverage_v1.py',
        'lewm/tests/test_mesh_subpixel_coverage_diagnosis_development.py',
        'docs/go2_llvmpipe_subpixel_source_review_2026-09-08.md'), bindings[-1][2]['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+128*1024**2:
        raise ValueError('bounded mesh subpixel coverage resources unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        completed_input_bindings={str(p): ids for p, ids, _ in bindings},
        native_execution=False, model_loaded=False, model_training=False, diagnosis_workers=1,
        concurrency_reason='small CPU geometry readout alongside one separately owned native scene')
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('MESH_SUBPIXEL_COVERAGE_LAUNCHED', digest(OUTPUT/'launch.json'), len(sources), flush=True)
    try:
        report = diagnose(bindings[-1][2]); verify(launch)
        for root, ids, _ in bindings: verify_artifacts(root, ids)
        write_json(OUTPUT/'result.json', dict(status='MESH_SUBPIXEL_COVERAGE_DIAGNOSIS_COMPLETE', report=report,
            launch_sha256=digest(OUTPUT/'launch.json'), source_sha256=sources,
            native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('MESH_SUBPIXEL_COVERAGE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MESH_SUBPIXEL_COVERAGE_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
