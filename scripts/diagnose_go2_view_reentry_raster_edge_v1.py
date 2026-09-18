"""Bound completed visibility failures to measured depth and projected edges."""
import numpy as np
from lewm.physical_first_surface_depth_development import expected_optical_depth, evaluate_visibility
from lewm.raster_edge_quantization_diagnosis_development import projected_edge_witnesses
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.run_go2_view_reentry_maze_pilot_v1 import OUTPUT as INPUT, CASE
from scripts.read_go2_view_reentry_maze_pilot_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_view_reentry_raster_edge_diagnosis_v1_attempt_001'
PROTOCOL = 'docs/go2_view_reentry_raster_edge_diagnosis_v1_2026-09-08.md'
INPUT_SHA = '0f40eb01e5d5feaf004d0c0e98a9b6d712791dcd676b6013ac965fbb1603ffb8'
READOUT_SHA = '7f3105a1864b21f99f24726350d7ffa1636f2ed8ef0f8ce72d849fc29c13fe76'


def diagnose():
    directory = INPUT/CASE[0]; audit = read_json(INPUT, CASE[0]+'_audit.json')
    cameras = read_json(directory, 'camera_audit.json'); spec = read_json(directory, 'specification.json')
    failed = [i for i, r in enumerate(audit['footprint_checks']) if not r['original_strict_score']['passes_sampled_physical_visibility']]
    assert failed == [909] and audit['strict_physical_visibility_pass'] is False
    reports = []
    for frame in failed:
        with np.load(directory/f'native_depth_{frame:04d}.npz', allow_pickle=False) as archive:
            assert archive.files == ['optical_depth_m']; native = archive['optical_depth_m']
        transform = cameras[frame]['world_from_optical']; boxes = spec['geometry']['wall_boxes']
        strict = evaluate_visibility(native, boxes, transform, render_near_m=.005)
        assert strict == audit['footprint_checks'][frame]['original_strict_score']
        precision = read_json(directory, f'raster_{frame:04d}.json')['precision']
        ref = expected_optical_depth(boxes, transform, stride=8)
        measured = native[np.ix_(ref['rows'], ref['columns'])]; expected = ref['expected_depth_m']
        domain = np.isfinite(expected)&ref['surface_interior']&(expected<4.98)
        bad = np.argwhere(domain & (~np.isfinite(measured) | (np.abs(measured-expected) > .001)))
        assert len(bad) == 1
        full = expected_optical_depth(boxes, transform, stride=1); pixels = []
        for a, b in bad:
            r, c = int(ref['rows'][a]), int(ref['columns'][b])
            assert 1 <= r < 479 and 1 <= c < 639
            edges = projected_edge_witnesses(boxes, transform, [r, c], precision['subpixel_bits'])
            pixels.append(dict(pixel_row_column=[r, c], native_optical_depth_m=float(measured[a,b]),
                expected_optical_depth_m=float(expected[a,b]), expected_object=ref['object_names'][int(ref['object_index'][a,b])],
                native_3x3_depth_m=native[r-1:r+2,c-1:c+2].tolist(),
                expected_3x3_depth_m=full['expected_depth_m'][r-1:r+2,c-1:c+2].tolist(),
                nearest_projected_edges=edges[:5]))
        reports.append(dict(frame=frame, original_strict_score=strict,
            original_footprint_diagnostic=audit['footprint_checks'][frame],
            captured_raster_precision=precision, failing_sampled_pixels=pixels))
    return dict(failed_frames=failed, frames=reports, original_strict_visibility_pass=False,
        strict_failure_unchanged=True, native_rounding_rule_proven=False,
        hypothetical_endpoint_quantization_only=True, sensor_or_pose_precision_bound_proven=False,
        pixels_or_public_masks_changed=False, policy_filter=False, navigation_qualified=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive raster diagnosis required')
    bound = []
    for root, sha, status in ((INPUT, INPUT_SHA, 'VIEW_REENTRY_MAZE_PILOT_COMPLETE'),
            (READOUT, READOUT_SHA, 'VIEW_REENTRY_MAZE_READOUT_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json')
        assert result['status'] == status
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bound.append((root, ids, result))
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_view_reentry_raster_edge_v1.py',
        'lewm/tests/test_raster_edge_quantization_diagnosis_development.py'), bound[-1][2]['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+128*1024**2:
        raise ValueError('bounded raster readout resources unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        completed_input_bindings={str(p): ids for p, ids, _ in bound}, native_execution=False,
        model_training=False, model_loaded=False, diagnosis_workers=1,
        concurrency_reason='bounded CPU geometry readout alongside separately owned recorded-sensor prefix; no native scene')
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('VIEW_REENTRY_RASTER_DIAGNOSIS_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = diagnose(); verify(launch)
        for root, ids, _ in bound: verify_artifacts(root, ids)
        write_json(OUTPUT/'result.json', dict(status='VIEW_REENTRY_RASTER_EDGE_DIAGNOSIS_COMPLETE', report=report,
            source_sha256=sources, launch_sha256=digest(OUTPUT/'launch.json'),
            native_execution=False, model_training=False, navigation_qualified=False, goal_achieved=False))
        print('VIEW_REENTRY_RASTER_DIAGNOSIS_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_VIEW_REENTRY_RASTER_DIAGNOSIS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
