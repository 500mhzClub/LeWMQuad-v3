"""Conditional visible-surface ranges for synthetic and recorded boundary cases."""
import numpy as np
from lewm.depth_boundary_counterexamples_development import fixture, PIXEL
from lewm.pixel_visible_surface_intervals_development import visible_intervals, supported_depth
from lewm.physical_first_surface_depth_development import evaluate_visibility
from scripts.run_go2_view_reentry_maze_pilot_v1 import OUTPUT as INPUT, CASE
from scripts.diagnose_go2_view_reentry_raster_edge_v1 import OUTPUT as EDGE
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_visible_surface_intervals_v1_attempt_001'
PROTOCOL = 'docs/go2_visible_surface_intervals_v1_2026-09-08.md'
INPUT_SHA = '0f40eb01e5d5feaf004d0c0e98a9b6d712791dcd676b6013ac965fbb1603ffb8'
EDGE_SHA = '73d6077b81430ebe78518634568a2fe4f5295966d551ca258c0c53f0a1f0b9a1'


def diagnose(edge):
    rows = []
    for near in (False, True):
        T, boxes, depth = fixture(near=near)
        for radius in (1/256, .5):
            r = visible_intervals(boxes, T, PIXEL, radius_pixels=radius)
            background_supported = supported_depth(r, float(depth[PIXEL]), metric_tolerance_m=.001)
            assert not supported_depth(r, 1.5, metric_tolerance_m=.001)
            assert supported_depth(r, .004 if near else .96, metric_tolerance_m=.001)
            assert background_supported is (not near and radius == .5)
            rows.append(dict(case='near_occluder' if near else 'thin_post', interval_report=r,
                synthetic_background_depth_m=float(depth[PIXEL]), background_supported=background_supported,
                fabricated_gap_depth_supported=False))
    directory = INPUT/CASE[0]; spec = read_json(directory, 'specification.json')
    T = read_json(directory, 'camera_audit.json')[909]['world_from_optical']
    with np.load(directory/'native_depth_0909.npz', allow_pickle=False) as z:
        assert z.files == ['optical_depth_m']; native = z['optical_depth_m']
    strict = evaluate_visibility(native, spec['geometry']['wall_boxes'], T, render_near_m=.005)
    witness = edge['report']['frames'][0]
    assert witness['frame'] == 909 and strict == witness['original_strict_score']
    assert strict['passes_sampled_physical_visibility'] is False
    pixel = witness['failing_sampled_pixels'][0]['pixel_row_column']; value = float(native[tuple(pixel)])
    assert pixel == [260, 428] and value == witness['failing_sampled_pixels'][0]['native_optical_depth_m']
    recorded = []
    for radius in (1/256, .5):
        r = visible_intervals(spec['geometry']['wall_boxes'], T, pixel, radius_pixels=radius)
        assert supported_depth(r, value, metric_tolerance_m=.001)
        assert not supported_depth(r, 1.5, metric_tolerance_m=.001)
        recorded.append(dict(interval_report=r, native_optical_depth_m=value,
            native_depth_supported_under_supplied_assumption=True, fabricated_1_5m_gap_supported=False))
    return dict(synthetic_cases=rows, recorded_frame=909, recorded_cases=recorded,
        original_strict_score=strict, original_strict_failure_unchanged=True,
        evaluated_recorded_pixels=1, full_image_or_all_frame_evaluation=False,
        angular_radius_is_assumed_not_a_native_error_bound=True,
        policy_side_uncertainty_implemented=False, floating_point_bound_proven=False,
        native_pixels_masks_or_commands_changed=False, navigation_qualified=False, goal_achieved=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive conditional interval diagnosis required')
    bindings = []
    for root, sha, status in ((INPUT, INPUT_SHA, 'VIEW_REENTRY_MAZE_PILOT_COMPLETE'),
            (EDGE, EDGE_SHA, 'VIEW_REENTRY_RASTER_EDGE_DIAGNOSIS_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json')
        assert result['status'] == status
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bindings.append((root, ids, result))
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_visible_surface_intervals_v1.py',
        'lewm/tests/test_pixel_visible_surface_intervals_development.py',
        'lewm/tests/test_depth_boundary_counterexamples_development.py'), bindings[-1][2]['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+128*1024**2:
        raise ValueError('bounded conditional interval resources unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        completed_input_bindings={str(p): ids for p, ids, _ in bindings},
        native_execution=False, model_loaded=False, model_training=False, diagnosis_workers=1,
        concurrency_reason='small CPU geometry readout alongside one separately owned native scene')
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('VISIBLE_SURFACE_INTERVALS_LAUNCHED', digest(OUTPUT/'launch.json'), len(sources), flush=True)
    try:
        report = diagnose(bindings[-1][2]); verify(launch)
        for root, ids, _ in bindings: verify_artifacts(root, ids)
        write_json(OUTPUT/'result.json', dict(status='VISIBLE_SURFACE_INTERVALS_DIAGNOSIS_COMPLETE', report=report,
            launch_sha256=digest(OUTPUT/'launch.json'), source_sha256=sources,
            native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('VISIBLE_SURFACE_INTERVALS_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_VISIBLE_SURFACE_INTERVALS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
