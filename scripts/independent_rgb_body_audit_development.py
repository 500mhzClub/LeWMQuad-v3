"""Full raw reconstruction plus prospective per-modality eligibility."""
from lewm.terminal_event_coverage_development import classify_coverage
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.independent_layout_collection_audit_development import audit_condition
from scripts.independent_rgb_body_batch_development import output_root, eligibility
from scripts.startup_raw_sensor_audit_development import read_json, read_npz


def audit_rgb_body_condition(directory, spec, result, definition, *, batch):
    if directory != output_root(batch) / spec['trial']:
        raise ValueError('fresh declared collection only; no predecessor eligibility change')
    report, prefix, window, targets = audit_condition(directory, spec, result, definition)
    cameras = read_json(directory, 'camera_audit.json')
    coverage = classify_coverage(spec, read_npz(directory, 'physics_trace.npz'),
        read_npz(directory, 'native_contacts.npz'), read_json(directory, 'contact_events.json'),
        cameras, read_json(directory, 'command_tape.json'), result, report, prefix, window, targets)
    footprints, rasters = [], []
    for i, camera in enumerate(cameras):
        row = read_json(directory, f'raster_{i:04d}.json')
        assert row['physical_sample_index'] == camera['physical_sample_index']
        assert row['order']['order'] == 'floor_first' and row['order']['roles'] == ['floor', 'walls']
        assert set(row['order']['surfaces']) == {'floor', 'walls'}
        precision = row['precision']; positions = precision['rgb_target_sample_positions']
        assert 1 <= precision['subpixel_bits'] <= 32 and 1 <= precision['depth_target_depth_bits'] <= 64
        assert 1 <= precision['rgb_target_samples'] <= 32 and len(positions) == precision['rgb_target_samples']
        assert all(len(p) == 2 and all(0 <= x <= 1 for x in p) for p in positions)
        if rasters: assert row['order'] == rasters[0]['order'] and precision == rasters[0]['precision']
        rasters.append(row)
        native = read_npz(directory, f'native_depth_{i:04d}.npz')['optical_depth_m']
        score = evaluate_footprint(native, spec['geometry']['wall_boxes'], camera['world_from_optical'], render_near_m=.005)
        assert score['original_strict_score'] == report['depth_checks'][i]['physical_visibility']
        footprints.append(dict(frame=i, score=score))
    return dict(report=report, prefix=prefix, window=window, targets=targets, coverage=coverage,
        raster_readbacks=rasters, footprint_diagnostics=footprints,
        eligibility=eligibility(coverage, window, footprints))
