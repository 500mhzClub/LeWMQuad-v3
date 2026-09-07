import math

import numpy as np
import pytest

from lewm.active_gyro_scan_development import ActiveGyroScan
from lewm.active_exit_scan_scene_development import scan_scenes
from lewm.tests.test_relative_gyro_turn_development import initialized, append, packet


def test_four_views_and_return_use_one_continuous_gyro_reference():
    buffer = initialized()
    scan = ActiveGyroScan()
    result = scan.begin(packet(buffer, 80), now_ns=1_600_000_000)
    command = result['requested_command']
    for tick in range(1, 301):
        for step in range(80 + (tick - 1) * 5 + 1, 81 + tick * 5):
            append(buffer, step, [0, 0, command[2]])
        p = packet(buffer, 80 + tick * 5)
        result = scan.step(p, now_ns=p['image']['measured_ns'])
        command = result['requested_command']
        assert command[:2] == [0., 0.] and abs(command[2]) <= .35
        if result['status'] == 'COMPLETE':
            break
    assert result['status'] == 'COMPLETE' and tick <= 300
    assert result['completed_target_views'] == 4 and len(scan.views) == 5
    assert [v['view_index'] for v in scan.views] == [0, 1, 2, 3, 4]
    assert abs(result['relative_heading_rad']) <= .08
    assert command == [0, 0, 0]
    assert not result['metric_clearance_qualified'] and not result['translation_compensated']


def test_unresponsive_scan_times_out_without_fake_completed_views():
    buffer = initialized()
    scan = ActiveGyroScan()
    scan.begin(packet(buffer, 80), now_ns=1_600_000_000)
    for tick in range(1, 301):
        for step in range(80 + (tick - 1) * 5 + 1, 81 + tick * 5):
            append(buffer, step, [0, 0, 0])
        p = packet(buffer, 80 + tick * 5)
        result = scan.step(p, now_ns=p['image']['measured_ns'])
    assert result['status'] == 'FAILED_TIMEOUT' and result['requested_command'] == [0, 0, 0]
    assert len(scan.views) == 1


@pytest.mark.parametrize('fault', ['clock', 'rewrite', 'privilege'])
def test_bad_scan_inputs_latch_failure(fault):
    buffer = initialized()
    scan = ActiveGyroScan()
    scan.begin(packet(buffer, 80), now_ns=1_600_000_000)
    for step in range(81, 86):
        append(buffer, step, [0, 0, .2])
    p = packet(buffer, 85)
    if fault == 'rewrite':
        p['sensor_state']['sensed']['gyro']['values'][-6, 0] = .1
    if fault == 'privilege':
        p['world_yaw'] = 0.
    with pytest.raises(ValueError):
        scan.step(p, now_ns=1_700_000_001 if fault == 'clock' else 1_700_000_000)
    assert scan.status == 'FAILED_SENSOR'
    with pytest.raises(ValueError):
        scan.step(p, now_ns=1_700_000_000)


def test_fixed_scan_scene_factorial_geometry_has_no_runtime_route_graph():
    specs = scan_scenes()
    assert len(specs) == 16 and len({s['scene_id'] for s in specs}) == 16
    assert [s['procedural_seed'] for s in specs] == list(range(2026092500, 2026092516))
    for motif in ('dead_end', 'corner', 'tee', 'cross'):
        rows = [s for s in specs if s['motif'] == motif]
        assert {(s['width_m'], s['initial_heading_rad']) for s in rows} == {(.9, -.15), (.9, .15), (1.2, -.15), (1.2, .15)}
        assert {len(s['evaluation_open_directions']) for s in rows} == {dict(dead_end=1, corner=2, tee=3, cross=4)[motif]}
        for s in rows:
            assert 'route_geometries' not in s and 'graph_connections' not in s
            assert s['geometry']['spawn_se2_world'] == [0., 0., s['initial_heading_rad']]
            assert all(w['size_xyz'][2] == .6 for w in s['geometry']['wall_boxes'])
