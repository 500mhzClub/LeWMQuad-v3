"""Pixel/timing and fixed-scene contracts; actual renderer evidence is separate."""
from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.marker_beacon_scene_development import CASES, trials
from lewm.rgb_marker_beacon_development import MARKER_ID, MarkerDiscovery, observe_marker
from lewm.tests.test_observed_traversal_controller_development import Stream


def frame(stream, tick, case='positive'):
    p, _, now = stream.frame(tick)
    p['image']['rgb'][:] = 32
    if case not in ('absent', 'occluded'):
        red, blue = [150, 20, 10], [15, 30, 150]
        if case == 'reversed': red, blue = blue, red
        width = 3 if case == 'tiny' else 50
        p['image']['rgb'][150:250, 240:240+width] = red
        if case != 'red_only':
            left = 420 if case == 'separated' else 295
            p['image']['rgb'][150:250, left:left+width] = blue
    return p, now


@pytest.mark.parametrize('case', [*CASES, 'tiny'])
def test_declared_pattern_and_distractors(case):
    p, now = frame(Stream(), 0, case)
    row = observe_marker(p, now_ns=now)
    assert bool(row['detections']) == (case == 'positive')
    assert row['place_identity'] is None and not row['metric_distance_available']
    if row['detections']: assert row['detections'][0]['marker_id'] == MARKER_ID


def test_three_frames_discover_once_and_duplicates_do_not_increment():
    stream = Stream(); tracker = MarkerDiscovery(); rows = []
    for tick in range(5):
        p, now = frame(stream, tick)
        row = tracker.observe(p, now_ns=now); rows.append(row)
        assert tracker.observe(p, now_ns=now) == row
    assert [r['distinct_marker_count'] for r in rows] == [0, 0, 1, 1, 1]
    assert [bool(r['newly_discovered']) for r in rows] == [False, False, True, False, False]
    assert len(rows[-1]['discovered'][MARKER_ID]) == 3
    rows[-1]['discovered'][MARKER_ID].clear()
    assert len(tracker.observe(p, now_ns=now)['discovered'][MARKER_ID]) == 3


@pytest.mark.parametrize('interruption', ['absent', 'gap'])
def test_absence_or_gap_breaks_persistence_without_inventing_evidence(interruption):
    stream = Stream(); tracker = MarkerDiscovery(); counts = []
    for tick in range(6):
        p, now = frame(stream, tick, 'absent' if tick == 2 and interruption == 'absent' else 'positive')
        if tick == 2 and interruption == 'gap': continue
        counts.append(tracker.observe(p, now_ns=now)['distinct_marker_count'])
    assert counts[-1] == 1 and all(c == 0 for c in counts[:-1])


@pytest.mark.parametrize('fault', ['rewrite', 'episode', 'privilege', 'future', 'stale'])
def test_sensor_or_identity_fault_latches(fault):
    stream = Stream(); tracker = MarkerDiscovery(); p, now = frame(stream, 0)
    tracker.observe(p, now_ns=now)
    if fault == 'rewrite': p['image']['rgb'][0, 0, 0] += 1
    if fault == 'episode': p['sensor_state']['identity'] = (0, 0, 1)
    if fault == 'privilege': p['beacon_coordinate'] = [1.5, 0.]
    if fault == 'future': p['image']['available_ns'] += 1
    if fault == 'stale': now -= 1
    with pytest.raises(SensorContractError): tracker.observe(p, now_ns=now)
    p, now = frame(stream, 1)
    with pytest.raises(SensorContractError): tracker.observe(p, now_ns=now)


def test_pattern_copy_is_not_distinguishable_from_a_beacon():
    p, now = frame(Stream(), 0)
    assert len(observe_marker(p, now_ns=now)['detections']) == 1
    # Identical physical/visual copies have the same declared identity, not
    # distinct beacons. The task cannot demand an unobservable hidden label.
    p['image']['rgb'][300:400, 240:345] = p['image']['rgb'][150:250, 240:345]
    row = observe_marker(p, now_ns=now)
    assert len(row['detections']) == 2 and {d['marker_id'] for d in row['detections']} == {MARKER_ID}


def test_scene_population_preserves_all_arena_walls_and_physical_panel_geometry():
    specs = trials(); assert len(specs) == 6
    arena = specs[0]['geometry']['wall_boxes'][:4]
    assert all(s['geometry']['wall_boxes'][:4] == arena for s in specs)
    assert len({s['procedural_seed'] for s in specs}) == 1
    assert [len(s['geometry']['wall_boxes']) for s in specs] == [6, 4, 7, 5, 6, 6]
    for spec in specs:
        panels = [b for b in spec['geometry']['wall_boxes'] if b['wall_id'].endswith('_panel')]
        assert all(b['size_xyz'] == [.08, .25, .50] for b in panels)
        assert len({b['wall_id'] for b in spec['geometry']['wall_boxes']}) == len(spec['geometry']['wall_boxes'])


def static_rows(spec):
    palette = {'NEUTRAL_WALL': [.35, .35, .35], 'landmark_red': [.85, .12, .08],
               'landmark_blue': [.10, .22, .85]}
    return [{'pack_object': {'object_id': b['wall_id'], 'kind': 'wall', 'center_xyz_m': b['centre_xyz'],
                'size_xyz_m': b['size_xyz'], 'yaw_rad': 0., 'material_id': b['material_id'], 'roll_rad': 0., 'pitch_rad': 0.},
             'native_name': b['wall_id'], 'native_collision_boxes': 1, 'native_box_size': b['size_xyz'],
             'native_position': b['centre_xyz'], 'native_quaternion_wxyz': [1., 0., 0., 0.],
             'fixed': True, 'collision_enabled': True, 'surface_rgb': palette[b['material_id']]}
            for b in spec['geometry']['wall_boxes']]


def test_static_audit_accepts_exact_all_case_records():
    from scripts.audit_go2_marker_beacon_development_v1 import check_static_objects
    for spec in trials(): check_static_objects(spec, static_rows(spec))


@pytest.mark.parametrize('fault', ['missing_wall', 'size', 'color', 'position', 'collision', 'rotation'])
def test_static_audit_rejects_missing_or_changed_physical_objects(fault):
    from scripts.audit_go2_marker_beacon_development_v1 import check_static_objects
    spec = trials()[0]; rows = deepcopy(static_rows(spec))
    if fault == 'missing_wall': rows.pop(0)
    if fault == 'size': rows[-1]['native_box_size'][0] = .001
    if fault == 'color': rows[-1]['surface_rgb'] = [.35, .35, .35]
    if fault == 'position': rows[-1]['native_position'][0] = 2.
    if fault == 'collision': rows[-1]['collision_enabled'] = False
    if fault == 'rotation': rows[-1]['native_quaternion_wxyz'] = [0., 0., 0., 1.]
    with pytest.raises(ValueError): check_static_objects(spec, rows)


def test_past_discovery_survives_later_absence_without_new_beacon_count():
    stream = Stream(); tracker = MarkerDiscovery()
    for tick in range(4):
        p, now = frame(stream, tick, 'absent' if tick == 3 else 'positive')
        row = tracker.observe(p, now_ns=now)
    assert row['distinct_marker_count'] == 1 and row['consecutive_observations'] == 0
    assert row['newly_discovered'] == [] and len(row['discovered'][MARKER_ID]) == 3
