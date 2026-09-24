"""Synthetic/source-only checks. No native collection or observer adoption."""
from copy import deepcopy
import inspect
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lewm.independent_tracking_challenge_development import (
    SCENES, SUPPORTS, DIRECTIONS, TRIALS, SEGMENTS, MAX_TICKS, MAX_FRAMES,
    MAX_PHYSICS_SAMPLES, SETTLE_SAMPLES, geometry, physical_geometry_identity,
    specification, validate_specification, pack, schedule, decision)
from lewm.independent_tracking_coverage_development import measured_coverage
from lewm.tests.test_independent_pulse_context_development import policy


def test_fixed_complete_population_and_resource_counts():
    assert len(TRIALS) == 8 and len(set(TRIALS)) == 8
    assert MAX_TICKS == 442 and MAX_FRAMES == 443 and MAX_PHYSICS_SAMPLES == 22850
    for scene in SCENES:
        siblings = [specification(t) for t in TRIALS if specification(t)['scene'] == scene]
        assert {(s['condition'], s['direction']) for s in siblings} == {(c, d) for c in SUPPORTS for d in DIRECTIONS}
        for key in ('geometry', 'appearance_seed', 'procedural_seed', 'geometry_identity'):
            assert all(s[key] == siblings[0][key] for s in siblings)
        assert all(s['data_role'] == 'development_challenge' and not s['model_training'] for s in siblings)


def test_geometry_novelty_is_not_just_names_seeds_or_wall_order():
    from lewm.independent_layout_inventory_development import build_inventory
    from lewm.longer_motion_collection_development import specification as predecessor
    old = [l['wall_boxes'] for l in build_inventory()['layouts']]
    old.append(predecessor('fit')['geometry']['wall_boxes'])
    identities = {physical_geometry_identity(w) for w in old}
    new = [physical_geometry_identity(geometry(s)['wall_boxes']) for s in SCENES]
    assert len(set(new)) == 2 and not set(new) & identities
    walls = geometry(SCENES[0])['wall_boxes']
    renamed = list(reversed(deepcopy(walls)))
    for i, w in enumerate(renamed):
        w['wall_id'] = f'new_label_{i}'
        w['material_id'] = 'different_label'
    assert physical_geometry_identity(walls) == physical_geometry_identity(renamed)
    renamed[0]['centre_xyz'][0] += .01
    assert physical_geometry_identity(walls) != physical_geometry_identity(renamed)


@pytest.mark.parametrize('scene', SCENES)
def test_native_builder_compatible_union_geometry_and_floor_domain(scene):
    from lewm_genesis.union_wall_surface_development import wall_union_boundary
    from lewm_genesis.union_wall_rgbd_scene_development import floor_domain
    spec = specification(f'{scene}_nominal_left')
    boundary = wall_union_boundary(spec['geometry']['wall_boxes'])
    assert boundary['union_surface_area_m2'] > 0 and not boundary['native_visibility_qualified']
    p = pack(spec)
    assert floor_domain(p)['minimum_declared_support_margin_m'] > .01
    assert len(p.static_objects) == len(spec['geometry']['wall_boxes'])
    for obj, w in zip(p.static_objects, spec['geometry']['wall_boxes'], strict=True):
        assert obj.center_xyz_m == tuple(w['centre_xyz']) and obj.size_xyz_m == tuple(w['size_xyz'])
    x, y, yaw = spec['geometry']['spawn_se2_world']
    assert p.robot.spawn_xyz_m == (x, y, .375)
    assert p.robot.spawn_quat_wxyz == (math.cos(yaw/2), 0., 0., math.sin(yaw/2))
    assert p.camera.near_m == .005 and p.camera.xyz_body_m == (.326, 0., .043)
    assert p.camera.far_m == 200. and p.physics_randomization.floor_friction_mu == 1.


@pytest.mark.parametrize('fault', ['friction', 'geometry', 'bool', 'extra', 'trial', 'role'])
def test_mutated_specification_rejected_before_scene_creation(fault, tmp_path):
    from scripts.independent_tracking_session_development import IndependentTrackingSession
    s = specification(TRIALS[0])
    if fault == 'friction': s['friction_mu'] = .9
    elif fault == 'geometry': s['geometry']['wall_boxes'][0]['centre_xyz'][0] += .01
    elif fault == 'bool': s['maximum_command_ticks'] = True
    elif fault == 'extra': s['native_pose'] = [0]*7
    elif fault == 'trial': s['trial'] = 'unplanned'
    elif fault == 'role': s['data_role'] = 'train'
    with pytest.raises(ValueError): pack(s)
    with pytest.raises(ValueError): IndependentTrackingSession(s, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_returns_do_not_alias_definition():
    s = specification(TRIALS[0]); s['geometry']['spawn_se2_world'][0] = 500
    assert specification(TRIALS[0])['geometry']['spawn_se2_world'][0] == -.83
    rows = schedule('left'); rows[0]['requested_command'][0] = 12
    assert schedule('left')[0]['requested_command'] == [0., 0., 0.]


def test_sustained_bounded_tape_and_exact_direction_pairing():
    from lewm.command_pulse_response_development import validate_command
    left, right = schedule('left'), schedule('right')
    assert len(left) == len(right) == MAX_TICKS
    for a, b in zip(left, right, strict=True):
        assert a['role'] == b['role'] and a['phase'] == b['phase']
        assert a['requested_command'][:2] == b['requested_command'][:2]
        assert a['requested_command'][2] == -b['requested_command'][2]
        validate_command(a['requested_command']); validate_command(b['requested_command'])
    for name, sign in (('turn_out', 1), ('turn_back', -1)):
        assert sum(r['requested_command'][2]*.1 for r in left if r['role'] == name) == pytest.approx(sign*3.15)
    assert tuple(inspect.signature(decision).parameters) == ('direction', 'tick', 'policy')
    d = decision('left', 80, policy(80))
    assert d['role'] == 'turn_out' and not d['tracker_required'] and not d['native_state_used']
    terminal = decision('right', MAX_TICKS, policy(MAX_TICKS))
    assert terminal['terminal'] and terminal['requested_command'] == [0., 0., 0.]


@pytest.mark.parametrize('fault', ['clock', 'privilege', 'identity', 'future', 'tick_bool', 'overrun', 'direction'])
def test_selector_fails_closed_on_sensor_and_schedule_faults(fault):
    p = policy(8); tick = 8; direction = 'left'
    if fault == 'clock': tick = 9
    elif fault == 'privilege': p['native_pose'] = [0]*7
    elif fault == 'identity': p['sensor_state']['identity'] = (0, 1, 0)
    elif fault == 'future': p['sensor_state']['sensed']['gyro']['available_ns'][-1] += 1
    elif fault == 'tick_bool': tick = True
    elif fault == 'overrun': tick = MAX_TICKS + 1
    elif fault == 'direction': direction = 'tune_from_observation'
    with pytest.raises(ValueError): decision(direction, tick, p)


def native_fixture(direction='left'):
    n = MAX_PHYSICS_SAMPLES
    t = np.arange(1, n+1)*.002
    p = np.zeros((n, 7)); p[:, 6] = 1
    v = np.zeros((n, 6)); yaw = np.zeros(n)
    offset = SETTLE_SAMPLES-1
    for name, count, c in SEGMENTS:
        size = count*50
        delta = np.arange(1, size+1)/size
        end = offset+size
        p[offset+1:end+1, 0] = p[offset, 0] + (.3*delta if c[0] else 0)
        change = (1 if direction == 'left' else -1) * (math.pi if c[2] > 0 else -math.pi if c[2] < 0 else 0.)
        yaw[offset+1:end+1] = yaw[offset] + change*delta
        offset = end
    p[:, 5] = np.sin(yaw/2); p[:, 6] = np.cos(yaw/2)
    return t, p, v


def coverage(t, p, v, **kwargs):
    return measured_coverage(kwargs.pop('direction', 'left'), t, p, v,
        **(dict(completed_ticks=MAX_TICKS, schedule_complete=True, physical_stop=None, acquisition_stop=None) | kwargs))


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_measured_turns_handle_wrapping_and_both_directions(direction):
    r = coverage(*native_fixture(direction), direction=direction)
    assert r['intended_motion_covered'] and all(r['turn_coverage'].values())
    assert all(r['translation_coverage'].values()) and r['final_second_stop_covered']
    assert not r['observer_accuracy_evaluated'] and not r['independent_observations_verified']
    assert not r['real_time_qualified'] and not r['navigation_qualified']
    assert r['segments']['turn_out']['signed_yaw_rad'] == pytest.approx(math.pi if direction == 'left' else -math.pi)


def test_complete_command_tape_with_no_motion_is_not_coverage():
    t, p, v = native_fixture(); p[:] = 0; p[:, 6] = 1
    r = coverage(t, p, v)
    assert r['schedule_complete'] and not r['intended_motion_covered']
    assert not any(r['turn_coverage'].values()) and not any(r['translation_coverage'].values())


def test_wrong_way_turns_do_not_pass_by_absolute_winding():
    r = coverage(*native_fixture('right'))
    assert not any(r['turn_coverage'].values()) and not r['intended_motion_covered']


def test_acquisition_stop_at_exact_tick_boundary_retains_negative_case():
    t, p, v = native_fixture(); n = SETTLE_SAMPLES + 100*50
    r = coverage(t[:n], p[:n], v[:n], completed_ticks=100, schedule_complete=False,
                 acquisition_stop='CURRENT_RGBD_UNAVAILABLE')
    assert not r['intended_motion_covered'] and r['acquisition_stop'] == 'CURRENT_RGBD_UNAVAILABLE'
    with pytest.raises(ValueError):
        coverage(t[:n+1], p[:n+1], v[:n+1], completed_ticks=100, schedule_complete=False,
                 acquisition_stop='CURRENT_RGBD_UNAVAILABLE')


def test_setup_only_stop_has_no_fake_future_motion_or_final_stop_window():
    t, p, v = native_fixture(); n = SETTLE_SAMPLES
    r = coverage(t[:n], p[:n], v[:n], completed_ticks=0, schedule_complete=False,
                 physical_stop='SETUP_REJECTED')
    assert r['segments']['initial_hold']['available'] and not r['segments']['initial_hold']['complete']
    assert all(not row['available'] for name, row in r['segments'].items() if name != 'initial_hold')
    assert not r['intended_motion_covered'] and r['final_second_linear_speed_max_m_s'] is None


@pytest.mark.parametrize('samples', [0, 1, 749])
def test_failure_before_settling_is_retained_without_claiming_any_motion(samples):
    t, p, v = native_fixture()
    r = coverage(t[:samples], p[:samples], v[:samples], completed_ticks=0, schedule_complete=False,
                 physical_stop='SETTLING_STOP')
    assert r['physics_samples'] == samples and not r['settling_complete']
    assert not any(row['available'] for row in r['segments'].values())
    assert not r['intended_motion_covered']


@pytest.mark.parametrize('axis,value', [(0, .021), (5, .051)])
def test_one_bad_sample_in_final_second_prevents_stop_coverage(axis, value):
    t, p, v = native_fixture(); v[-499, axis] = value
    r = coverage(t, p, v)
    assert not r['final_second_stop_covered'] and not r['intended_motion_covered']


@pytest.mark.parametrize('partial_samples', [0, 1, 49, 50])
def test_physical_stops_preserve_partial_segments_and_denominators(partial_samples):
    t, p, v = native_fixture(); count = 100; n = SETTLE_SAMPLES + count*50 + partial_samples
    r = coverage(t[:n], p[:n], v[:n], completed_ticks=count, schedule_complete=False,
                 physical_stop='CONTACT_STOP')
    assert r['physics_samples'] == n and not r['intended_motion_covered']
    assert not r['segments']['turn_out']['complete'] and r['segments']['turn_out']['available']
    assert not r['segments']['turn_back']['available']
    assert r['final_second_linear_speed_max_m_s'] is None


@pytest.mark.parametrize('fault', ['clock', 'shape', 'nan', 'quaternion', 'missing', 'extra', 'complete', 'ticks', 'no_stop'])
def test_invalid_or_misrepresented_native_prefix_is_rejected(fault):
    t, p, v = native_fixture(); kw = {}
    if fault == 'clock': t[800] += .001
    elif fault == 'shape': p = p[:, :6]
    elif fault == 'nan': v[800, 1] = np.nan
    elif fault == 'quaternion': p[800, 6] *= .9
    elif fault == 'missing': t, p, v = t[:-1], p[:-1], v[:-1]
    elif fault == 'extra': t, p, v = np.append(t, t[-1]+.002), np.vstack([p, p[-1]]), np.vstack([v, v[-1]])
    elif fault == 'complete': kw['physical_stop'] = 'STOP'
    elif fault == 'ticks': kw['completed_ticks'] = True
    elif fault == 'no_stop':
        t, p, v = t[:750], p[:750], v[:750]; kw = dict(completed_ticks=0, schedule_complete=False)
    with pytest.raises(ValueError): coverage(t, p, v, **kw)


def test_new_session_retains_real_recording_and_stop_chain_without_inventory_impersonation():
    from scripts.independent_tracking_session_development import IndependentTrackingSession, IndependentTrackingPhysicalInit
    from scripts.independent_pulse_context_physical_init_development import PulseContextPhysicalInit
    from scripts.independent_pulse_context_session_development import PulseContextSession
    from scripts.core_ordered_dynamic_session_development import CoreOrderedDynamicSession
    from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession
    mro = IndependentTrackingSession.__mro__
    assert mro.index(AttributedSession) < mro.index(IndependentTrackingPhysicalInit) < mro.index(PulseContextPhysicalInit)
    assert IndependentTrackingSession.capture_fixed_rgb is CoreOrderedDynamicSession.capture_fixed_rgb
    for name in ('sensor_packets', 'command_tick', 'install_contact_identity'):
        assert getattr(IndependentTrackingSession, name) is getattr(PulseContextSession, name)
    assert IndependentTrackingSession._sample is not PulseContextSession._sample  # Explicit pre-record errno guard.
    assert 'inventory' not in inspect.signature(IndependentTrackingSession).parameters


@pytest.mark.parametrize('fault', [None, 'order', 'friction', 'policy', 'topology'])
def test_native_initializer_exact_construction_and_cleanup_with_mock_backend(monkeypatch, tmp_path, fault):
    import scripts.independent_tracking_session_development as mod
    import lewm_genesis.lewm_contract as contract
    import lewm_genesis.rollout as rollout
    import lewm_genesis.union_wall_rgbd_scene_development as scene
    import lewm_genesis.ordered_union_raster_development as raster
    import lewm_genesis.scene_loader as loader
    import scripts.run_go2_oracle_branch_pilot_v1 as branch
    events = []; seen = {}
    build = NS(scene=NS(destroy=lambda: events.append('destroy')),
               camera=NS(_rasterizer=NS(_context=NS(_scene=object()))))
    def maybe(name, result):
        if fault == name: raise RuntimeError(name)
        return result
    def builder(definition, **kwargs):
        seen.update(pack=definition, options=kwargs)
        return build
    monkeypatch.setattr(scene, 'build_scene_from_pack', builder)
    monkeypatch.setattr(loader, 'load_platform_manifest', lambda _: 'platform')
    monkeypatch.setattr(raster, 'install_order', lambda _, order: maybe('order', order))
    monkeypatch.setattr(mod, 'native_friction', lambda b, mu, **kw: maybe('friction', dict(mu=mu, **kw)))
    monkeypatch.setattr(contract.PrimitiveRegistry, 'from_yaml', lambda _: 'registry')
    monkeypatch.setattr(contract.SafetyLimits, 'from_manifest', lambda _: 'safety')
    monkeypatch.setattr(rollout.GenesisGo2PPOPolicy, 'from_platform_manifest',
                        lambda *a, **k: maybe('policy', NS(simulate_action_latency=False)))
    def runner(*args, config): seen['runner_config'] = config; return 'runner'
    monkeypatch.setattr(rollout, 'RolloutRunner', runner)
    monkeypatch.setattr(branch, 'BranchContext', lambda **kw: NS(**kw))
    monkeypatch.setattr(mod, '_collect_solver_fields_compat', lambda _: {})
    s = NS(output=tmp_path, _build_contact_topology=lambda: maybe('topology', {'ground': {1}}))
    spec = specification('offset_niche_lower_friction_right')
    if fault:
        with pytest.raises(RuntimeError, match=fault): mod.IndependentTrackingPhysicalInit.__init__(s, spec, backend='cpu')
        assert events == ['destroy']
    else:
        mod.IndependentTrackingPhysicalInit.__init__(s, spec, backend='cpu')
        assert not events and s.raster_order == 'floor_first'
        assert s.initial_friction == dict(mu=.15, install=True)
        assert seen['options']['render_robot'] is False and seen['options']['backend'] == 'cpu'
        assert seen['pack'].physics_randomization.floor_friction_mu == .15
        assert seen['runner_config'].randomize_spawn_pose is False
        assert s._command_history.shape == (15, 3) and not s._runtime['real_time_qualified']
        assert s._runtime['physics_paused_during_compute']
        spec['geometry']['spawn_se2_world'][0] += 1
        assert s.geometry['spawn_se2_world'][0] == -.83


def test_non_cpu_initializer_rejected_before_native_or_output_access(tmp_path):
    from scripts.independent_tracking_session_development import IndependentTrackingPhysicalInit
    with pytest.raises(Exception, match='CPU'):
        IndependentTrackingPhysicalInit.__init__(NS(output=tmp_path), specification(TRIALS[0]), backend='gpu')
    assert not list(tmp_path.iterdir())
