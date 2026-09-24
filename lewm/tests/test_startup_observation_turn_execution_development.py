"""Synthetic execution/accounting failures; no Genesis initialization."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_go2_startup_observation_turn_development_v1 import audit_commands, stopping_metrics, json_same, native_materials
from scripts.startup_source_inventory_development import allowed_relative, discover_sources
from scripts.startup_observation_turn_session_development import StartupObservationSession
from scripts.bounded_rgbd_session_development import BoundedRGBDSession
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop


def tape_fixture():
    n = 950
    raw = dict(timestamp_s=np.arange(1, n+1)*.002, phase=np.r_[np.zeros(750), np.ones(50), np.full(150, 2)],
        requested_command=np.r_[np.zeros((750, 3)), np.tile([0., 0., .35], (50, 1)), np.zeros((150, 3))],
        base_twist_world=np.zeros((n, 6)), base_pose_world=np.tile([0., 0., .33, 0., 0., 0., 1.], (n, 1)))
    raw['applied_command'] = raw['requested_command'].astype(np.float32).astype(float)
    raw['post_slew_applied_command'] = raw['applied_command'].copy()
    tape = [dict(tick=i, pre_sample_index=749+50*i, post_sample_index=799+50*i,
        requested_command=[0., 0., .35] if i == 0 else [0., 0., 0.], phase=1 if i == 0 else 2,
        decision_index=0 if i == 0 else None, execution_wall_ms=1.) for i in range(4)]
    decisions = [dict(observation_index=i, decision=dict(decision_ns=1_500_000_000+i*100_000_000,
        requested_command=[0., 0., .35] if i == 0 else [0., 0., 0.], terminal=bool(i),
        status='OBSERVATION_TURN' if i == 0 else 'FAILED_SENSOR_OR_FUSION')) for i in range(2)]
    result = dict(requested_ticks=4, controller_decisions=2, stopping_tail_ticks=3,
        physical_stop_reason=None, controller_status='FAILED_SENSOR_OR_FUSION')
    return raw, tape, decisions, result


def test_failed_controller_can_have_verified_real_zero_tail_without_becoming_success():
    raw, tape, decisions, result = tape_fixture()
    report = audit_commands(raw, tape, decisions, result)
    assert report['complete_zero_tail_ticks'] == 3
    assert stopping_metrics(raw, tape)['final_quiet_window']
    assert result['controller_status'].startswith('FAILED_')


@pytest.mark.parametrize('fault', ['requested', 'applied', 'phase', 'missing_tick', 'sample_gap',
    'decision_after_terminal', 'unrecorded_command', 'tail_nonzero', 'tail_summary', 'clock', 'nonfinite_time'])
def test_command_audit_rejects_corruption(fault):
    raw, tape, decisions, result = tape_fixture()
    if fault == 'requested': raw['requested_command'][751, 0] = .1
    if fault == 'applied': raw['applied_command'][751, 2] = .2
    if fault == 'phase': raw['phase'][851] = 1
    if fault == 'missing_tick': tape.pop()
    if fault == 'sample_gap': tape[1]['pre_sample_index'] += 1
    if fault == 'decision_after_terminal': decisions.append(deepcopy(decisions[-1])); result['controller_decisions'] += 1
    if fault == 'unrecorded_command': tape[0]['decision_index'] = 1
    if fault == 'tail_nonzero': tape[-1]['requested_command'][2] = .1
    if fault == 'tail_summary': result['stopping_tail_ticks'] = 2
    if fault == 'clock': decisions[1]['decision']['decision_ns'] += 1
    if fault == 'nonfinite_time': tape[0]['execution_wall_ms'] = float('nan')
    with pytest.raises((ValueError, AssertionError)): audit_commands(raw, tape, decisions, result)


def test_partial_physical_stop_is_preserved_not_three_complete_tail_ticks():
    raw, tape, decisions, result = tape_fixture()
    raw = {k: v[:925] for k, v in raw.items()}; tape[-1]['post_sample_index'] = 924
    result.update(physical_stop_reason='BODY_STABILITY_LIMIT', stopping_tail_ticks=2)
    assert audit_commands(raw, tape, decisions, result)['complete_zero_tail_ticks'] == 2
    assert not stopping_metrics(raw, tape)['complete']


@pytest.mark.parametrize('channel,value', [(0, .051), (5, .101)])
def test_zero_command_and_quiet_endpoint_do_not_hide_motion_in_final_window(channel, value):
    raw, tape, _, _ = tape_fixture(); raw['base_twist_world'][-25, channel] = value
    assert not stopping_metrics(raw, tape)['final_quiet_window']


@pytest.mark.parametrize('path', ['sealed/a.py', 'a/sealed_test.json', 'a/sealed_data/x.py', '/tmp/a.py', '../x.py'])
def test_source_inventory_rejects_custody_paths_before_read(path):
    with pytest.raises(ValueError): allowed_relative(path)
    with pytest.raises(ValueError): discover_sources([path], {})


def test_source_inventory_includes_new_direct_and_transitive_dependencies():
    # Stop at explicit inherited sources as documented, rather than scanning a tree.
    inherited = {'scripts/run_go2_successive_choice_maze_development_v1.py': 'a'*64}
    sources = discover_sources(['scripts/startup_source_inventory_development.py'], inherited)
    assert set(sources) == set(inherited) | {'scripts/startup_source_inventory_development.py'}
    assert sources[next(iter(inherited))] == 'a'*64


def test_json_boundary_changes_key_encoding_not_native_identity():
    json_same({'feet': {12: 'FL_foot:0'}, 'identity': (0, 0, 0)}, {'feet': {'12': 'FL_foot:0'}, 'identity': [0, 0, 0]})
    with pytest.raises(ValueError): json_same({'feet': {12: 'FL_foot:0'}}, {'feet': {'13': 'FL_foot:0'}})


@pytest.mark.parametrize('failure', [None, 'speed', 'nonfoot', 'inherited'])
def test_live_wrapper_retains_native_stop_and_records_strict_extra_guards(monkeypatch, failure):
    row = dict(base_twist_world=np.array([.301 if failure == 'speed' else 0., 0., 0., 0., 0., 0.]))
    packet = dict(geom_a=np.array([[0]]), geom_b=np.array([[12 if failure == 'nonfoot' else 11]]),
        valid_mask=np.array([[True]]), force_a=np.array([[[0., 0., -20.]]]), force_b=np.array([[[0., 0., 20.]]]))
    session = SimpleNamespace(samples=[row], packets=[packet], startup_guard_rows=[],
        startup_guard=dict(robot_geom_ids=[11, 12], foot_geom_ids=[11], ground_geom_ids=[0]))
    def sample(self, *args):
        if failure == 'inherited': raise PhysicalStop('DISALLOWED_CONTACT')
        return row
    monkeypatch.setattr(BoundedRGBDSession, '_sample', sample)
    # __new__ supplies the real MRO without constructing a simulator.
    instance = StartupObservationSession.__new__(StartupObservationSession); instance.__dict__.update(session.__dict__)
    if failure:
        with pytest.raises(PhysicalStop): instance._sample([0.]*3, [0.]*3, 1.502)
    else: assert instance._sample([0.]*3, [0.]*3, 1.502) is row
    assert len(instance.startup_guard_rows) == (0 if failure == 'inherited' else 1)


def test_material_identity_comparison_excludes_motion_but_not_friction():
    row = dict(geom_id=3, link_id=5, link_name='FL_calf', geom_type='SPHERE', data=[.022]+[0.]*6,
               friction=.6, solver_parameters=[.02, 1.], position_world_m=[0., 0., 0.])
    moved = deepcopy(row); moved['position_world_m'][0] = .1
    assert native_materials([row]) == native_materials([moved])
    moved['friction'] = .7
    assert native_materials([row]) != native_materials([moved])


def test_actual_body_region_check_uses_initial_frame_and_full_primitives():
    from scripts.audit_go2_startup_observation_turn_development_v1 import padded_body_inside_setup
    from scripts.startup_observation_turn_session_development import make_priors
    from lewm.tests.test_native_foot_geometry_evaluation_development import fixture
    _, geometry, q, pose = fixture(); _, region = make_priors(1_500_000_000, 'a'*64)
    assert padded_body_inside_setup(geometry, region, pose, pose, q)
    moved = pose.copy(); moved[0] += 1.
    assert not padded_body_inside_setup(geometry, region, pose, moved, q)
    shifted = pose.copy(); shifted[:3] += [100., -100., 2.]
    assert padded_body_inside_setup(geometry, region, shifted, shifted, q)
