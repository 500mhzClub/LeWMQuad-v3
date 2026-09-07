import ast
import inspect
import json
import math
import textwrap

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import RelativeBearingAlignment
from lewm.initially_aligned_continuation_development import FineInitialBearingAlignment, InitiallyAlignedContinuation
from lewm.initially_aligned_scene_development import trials
from lewm.initially_aligned_metrics_development import initial_alignment_geometry
from lewm.observed_continuation_scene_development import trials as old_trials
from lewm.online_temporal_choice_development import OnlineTemporalChoice
from lewm.tests.test_observed_continuation_development import MovingStream
from lewm.fast_gyro_development import FastRelativeOrientation
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def test_fine_alignment_changes_only_error_tolerance_not_packet_rate_or_budget():
    old = ast.parse(textwrap.dedent(inspect.getsource(RelativeBearingAlignment.observe)))
    new = ast.parse(textwrap.dedent(inspect.getsource(FineInitialBearingAlignment.observe)))
    count = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Constant) and node.value == .02:
            node.value = .08
            count += 1
    assert count == 1
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


@pytest.mark.parametrize('target', [.08726646259971649, -.1, .3])
def test_small_observed_bearings_are_actually_aligned_in_ideal_stream(target):
    stream, tracker = MovingStream(), FastRelativeOrientation()
    align = FineInitialBearingAlignment([math.cos(target), math.sin(target), 0.])
    command = [0., 0., 0.]
    for tick in range(121):
        packet, fast, now = stream.frame(tick, command)
        attitude = tracker.begin(packet, fast, now_ns=now) if tick == 0 else tracker.step(packet, fast, now_ns=now)
        result = align.observe(packet, attitude, now_ns=now)
        command = result['requested_command']
        if result['status'] != 'ALIGNING': break
    assert result['status'] == 'COMPLETE' and abs(result['heading_error_rad']) <= .02
    assert abs(math.atan2(tracker.rotation[1, 0], tracker.rotation[0, 0])) >= abs(target)-.02
    assert command == [0., 0., 0.]


def test_unresponsive_fine_alignment_times_out_without_translation():
    stream, tracker = MovingStream(), FastRelativeOrientation()
    align = FineInitialBearingAlignment([math.cos(.1), math.sin(.1), 0.])
    for tick in range(121):
        packet, fast, now = stream.frame(tick)
        attitude = tracker.begin(packet, fast, now_ns=now) if tick == 0 else tracker.step(packet, fast, now_ns=now)
        result = align.observe(packet, attitude, now_ns=now)
    assert result['status'] == 'FAILED_TIMEOUT' and result['requested_command'] == [0., 0., 0.]


def test_full_synthetic_continuation_has_fresh_first_warmup_and_one_global_reference():
    controller = InitiallyAlignedContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream, command, stages, first_children = MovingStream(), [0., 0., 0.], [], []
    for tick in range(801):
        child = controller.first if controller.stage == 'FIRST' else controller.second if controller.stage == 'SECOND' else None
        changed = child is not None and child.tick >= 3
        packet, fast, now = stream.frame(tick, command, changed=changed)
        result = controller.observe(packet, fast, now_ns=now)
        assert result['global_orientation']['samples_integrated'] == tick*50
        assert json.loads(json.dumps(result, allow_nan=False)) == result
        stages.append(result['stage'])
        if result['stage'].startswith('INITIAL_'):
            assert result['child'] is None and controller.first.tick == -1
            assert result['requested_command'][:2] == [0., 0.]
        elif result['stage'] == 'FIRST': first_children.append(result['child'])
        if result['terminal']: break
        command = result['requested_command']
    assert result['status'] == 'COMPLETE_PROVISIONAL'
    assert list(dict.fromkeys(stages))[:4] == ['INITIAL_OBSERVE', 'INITIAL_ALIGN', 'INITIAL_HOLD', 'FIRST']
    assert [r['status'] for r in first_children[:4]] == ['WARMUP']*3+['TRAVERSING']
    assert result['trusted_graph_edges'] == 0


def test_no_initial_floor_evidence_never_starts_alignment_or_traversal():
    controller = InitiallyAlignedContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream = MovingStream()
    for tick in range(4):
        packet, fast, now = stream.frame(tick, floor=False)
        result = controller.observe(packet, fast, now_ns=now)
    assert result['status'] == 'FAILED_INITIAL_NO_EXIT'
    assert controller.first.tick == -1 and controller.first.ledger.snapshot() is None
    with pytest.raises(SensorContractError): controller.observe(packet, fast, now_ns=now)


@pytest.mark.parametrize('method', ['direct_direct', 'supervised_rollout', 'jepa_rollout'])
def test_actual_frozen_ensemble_begins_only_after_new_aligned_observations(method):
    template = OnlineTemporalChoice.from_completed_study(method)
    controller = InitiallyAlignedContinuation(method, ArticulatedCollisionGeometry(URDF), template)
    stream, command, first_times = MovingStream(), [0., 0., 0.], []
    for tick in range(100):
        packet, fast, now = stream.frame(tick, command)
        result = controller.observe(packet, fast, now_ns=now)
        assert not result['terminal']
        command = result['requested_command']
        if result['stage'] == 'FIRST': first_times.append(now)
        child = result['child']
        if child is not None and child['selection'] is not None:
            selected = child['selection']
            assert len(first_times) == 4
            assert [r['measured_ns'] for r in selected['input_images']] == first_times
            assert selected['model_bindings'] == template.bindings
            break
    else: raise AssertionError('never reached actual fitted selection')


def test_new_panel_changes_initial_operator_not_fixture_geometry():
    assert len(trials()) == 16
    for old, new in zip(old_trials(), trials(), strict=True):
        assert old['geometry'] == new['geometry']
        assert old['evaluation_leg_geometries'] == new['evaluation_leg_geometries']
        assert old['method'] == new['method']
        assert old['scene_id'] != new['scene_id']


def test_initial_heading_diagnostic_is_relative_and_does_not_assert_centering():
    yaw0, yaw1, target = .7, .78, .1
    poses = np.array([[0., 0., .3, 0., 0., math.sin(yaw0/2), math.cos(yaw0/2)],
                      [.01, -.02, .3, 0., 0., math.sin(yaw1/2), math.cos(yaw1/2)]])
    raw = {'base_pose_world': poses}
    decisions = [{'pre_sample_index': 0, 'controller': {'stage': 'INITIAL_OBSERVE',
        'selected_initial_bearing': {'direction_initial_body': [math.cos(target), math.sin(target), 0.]}}},
        {'pre_sample_index': 1, 'controller': {'stage': 'FIRST'}}]
    result = initial_alignment_geometry(raw, 0, decisions)
    assert result['first_start_true_heading_initial_rad'] == pytest.approx(.08)
    assert result['first_start_abs_heading_error_rad'] == pytest.approx(.02)
    assert result['initial_phase_actual_translation_m'] == pytest.approx(math.sqrt(.0005))
    assert result['evaluation_only'] and not result['corridor_centering_qualified']
    assert initial_alignment_geometry(raw, 0, decisions[:1]) is None


def test_physical_collector_and_replay_share_successor_and_correct_long_reader():
    import scripts.run_go2_initially_aligned_continuation_development_v1 as runner
    import scripts.audit_go2_initially_aligned_continuation_development_v1 as auditor
    from lewm.continuation_rgb_dataset_development import load_continuation_observation
    assert runner.ObservedContinuation is auditor.ObservedContinuation is InitiallyAlignedContinuation
    assert auditor.load_route_observation is load_continuation_observation
    assert runner.reduce_continuation is auditor.reduce_continuation
    assert runner.OUTPUT == auditor.OUTPUT
    assert runner.OUTPUT.name == 'go2_initially_aligned_continuation_development_v1_attempt_001'
