"""Separate evidence-acquisition decisions from physical and identity claims."""
import ast
import copy
import inspect
import math
import textwrap

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import choose_side_branch
from lewm.fixed_forward_traversal_development import FixedForwardTraversal
from lewm.observed_traversal_controller_development import ObservedTraversalController
from lewm.online_temporal_choice_development import OnlineTemporalChoice
from lewm.persistent_alignment_continuation_development import PersistentAlignedContinuation
from lewm.persistent_alignment_scene_development import trials as old_trials
from lewm.stop_observe_traversal_development import StopObserveTraversal
from lewm.task_acquisition_continuation_development import TaskAcquisitionContinuation, POLICIES
from lewm.task_acquisition_metrics_development import reduce_task_acquisition, task_scan_acquisition
from lewm.task_acquisition_scene_development import trials
from lewm.tests.test_observed_continuation_development import MovingStream, branch
from lewm.tests.test_observed_continuation_metrics_development import fixture
from lewm.tests.test_observed_traversal_controller_development import Stream
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def tree(method):
    return ast.parse(textwrap.dedent(inspect.getsource(method)))


def test_only_translation_cap_branch_changes_not_any_arrival_predicate():
    new, old = tree(StopObserveTraversal._observe), tree(ObservedTraversalController._observe)
    count = 0
    for node in ast.walk(new):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'progress >= 1.4':
            assert [ast.unparse(n) for n in node.body] == ["self.status = 'BRAKING'", 'self.brake_ns = now_ns']
            node.body = ast.parse("self.status = 'FAILED_NO_VISUAL_CHANGE'").body
            count += 1
    assert count == 1 and ast.dump(new) == ast.dump(old)


def stopped_arrival(*, evidence_after_stop=True, quiet=True):
    controller = StopObserveTraversal('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream, command, rows = Stream(), [0., 0., 0.], []
    for tick in range(145):
        p, fast, now = stream.frame(tick, command,
            changed=evidence_after_stop and controller.status == 'BRAKING', quiet=quiet)
        row = controller.observe(p, fast, now_ns=now)
        rows.append(row)
        if row['terminal']: return controller, rows
        command = row['requested_command']
    raise AssertionError('unbounded stopped observation')


def test_cap_stops_motion_then_acquires_new_evidence_without_creating_early_arrival():
    controller, rows = stopped_arrival()
    brake = next(r for r in rows if r['status'] == 'BRAKING')
    assert brake['command_progress_proxy_m'] >= 1.4
    assert brake['floor_mask_change_fraction'] == 0. and not brake['terminal']
    assert brake['ledger']['status'] == 'PENDING'
    assert all(r['requested_command'] == [0., 0., 0.] for r in rows if r['decision_ns'] >= brake['decision_ns'])
    assert rows[-1]['status'] == 'ARRIVAL_CANDIDATE'
    assert rows[-1]['decision_ns']-brake['decision_ns'] >= 500_000_000
    assert min(r['floor_mask_change_fraction'] for r in rows[-3:]) >= .1
    record = controller.ledger.snapshot()
    assert record['trusted_graph_edges'] == 0 and record['arrival']['place_identity'] is None


@pytest.mark.parametrize('evidence,quiet,status', [(False, True, 'FAILED_NO_VISUAL_CHANGE'), (True, False, 'FAILED_SETTLING')])
def test_stopping_does_not_replace_missing_evidence_or_quiet_dwell(evidence, quiet, status):
    controller, rows = stopped_arrival(evidence_after_stop=evidence, quiet=quiet)
    assert rows[-1]['status'] == status and rows[-1]['requested_command'] == [0., 0., 0.]
    assert controller.ledger.snapshot()['arrival'] is None


def synthetic(policy, *, compare_baseline=False):
    geometry = ArticulatedCollisionGeometry(URDF)
    controller = TaskAcquisitionContinuation('fixed_forward', geometry, acquisition_policy=policy)
    old = PersistentAlignedContinuation('fixed_forward', geometry) if compare_baseline else None
    stream, command, rows = MovingStream(), [0., 0., 0.], []
    for tick in range(801):
        child = controller.first if controller.stage == 'FIRST' else controller.second if controller.stage == 'SECOND' else None
        p, fast, now = stream.frame(tick, command, changed=child is not None and child.tick >= 3)
        row = controller.observe(p, fast, now_ns=now)
        assert row['global_orientation']['samples_integrated'] == tick*50
        if old:
            expected = old.observe(p, fast, now_ns=now)
            assert {k: v for k, v in row.items() if k not in ('acquisition_policy', 'scan_stop_evidence')} == expected
        rows.append(row)
        if row['terminal']: return controller, rows
        command = row['requested_command']
    raise AssertionError('unbounded acquisition task')


@pytest.mark.parametrize('policy', POLICIES)
def test_all_four_policy_variants_complete_synthetic_task_with_honest_scan_status(policy):
    controller, rows = synthetic(policy, compare_baseline=policy == 'baseline')
    assert rows[-1]['status'] == 'COMPLETE_PROVISIONAL' and rows[-1]['trusted_graph_edges'] == 0
    interrupted = [r for r in rows if r['scan_stop_evidence']]
    assert len(interrupted) == (1 if policy in ('scan_only', 'both') else 0)
    if interrupted:
        row = interrupted[0]
        assert row['requested_command'] == [0., 0., 0.] and row['next_stage'] == 'HOLD_ALIGN'
        assert row['scan']['status'] == controller.scan.status == 'SCANNING'
        assert row['selected_side_branch']['observed_ns'] == row['decision_ns']
        assert row['selected_side_branch']['requires_fresh_forward_reobservation']
        assert all(r['stage'] != 'SCAN' for r in rows[rows.index(row)+1:])
    else: assert controller.scan.status == 'COMPLETE'
    second = [r['child'] for r in rows if r['stage'] == 'SECOND']
    assert [r['status'] for r in second[:4]] == ['WARMUP']*3+['TRAVERSING']
    assert type(controller.first) is (StopObserveTraversal if policy in ('arrival_only', 'both') else FixedForwardTraversal)


def partial_row(now=5_500_000_000):
    item = branch(math.pi/2, timestamp=now)
    selected = choose_side_branch([item], [1., 0., 0.], now_ns=now)
    return {'pre_sample_index': 2749, 'decision_ns': now,
        'controller': {'stage': 'SCAN', 'next_stage': 'HOLD_ALIGN', 'status': 'RUNNING',
            'terminal': False, 'child': None, 'requested_command': [0., 0., 0.],
            'acquisition_policy': 'both', 'selected_view_proposals': [item['candidate']],
            'selected_side_branch': selected,
            'scan': {'status': 'SCANNING', 'new_completed_view': {'decision_ns': now}, 'completed_target_views': 1},
            'scan_stop_evidence': {'reason': 'FRESH_SIDE_BRANCH_AT_ACQUIRED_VIEW', 'decision_ns': now,
                'observed_ns': now, 'completed_target_views': 1, 'full_circle_complete': False}}}


def test_old_branch_memory_does_not_trigger_early_stop_without_current_view_evidence(monkeypatch):
    row = partial_row(); now = row['decision_ns']
    c = TaskAcquisitionContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF), acquisition_policy='both')
    c.stage, c.incoming = 'SCAN', np.array([1., 0., 0.])
    c.observations = [branch(math.pi/2, timestamp=now-100_000_000)]
    parent = copy.deepcopy(row['controller']); parent['next_stage'] = 'SCAN'
    parent['selected_side_branch'] = None; parent['selected_view_proposals'] = []
    monkeypatch.setattr(PersistentAlignedContinuation, '_observe', lambda *a, **k: copy.deepcopy(parent))
    result = c.observe(None, None, now_ns=now)
    assert result['scan_stop_evidence'] is None and c.stage == 'SCAN' and c.selected is None


def test_partial_scan_task_success_retains_false_full_scan_metric_and_exact_physical_windows():
    spec, raw, decisions, geometry = fixture()
    spec['acquisition_policy'] = 'both'; decisions[2] = partial_row()
    result = reduce_task_acquisition(spec, raw, 749, decisions, 'COMPLETE_PROVISIONAL', None, None, geometry)
    assert result['task_two_leg_integration_success']
    assert not result['checks']['completed_scan'] and not result['two_leg_integration_success']
    assert result['scan_interrupted_after_evidence'] and result['trusted_graph_edges'] == 0
    assert result['legs'][0]['end_sample_index'] == 1999 and result['legs'][1]['end_sample_index'] == 4499
    raw['requested_command'][1800, 0] = .1
    failed = reduce_task_acquisition(spec, raw, 749, decisions, 'COMPLETE_PROVISIONAL', None, None, geometry)
    assert not failed['task_two_leg_integration_success']


@pytest.mark.parametrize('fault', ['stale', 'missing_image_evidence', 'nonzero', 'full_scan_lie', 'qualified', 'duplicate', 'no_view', 'wrong_policy'])
def test_partial_scan_cannot_be_fabricated_or_used_to_relax_other_evidence(fault):
    row = partial_row(); c = row['controller']; rows = [row]; policy = 'both'
    if fault == 'stale': c['selected_side_branch']['observed_ns'] -= 1
    elif fault == 'missing_image_evidence': c['selected_view_proposals'] = []
    elif fault == 'nonzero': c['requested_command'][2] = .1
    elif fault == 'full_scan_lie': c['scan']['status'] = 'COMPLETE'
    elif fault == 'qualified': c['selected_side_branch']['qualified_exit'] = True
    elif fault == 'duplicate': rows.append(copy.deepcopy(row))
    elif fault == 'no_view': c['scan']['new_completed_view'] = None
    else: policy = 'baseline'
    with pytest.raises(ValueError): task_scan_acquisition(rows, policy)


@pytest.mark.parametrize('method', ['direct_direct', 'supervised_rollout', 'jepa_rollout'])
def test_actual_frozen_ensembles_start_on_four_fresh_frames_with_new_arrival_operator(method):
    template = OnlineTemporalChoice.from_completed_study(method)
    c = TaskAcquisitionContinuation(method, ArticulatedCollisionGeometry(URDF), template, acquisition_policy='both')
    stream, command, times = MovingStream(), [0., 0., 0.], []
    for tick in range(100):
        p, fast, now = stream.frame(tick, command)
        row = c.observe(p, fast, now_ns=now); assert not row['terminal']
        command = row['requested_command']
        if row['stage'] == 'FIRST': times.append(now)
        if row['child'] and row['child']['selection']:
            selected = row['child']['selection']
            assert len(times) == 4 and [r['measured_ns'] for r in selected['input_images']] == times
            assert selected['model_bindings'] == template.bindings
            assert type(c.first) is StopObserveTraversal
            break
    else: raise AssertionError('no fitted first decision')


def test_fault_latches_before_any_operator_or_graph_promotion():
    c = TaskAcquisitionContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF), acquisition_policy='both')
    stream = MovingStream()
    for tick in range(4):
        p, fast, now = stream.frame(tick); c.observe(p, fast, now_ns=now)
    p, fast, now = stream.frame(4); fast['valid'][-3] = False
    with pytest.raises(SensorContractError): c.observe(p, fast, now_ns=now)
    assert c.status == 'FAILED_SENSOR' and c.first.ledger.snapshot() is None
    with pytest.raises(SensorContractError): c.observe(p, fast, now_ns=now)


def test_fixed28_population_and_collector_auditor_model_policy_identity():
    import scripts.run_go2_task_acquisition_continuation_development_v1 as runner
    import scripts.audit_go2_task_acquisition_continuation_development_v1 as auditor
    rows = trials(); previous = {(s['case_index'], s['method']): s for s in old_trials()}
    assert len(rows) == len({s['scene_id'] for s in rows}) == 28
    for case in range(4):
        group = [s for s in rows if s['case_index'] == case]
        assert [(s['method'], s['acquisition_policy']) for s in group] == [
            *[('fixed_forward', p) for p in POLICIES],
            ('direct_direct', 'both'), ('supervised_rollout', 'both'), ('jepa_rollout', 'both')]
        assert len({s['procedural_seed'] for s in group}) == 1
        for s in group:
            p = previous[(case, s['method'])]
            assert s['geometry'] == p['geometry'] and s['evaluation_leg_geometries'] == p['evaluation_leg_geometries']
    assert runner.ObservedContinuation is auditor.ObservedContinuation is TaskAcquisitionContinuation
    assert runner.reduce_continuation is auditor.reduce_continuation is reduce_task_acquisition
    assert runner.OUTPUT == auditor.OUTPUT
    for func in (runner.collect, auditor.audit_trial):
        calls = [n for n in ast.walk(tree(func)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'ObservedContinuation']
        assert len(calls) == 1 and ast.unparse(calls[0].keywords[0]) == "acquisition_policy=spec['acquisition_policy']"
