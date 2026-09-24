"""Causal replay boundary and complete unchanged-state checks."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.residual_first_interval_prefix_development import compare_step, MAX_FRAMES
from scripts import replay_go2_residual_first_interval_prefix_v1 as runner


def decisions(frame=3, *, action=None):
    old = dict(tick=frame, controller='measured_floor_transport_round_trip_controller_v1',
        requested_command=[0., 0., 0.], terminal=None, failure=None,
        selected_action=None, plan_offset=0, infeasible_wait_active=True,
        consecutive_infeasible_observations=1, feasible_action_recoveries=0,
        evidence={'pose':[1., 2.]}, mission_receipt={'phase':'OUTBOUND'}, memory_receipt={'cells':17},
        causal_residual_receipt=dict(frame=frame, residuals=[], pending_forecast_tick=frame),
        new_selection=dict(action=None, action_index=None, requested_command=[0., 0., 0.],
            phase_admissible_candidates=0, prediction=[[[.01, 0.]]],
            candidates=[{'utility_m':.5}], surface_checks=[{'possible_intersection':False}],
            nominal_path_checks=[{'clear':False}]))
    candidate = deepcopy(old)
    candidate.update(controller='residual_first_interval_feasibility_controller_v1',
                     residual_first_interval_feasibility_fallback_enabled=True)
    if action is not None:
        command = [.2, 0., 0.] if action == 'forward' else [0., 0., 0.]
        candidate.update(requested_command=command, selected_action=action, plan_offset=1,
            infeasible_wait_active=False, consecutive_infeasible_observations=0)
        candidate['new_selection'].update(action=action, action_index=int(action=='forward'),
            requested_command=command, phase_admissible_candidates=1,
            residual_first_interval_feasibility=dict(original_action=None, frame=frame,
                measured_ns=1_500_000_000+frame*100_000_000, selected_action=action,
                raw_predictions_remain_residual_targets=True, original_surface_vetoes_preserved=True,
                all_eight_segments_checked=True))
    return old, candidate


@pytest.mark.parametrize('fault', [None, 'pose', 'mission', 'memory', 'residual', 'prediction',
    'utility', 'surface', 'path', 'extra', 'metadata', 'receipt_time', 'actual_command', 'menu'])
def test_only_declared_policy_changes_are_admitted(fault):
    old, new = decisions(action='forward')
    if fault == 'pose': new['evidence']['pose'][0] = 5.
    elif fault == 'mission': new['mission_receipt']['phase'] = 'RETURN'
    elif fault == 'memory': new['memory_receipt']['cells'] += 1
    elif fault == 'residual': new['causal_residual_receipt']['residuals'] = [.1]
    elif fault == 'prediction': new['new_selection']['prediction'][0][0][0] = 0.
    elif fault == 'utility': new['new_selection']['candidates'][0]['utility_m'] = 9.
    elif fault == 'surface': new['new_selection']['surface_checks'][0]['possible_intersection'] = True
    elif fault == 'path': new['new_selection']['nominal_path_checks'][0]['clear'] = True
    elif fault == 'extra': new['unreviewed_change'] = True
    elif fault == 'metadata': new['residual_first_interval_feasibility_fallback_enabled'] = False
    elif fault == 'receipt_time': new['new_selection']['residual_first_interval_feasibility']['frame'] = 4
    elif fault == 'actual_command': old['requested_command'] = [.1, 0., 0.]
    elif fault == 'menu': new['requested_command'] = [.3, 0., 0.]
    before = deepcopy((old, new))
    if fault is None:
        check = compare_step(old, new, [0., 0., 0.], frame=3)
        assert check['requested_command_changed'] and check['policy_action_changed']
        assert check['complete_original_selection_preserved']
    else:
        with pytest.raises(ValueError): compare_step(old, new, [0., 0., 0.], frame=3)
    assert (old, new) == before


def test_hold_can_change_wait_state_without_changing_actual_command():
    old, new = decisions(action='hold')
    check = compare_step(old, new, [0., 0., 0.], frame=3)
    assert check['policy_action_changed'] and not check['requested_command_changed']
    old, new = decisions(frame=4)
    new['feasible_action_recoveries'] = 1
    with pytest.raises(ValueError): compare_step(old, new, [0., 0., 0.], frame=4)
    assert not compare_step(old, new, [0., 0., 0.], frame=4, prior_policy_changed=True)['requested_command_changed']


def test_terminal_boundary_can_change_only_current_pending_forecast():
    old, new = decisions(action='hold'); new['terminal'] = 'VIEW_BUDGET_EXHAUSTED'
    new['causal_residual_receipt']['pending_forecast_tick'] = None
    assert compare_step(old, new, [0., 0., 0.], frame=3)['terminal_changed']
    new['causal_residual_receipt']['pending_forecast_tick'] = 4
    with pytest.raises(ValueError): compare_step(old, new, [0., 0., 0.], frame=3)


@pytest.mark.parametrize('boundary', ['command', 'terminal', 'old_terminal', 'limit', 'mutated_input', 'model_change'])
def test_replay_stops_before_post_intervention_packet(monkeypatch, tmp_path, boundary):
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda *a:object())
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda *a:SimpleNamespace(free=100*1024**3))
    weight = {'value':runner.MODEL_STATE}
    model = SimpleNamespace(state_dict=lambda:dict(weight), parameters=lambda:[])
    monkeypatch.setattr(runner, 'load_assigned', lambda *a:(model, 'jepa', 'full'))
    monkeypatch.setattr(runner, 'state_digest', lambda d:d['value'])
    packet_reads = []; decision_reads = []
    stop = boundary in ('command', 'terminal', 'old_terminal')
    def rows(path):
        for i in range(MAX_FRAMES):
            decision_reads.append(i)
            if stop and i > 3: pytest.fail('consumed post-boundary decision')
            old, _ = decisions(i)
            if boundary == 'old_terminal' and i == 3: old['terminal'] = 'VIEW_BUDGET_EXHAUSTED'
            yield dict(tick=i, observation_index=i, pre_sample_index=749+50*i, decision=old)
        pytest.fail('consumed beyond fixed prefix limit')
    monkeypatch.setattr(runner, 'read_rows', rows)
    class Reader:
        frames = list(range(MAX_FRAMES+2))
        def packet(self, i):
            packet_reads.append(i)
            if stop and i > 3: pytest.fail('consumed post-boundary packet')
            return {'tick':i}, {}, {}, 1_500_000_000+i*100_000_000
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', lambda *a:Reader())
    tape = [dict(requested_command=[0., 0., 0.], completed=True) for _ in range(MAX_FRAMES+1)]
    monkeypatch.setattr(runner, 'read_json', lambda p,n:tape if n=='command_tape.json' else [{}]*(MAX_FRAMES+2))
    monkeypatch.setattr(runner, 'public_acquisition', lambda r:r)
    monkeypatch.setattr(runner, 'packet', lambda *a,**k:({}, {}))
    class Controller:
        def observe(self, policy, *a, **k):
            i = policy['tick']; _, result = decisions(i)
            if i == 3:
                if boundary == 'command': _, result = decisions(i, action='forward')
                elif boundary == 'terminal':
                    _, result = decisions(i, action='hold'); result['terminal'] = 'VIEW_BUDGET_EXHAUSTED'
                    result['causal_residual_receipt']['pending_forecast_tick'] = None
                elif boundary == 'old_terminal': result['terminal'] = 'VIEW_BUDGET_EXHAUSTED'
                elif boundary == 'model_change': weight['value'] = 'changed'
                elif boundary == 'mutated_input': policy['changed'] = True
            return result
    monkeypatch.setattr(runner, 'ResidualFirstIntervalController', lambda *a,**k:Controller())
    if boundary in ('mutated_input', 'model_change'):
        with pytest.raises(ValueError, match='mutated public' if boundary=='mutated_input' else 'unchanged weights'):
            runner.replay(dict(correction_admission={}))
    else:
        r = runner.replay(dict(correction_admission={}))
        assert packet_reads == decision_reads == list(range(4 if stop else MAX_FRAMES))
        assert r['first_requested_command_difference'] == (3 if boundary=='command' else None)
        assert r['first_terminal_policy_difference'] == (3 if boundary=='terminal' else None)
        assert not r['following_recorded_observations_consumed'] and not r['native_execution']


@pytest.mark.parametrize('fault', [None, 'order', 'raw_audit', 'model', 'controller'])
def test_completed_cohort_admission_keeps_negative_outcome(fault):
    record = dict(case=runner.CASE[0], layout_index=2, status='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED', verified_round_trip=False)
    result = dict(status='INDEPENDENT_FLOOR_TRANSPORT_MAZES_V1_COMPLETE', all_fixed_cases_executed=True,
        conditions=[{'layout_index':1}, record, {'layout_index':3}])
    old = dict(model_state_sha256=runner.MODEL_STATE, planned_cases=[list(runner.CASE)], implementation_class='MeasuredFloorTransportController')
    audit = dict(layout_index=2, verified_round_trip=False, raw_sensor_reconstruction_pass=True,
        additional_auxiliary_rgb_reconstructed=True, raw_model_command_replay_pass=True,
        raw_command_audit_pass=True, model_state_unchanged=True)
    if fault == 'order': result['conditions'].reverse()
    elif fault == 'raw_audit': audit['raw_command_audit_pass'] = False
    elif fault == 'model': old['model_state_sha256'] = 'changed'
    elif fault == 'controller': old['implementation_class'] = 'Other'
    if fault is None: runner.admit(result, audit, old)
    else:
        with pytest.raises(ValueError): runner.admit(result, audit, old)
