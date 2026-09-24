"""Original physics/audits, exact causal intervention and stopped comparisons."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.tests.test_executed_waypoint_score_development import fixture
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.commitment_contact_score_development import score_commitment_contact
from lewm.commitment_contact_prefix_development import PrefixComparison, MAX_FRAMES
from scripts import supervised_commitment_contact_native_prefix_development as prefix


@pytest.mark.parametrize('kind,name', [('episode', 'collect'), ('episode', 'artifacts'), ('audit', 'audit')])
def test_original_physical_collection_and_complete_raw_audit_calculations_preserved(kind, name):
    def function(path):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'CommitmentContactController': node.id = 'MeasuredFloorTransportController'
            return node
        def visit_Constant(self, node):
            if isinstance(node.value, str): node.value = node.value.replace('SUPERVISED_COMMITMENT_CONTACT_MAZE01', 'MEASURED_FLOOR_TRANSPORT_MAZE')
            return node
        def visit_Call(self, node):
            node.keywords = [k for k in node.keywords if k.arg != 'commitment_contact_policy_enabled']
            return self.generic_visit(node)
        def visit_Assert(self, node):
            if ast.unparse(node.test) == "result['commitment_contact_policy_enabled'] is True": return None
            return self.generic_visit(node)
    old = function('scripts/measured_floor_transport_maze_'+kind+'_development.py')
    new = function('scripts/supervised_commitment_contact_maze01_'+kind+'_development.py')
    assert ast.dump(old) == ast.dump(Normalize().visit(new))


def evidence():
    s, residual = fixture()
    for row in s['prediction'][1]: row[0] = 0.
    for row in s['prediction'][3]: row[0] = .015
    s['prediction'][3][-1][4] = 0.
    for row in s['prediction'][5]: row[0] = .005
    old_selection = score_waypoint_execution(s, residual); new_selection = score_commitment_contact(old_selection)
    assert old_selection['action'] == 'right_turn' and new_selection['action'] == 'right_arc'
    originals = []; saved = []; comparator = PrefixComparison()
    for i in range(4):
        old = dict(tick=i, controller='measured_floor_transport_round_trip_controller_v1',
            model_condition='supervised_rollout', input_variant='full', memory_variant='persistent', terminal=None, failure=None,
            new_selection=deepcopy(old_selection) if i == 3 else None, evidence={'observed':i},
            selected_action='right_turn' if i == 3 else None,
            requested_command=[0., 0., -.45] if i == 3 else [0., 0., 0.])
        new = deepcopy(old); new.update(controller='commitment_contact_round_trip_controller_v1', commitment_contact_policy_enabled=True)
        if i == 3: new.update(new_selection=deepcopy(new_selection), selected_action='right_arc', requested_command=[.16, 0., -.45])
        check = comparator.compare(old, new, old['requested_command'], frame=i)
        originals.append(dict(tick=i, observation_index=i, pre_sample_index=749+50*i, decision=old))
        saved.append(dict(tick=i, decision=new, comparison=check, original_requested_command=old['requested_command'], public_input_arrays_unchanged=True))
    report = dict(case=prefix.CASE, layout_index=1, frames=4, maximum_frames=MAX_FRAMES,
        first_requested_command_difference=3, raw_model_forecast_comparisons=1,
        final_requested_command=[.16, 0., -.45], prior_requested_command=[0., 0., -.45],
        final_terminal=None, prior_terminal=None, final_failure=None, boundary_comparison=check,
        complete_selection_transform_verified_every_frame=True, original_actual_commands_before_intervention_exact=True,
        all_shared_observed_state_exact=True, stopped_at_first_changed_command_or_either_terminal=True,
        following_recorded_observations_consumed=False, public_input_arrays_unchanged=True,
        model_state_sha256=prefix.SUPERVISED_STATE, model_state_unchanged=True,
        scored_pose_horizon_ns=100_000_000, scored_contact_horizon_ns=100_000_000, path_constraint_horizon_ns=800_000_000,
        contact_scores_calibrated=False, unexecuted_outcomes_inferred=False, native_execution=False, navigation_verified=False)
    result = dict(status='SUPERVISED_COMMITMENT_CONTACT_PREFIX_V1_COMPLETE', diagnostic_result_sha256=prefix.DIAGNOSTIC_SHA,
        model_loaded=True, model_training=False, native_execution=False, shadow_replay_only=True, report=report)
    return result, originals, saved


@pytest.mark.parametrize('fault', [None, 'short', 'extra', 'model', 'future', 'terminal', 'boundary', 'forecast_count',
    'old_state', 'new_state', 'wrong_diagnostic', 'boolean_frame'])
def test_admission_reconstructs_every_actual_prefix_comparison(monkeypatch, fault):
    result, old, saved = evidence(); report = result['report']
    if fault == 'short': saved.pop()
    elif fault == 'extra': saved.append(deepcopy(saved[-1]))
    elif fault == 'model': report['model_state_sha256'] = '0'*64
    elif fault == 'future': report['following_recorded_observations_consumed'] = True
    elif fault == 'terminal': report['final_terminal'] = 'SENSOR_OR_MODEL_FAILURE'
    elif fault == 'boundary': report['first_requested_command_difference'] = 2
    elif fault == 'forecast_count': report['raw_model_forecast_comparisons'] = 2
    elif fault == 'old_state': old[2]['decision']['evidence']['observed'] = -1
    elif fault == 'new_state': saved[2]['decision']['evidence']['observed'] = -1
    elif fault == 'wrong_diagnostic': result['diagnostic_result_sha256'] = '0'*64
    elif fault == 'boolean_frame': report['layout_index'] = True
    monkeypatch.setattr(prefix, 'read_rows', lambda p:iter(saved if p is None else old))
    if fault is None: assert prefix.admit_prefix(None, result) == report
    else:
        with pytest.raises(ValueError): prefix.admit_prefix(None, result)


@pytest.mark.parametrize('fault', [None, 'physics', 'future_physics', 'public', 'decision', 'prior_command',
    'boundary_command', 'incomplete_boundary', 'short_physics', 'short_decisions'])
def test_physical_comparison_stops_before_different_command_future(monkeypatch, tmp_path, fault):
    result, old, saved = evidence(); report = result['report']; paths = [tmp_path/n for n in ('old', 'new', 'saved')]
    for p in paths: p.mkdir()
    oldpath, newpath, savedpath = paths
    rows = {oldpath: old, newpath:[dict(tick=i, observation_index=i, pre_sample_index=749+50*i, decision=deepcopy(r['decision']))
        for i, r in enumerate(saved)], savedpath:saved}
    data = dict(timestamp_s=np.arange(950)*.002, base_pose_world=np.zeros((950, 7)))
    for p in (oldpath, newpath): np.savez(p/'physics_trace.npz', **data)
    tapes = {p:[dict(requested_command=r['decision']['requested_command'], completed=True) for r in rows[p]] for p in (oldpath, newpath)}
    if fault == 'physics': data['base_pose_world'][899, 0] = .1; np.savez(newpath/'physics_trace.npz', **data)
    elif fault == 'future_physics': data['base_pose_world'][900:, 0] = 100.; np.savez(newpath/'physics_trace.npz', **data)
    elif fault == 'decision': rows[newpath][3]['decision']['evidence']['observed'] = -1
    elif fault == 'prior_command': tapes[newpath][1]['requested_command'] = [.2, 0., 0.]
    elif fault == 'boundary_command': tapes[newpath][3]['requested_command'] = [0., 0., 0.]
    elif fault == 'incomplete_boundary': tapes[newpath][3]['completed'] = False
    elif fault == 'short_physics': np.savez(newpath/'physics_trace.npz', **{k:v[:899] for k, v in data.items()})
    elif fault == 'short_decisions': rows[newpath].pop()
    reads = []; packets = []
    def read_rows(p):
        for i, row in enumerate(rows[p]): reads.append((p, i)); yield row
        if fault != 'short_decisions': pytest.fail('consumed following recorded decision')
    def reader(p):
        def packet(i):
            packets.append((p, i))
            return dict(frame=i, changed=fault == 'public' and p == newpath and i == 3), {}, {}, 1_500_000_000+100_000_000*i
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix, 'read_rows', read_rows); monkeypatch.setattr(prefix, 'IntentReturnRGBDReplay', reader)
    monkeypatch.setattr(prefix, 'read_json', lambda p,n:tapes[p] if n == 'command_tape.json' else [{}]*4)
    monkeypatch.setattr(prefix, 'packet', lambda *a,**k:({}, {})); monkeypatch.setattr(prefix, 'public_acquisition', lambda r:r)
    if fault in (None, 'future_physics'):
        check = prefix.compare(oldpath, newpath, savedpath, report)
        assert check['physical_prefix_samples'] == 900 and check['common_prefix_frames'] == 4
        assert check['raw_model_forecast_comparisons'] == 1 and check['candidate_intervention_command_completed']
        assert len(reads) == 12 and len(packets) == 8 and not check['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError): prefix.compare(oldpath, newpath, savedpath, report)
