"""Cohort admission, failure accounting and fresh-model execution without a scene."""
from copy import deepcopy
import json
import sys
import pytest
from lewm.independent_floor_transport_study_development import (
    MODEL_STATE, LAYOUTS, planned_cases, admit_predecessor, independent_scope, require_resources)
from scripts import run_go2_independent_floor_transport_mazes_v1 as runner


def evidence():
    audit = dict(layout_index=0, raw_sensor_reconstruction_pass=True,
        raw_model_command_replay_pass=True, raw_command_audit_pass=True,
        model_state_unchanged=True, verified_round_trip=False,
        native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=False, hard_measurement_failed_frames=[],
        renderer_capture_audit=dict(all_witnesses_match_raw_acquisitions=True))
    record = deepcopy(audit) | dict(case=runner.CASE[0],
        status='MEASURED_FLOOR_TRANSPORT_MAZE_COLLECTED_AND_RAW_AUDITED',
        prefix_comparison=dict(physical_and_public_prefix_exact=True,
            complete_candidate_decisions_match_prospective_prefix=True))
    result = dict(status='MEASURED_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE', conditions=[record],
        artifact_sha256={'launch.json':'a'*64})
    launch = dict(planned_case=list(runner.CASE), prefix_report=dict(model_state_sha256=MODEL_STATE),
        implementation_class='MeasuredFloorTransportController', renderer_capture_witnesses_enabled=True,
        additional_auxiliary_rgb_for_motion=True, source_sha256={}, correction_admission={'fixture':True})
    for key in ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
            'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment',
            'robot_urdf_path', 'robot_urdf_sha256'):
        launch[key] = {}
    return result, audit, launch


@pytest.mark.parametrize('fault', [None, 'incomplete', 'multiple', 'layout', 'wrong_case',
    'prefix', 'replay', 'raw', 'command', 'model', 'changed_weights', 'controller',
    'renderer', 'rgb', 'audit_disagreement'])
def test_completed_negative_predecessor_admitted_but_infrastructure_and_identity_faults_rejected(fault):
    result, audit, launch = evidence(); record = result['conditions'][0]
    if fault == 'incomplete': result['status'] = 'RUNNING'
    elif fault == 'multiple': result['conditions'].append(deepcopy(record))
    elif fault == 'layout': record['layout_index'] = 1
    elif fault == 'wrong_case': launch['planned_case'][0] = 'other'
    elif fault == 'prefix': record['prefix_comparison']['physical_and_public_prefix_exact'] = False
    elif fault == 'replay': audit['raw_model_command_replay_pass'] = False
    elif fault == 'raw': audit['raw_sensor_reconstruction_pass'] = False
    elif fault == 'command': audit['raw_command_audit_pass'] = False
    elif fault == 'model': record['model_state_unchanged'] = False
    elif fault == 'changed_weights': launch['prefix_report']['model_state_sha256'] = '0'*64
    elif fault == 'controller': launch['implementation_class'] = 'DifferentController'
    elif fault == 'renderer': launch['renderer_capture_witnesses_enabled'] = False
    elif fault == 'rgb': launch['additional_auxiliary_rgb_for_motion'] = False
    elif fault == 'audit_disagreement': record['verified_round_trip'] = True
    if fault is not None:
        with pytest.raises(ValueError): admit_predecessor(result, audit, launch, runner.CASE)
    else:
        admitted = admit_predecessor(result, audit, launch, runner.CASE)
        assert admitted['predecessor_verified_round_trip'] is False
        assert admitted['predecessor_strict_physical_visibility_pass'] is False
        assert [c[1] for c in admitted['planned_cases']] == [1, 2, 3]
        assert all(c[2:] == tuple(runner.CASE[2:]) for c in admitted['planned_cases'])


@pytest.mark.parametrize('index', [1, 2, 3])
def test_scope_wrapper_preserves_every_scientific_field_and_input(index):
    report = evidence()[1] | dict(layout_index=index,
        independent_layout_development_execution=False, reused_development_layout=True)
    original = deepcopy(report); changed = independent_scope(report, index)
    assert report == original
    assert changed.pop('independent_layout_development_execution') is True
    assert changed.pop('reused_development_layout') is False
    original.pop('independent_layout_development_execution'); original.pop('reused_development_layout')
    assert changed == original


@pytest.mark.parametrize('index', [0, 4, True, 1.0])
def test_scope_rejects_outside_cohort(index):
    with pytest.raises(ValueError): independent_scope({'layout_index':index}, index)


@pytest.mark.parametrize('remaining', [1, 2, 3])
def test_resources_cover_all_remaining_cases_and_preserve_reserve(remaining):
    required = 40 + remaining*11
    resources = dict(memory_available_bytes=32*1024**3, artifact_free_bytes=required)
    allowance = require_resources(resources, remaining, reserve=40, collection=10, persistence=1)
    assert allowance['required_free_bytes'] == required
    for key in resources:
        short = resources | {key:resources[key]-1}
        with pytest.raises(ValueError): require_resources(short, remaining, reserve=40, collection=10, persistence=1)


def prepare_main(monkeypatch, tmp_path, *, failure_index=None, resource_failure=False):
    result, audit, old = evidence(); submitted = []; pools = []; admissions = []
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path/'output')
    monkeypatch.setattr(runner, 'validate_root', lambda *a, **k:None)
    monkeypatch.setattr(runner, 'verify_artifacts', lambda *a, **k:None)
    monkeypatch.setattr(runner, 'verify_predecessor', lambda *a, **k:None)
    monkeypatch.setattr(runner, 'verify_inputs', lambda *a, **k:None)
    monkeypatch.setattr(runner, 'discover_sources', lambda *a, **k:{'fixture':'a'*64})
    monkeypatch.setattr(runner, 'create_output', lambda p:p.mkdir())
    monkeypatch.setattr(runner, 'read_json', lambda p,n: result if n == 'result.json' else old if n == 'launch.json' else audit)
    def hardware():
        return dict(memory_available_bytes=80*1024**3, artifact_free_bytes=100*1024**3)
    monkeypatch.setattr(runner, 'hardware', hardware)
    original_resources = runner.resources_for
    def admit(resources, remaining):
        admissions.append(remaining)
        if resource_failure and remaining == 2: raise ValueError('synthetic resource exhaustion')
        return original_resources(resources, remaining)
    monkeypatch.setattr(runner, 'resources_for', admit)
    class Future:
        def __init__(self, record): self.record = record
        def result(self): return self.record
    class Pool:
        def __init__(self, **kwargs): pools.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, function, case, launch_sha):
            assert function is runner.worker
            submitted.append(case)
            record = dict(case=case[0], layout_index=case[1], artifact_sha256={}, verified_round_trip=False,
                status='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED')
            if case[1] == failure_index:
                record.update(status='INDEPENDENT_FLOOR_TRANSPORT_WORKER_FAILED', failure='synthetic raw replay failure')
            (runner.OUTPUT/(case[0]+'_worker.log')).write_text('synthetic worker\n')
            runner.write_json(runner.OUTPUT/(case[0]+'_worker_terminal.json'), record)
            return Future(record)
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(runner, 'wait', lambda futures, **kwargs:(set(futures), set()))
    monkeypatch.setattr(sys, 'argv', ['runner', '--native-result-sha256', 'b'*64])
    return submitted, pools, admissions


def test_main_keeps_all_negative_outcomes_and_fresh_process_configuration(monkeypatch, tmp_path):
    submitted, pools, admissions = prepare_main(monkeypatch, tmp_path)
    runner.main()
    result = json.loads((runner.OUTPUT/'result.json').read_text())
    assert [c[1] for c in submitted] == [1, 2, 3]
    assert len(pools) == 1 and pools[0]['max_workers'] == pools[0]['max_tasks_per_child'] == 1
    assert pools[0]['mp_context'].get_start_method() == 'spawn'
    assert admissions == [3, 3, 3, 2, 1]
    assert result['new_independent_layout_executions'] == 3
    assert result['measured_round_trip_successes'] == 0 and result['all_fixed_cases_executed']
    assert not result['goal_achieved'] and not result['navigation_qualified']
    assert len(result['conditions']) == 3 and not (runner.OUTPUT/'failure.json').exists()
    for count in (1, 2, 3):
        name = f'cohort_progress_after_{count:02d}.json'
        progress = json.loads((runner.OUTPUT/name).read_text())
        assert len(progress['completed_conditions']) == count
        assert progress['remaining_layouts'] == list(LAYOUTS[count:])
        assert result['artifact_sha256'][name] == runner.digest(runner.OUTPUT/name)


@pytest.mark.parametrize('resource_failure', [False, True])
def test_partial_cohort_retained_without_retry_or_skip(monkeypatch, tmp_path, resource_failure):
    submitted, _, _ = prepare_main(monkeypatch, tmp_path,
        failure_index=None if resource_failure else 2, resource_failure=resource_failure)
    with pytest.raises(ValueError): runner.main()
    failure = json.loads((runner.OUTPUT/'failure.json').read_text())
    assert [c[1] for c in submitted] == ([1] if resource_failure else [1, 2])
    assert len(failure['completed_conditions']) == len(submitted)
    assert failure['automatic_retry'] is False and failure['original_case_order'] == [1, 2, 3]
    assert not (runner.OUTPUT/'result.json').exists()
    for c in submitted: assert (runner.OUTPUT/(c[0]+'_worker_terminal.json')).is_file()


def test_preflight_does_not_create_outputs_or_spawn(monkeypatch, tmp_path):
    submitted, pools, _ = prepare_main(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, 'argv', sys.argv+['--preflight-only'])
    runner.main()
    assert not runner.OUTPUT.exists() and not submitted and not pools


@pytest.mark.parametrize('index', [1, 2, 3])
def test_worker_uses_new_model_for_collection_and_raw_replay_of_each_layout(monkeypatch, tmp_path, index):
    case = planned_cases(runner.CASE)[index-1]
    launch = dict(planned_cases=[list(c) for c in planned_cases(runner.CASE)],
        robot_urdf_sha256='c'*64, correction_admission={'fixture':True}, source_sha256={runner.PROTOCOL:'d'*64})
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'read_json', lambda *a:launch)
    monkeypatch.setattr(runner, 'verify_artifacts', lambda *a:None)
    monkeypatch.setattr(runner, 'verify_inputs', lambda *a:None)
    original_digest = runner.digest
    monkeypatch.setattr(runner, 'digest', lambda p:'c'*64 if p == runner.URDF else original_digest(p))
    monkeypatch.setattr(runner, 'state_digest', lambda state:MODEL_STATE)
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda *a:object())
    models = []; calls = []
    class Model:
        def state_dict(self): return {}
    def load(admission, model_name):
        assert admission == launch['correction_admission'] and model_name == case[4]
        models.append(Model()); return models[-1], case[3], case[2]
    monkeypatch.setattr(runner, 'load_assigned', load)
    def collect(i, definition, **kwargs):
        assert i == index and definition == 'd'*64 and kwargs['model'] is models[0]
        assert kwargs['episode_name'] == case[0] and kwargs['condition'] == case[3]
        calls.append('collect'); return {'fixture_collection':index}
    def audit(i, result, definition, **kwargs):
        assert i == index and result == {'fixture_collection':index} and definition == 'd'*64
        assert kwargs['model'] is models[1] and models[1] is not models[0]
        assert kwargs['episode_name'] == case[0] and kwargs['condition'] == case[3]
        calls.append('audit'); return evidence()[1] | dict(layout_index=index,
            independent_layout_development_execution=False, reused_development_layout=True)
    monkeypatch.setattr(runner, 'collect', collect); monkeypatch.setattr(runner, 'audit', audit)
    monkeypatch.setattr(runner, 'artifacts', lambda *a:[])
    terminal = runner.worker(case, 'a'*64)
    assert terminal['status'] == 'INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
    assert calls == ['collect', 'audit'] and len(models) == 2
    assert terminal['verified_round_trip'] is False and terminal['layout_index'] == index
    saved_audit = json.loads((tmp_path/(case[0]+'_audit.json')).read_text())
    assert saved_audit['independent_layout_development_execution'] is True
    assert saved_audit['reused_development_layout'] is False
