import pytest
from scripts import await_go2_commitment_contact_anchored_raw_prefix_v1 as wait


def test_waits_for_same_worker_then_returns_only_completed_identity(monkeypatch):
    active = iter([True, True, False]); seen = []; sleeps = []
    def live(owner):
        seen.append(owner)
        return next(active) if owner is wait.WORKER else True
    monkeypatch.setattr(wait, 'owner_live', live)
    monkeypatch.setattr(wait, 'completed_worker_identity', lambda: 'complete-sha')
    events = []
    assert wait.wait_for_worker(lambda *a, **k: events.append((a,k)), sleep=sleeps.append,
        clock=lambda: 0) == 'complete-sha'
    assert sleeps == [30, 30] and len(events) == 2
    assert seen == [wait.WORKER, wait.BATCH_OWNER, wait.WORKER, wait.BATCH_OWNER, wait.WORKER]


def test_missing_terminal_after_owner_ends_is_failure_not_restart(monkeypatch):
    monkeypatch.setattr(wait, 'owner_live', lambda owner: False)
    def incomplete(): raise ValueError('no terminal')
    monkeypatch.setattr(wait, 'completed_worker_identity', incomplete)
    with pytest.raises(ValueError, match='no terminal'):
        wait.wait_for_worker(lambda *a, **k: None, sleep=lambda n: pytest.fail('no sleep'), clock=lambda: 0)


def test_parent_loss_stops_handoff(monkeypatch):
    monkeypatch.setattr(wait, 'owner_live', lambda owner: owner is wait.WORKER)
    with pytest.raises(ValueError, match='parent'):
        wait.wait_for_worker(lambda *a, **k: None, sleep=lambda n: pytest.fail('no sleep'), clock=lambda: 0)


def test_timeout_never_restarts_original_or_launches_child(monkeypatch):
    monkeypatch.setattr(wait, 'owner_live', lambda owner: True)
    times = iter([0, wait.WAIT_SECONDS])
    with pytest.raises(ValueError, match='expired'):
        wait.wait_for_worker(lambda *a, **k: None, sleep=lambda n: pytest.fail('no sleep'), clock=lambda: next(times))


def test_changed_owner_identity_is_not_treated_as_completion(monkeypatch):
    def changed(owner): raise ValueError('identity changed')
    monkeypatch.setattr(wait, 'owner_live', changed)
    monkeypatch.setattr(wait, 'completed_worker_identity', lambda: pytest.fail('cannot admit replacement'))
    with pytest.raises(ValueError, match='identity changed'):
        wait.wait_for_worker(lambda *a, **k: None, clock=lambda: 0)


@pytest.mark.parametrize('fault', [None, 'model', 'pending', 'observed_state', 'frames',
    'command', 'future', 'source', 'worker'])
def test_completion_requires_exact_case_state_boundary_and_source(monkeypatch, tmp_path, fault):
    report = dict(frames=4, first_changed_command_frame=3, raw_model_forecast_comparisons=1,
        model_state_sha256=wait.replay.MODEL_SHA, candidate_requested_command=[.2,0.,0.],
        original_requested_command=[0.,0.,.45], no_observation_after_changed_request_consumed=True,
        complete_retained_observed_state_exact=True, selected_pending_forecasts_checked=True)
    result = dict(status='COMMITMENT_CONTACT_ANCHORED_RAW_PREFIX_V1_COMPLETE',
        artifact_sha256={}, original_worker_terminal_sha256='worker', source_sha256={'source':'sha'},
        report=report, native_execution=False)
    launch = dict(source_sha256={'source':'sha'}, input_admission={'original_worker_terminal_sha256':'worker'},
        saved_selection_result_sha256=wait.replay.SAVED_SHA)
    if fault == 'model': report['model_state_sha256'] = 'other'
    elif fault == 'pending': report['selected_pending_forecasts_checked'] = False
    elif fault == 'observed_state': report['complete_retained_observed_state_exact'] = False
    elif fault == 'frames': report['frames'] = 3
    elif fault == 'command': report['candidate_requested_command'] = [0.,0.,0.]
    elif fault == 'future': report['no_observation_after_changed_request_consumed'] = False
    elif fault == 'source': launch['source_sha256'] = {'source':'changed'}
    elif fault == 'worker': result['original_worker_terminal_sha256'] = 'other'
    monkeypatch.setattr(wait.replay, 'OUTPUT', tmp_path)
    monkeypatch.setattr(wait, 'digest', lambda path: 'result')
    monkeypatch.setattr(wait, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(wait, 'verify', lambda *args: None)
    monkeypatch.setattr(wait, 'read_json', lambda root,name: result if name == 'result.json' else launch)
    if fault:
        with pytest.raises(ValueError): wait.authenticate_replay('worker', {'source':'sha'})
    else:
        assert wait.authenticate_replay('worker', {'source':'sha'})['raw_replay_result_sha256'] == 'result'
