import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from scripts import external_body_projection_profile_replay_development as run
from lewm.tests import test_frozen_footprint_late_history_profile_development as fixture_source
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree


def test_exact_original_replay_derivative_keeps_all_undeclared_logic():
    manifest=json.loads(Path('docs/go2_external_body_projection_replay_source_derivative_2026-09-11.json').read_text())
    source=Path(manifest['source']).read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==manifest['source_sha256']
    old=next(n for n in ast.parse(source).body if getattr(n,'name',None)==manifest['source_function'])
    expected=ast.get_source_segment(source,old)
    for before,after in manifest['replacements']:
        assert expected.count(before)==1
        expected=expected.replace(before,after)
    target=ast.parse(Path(manifest['target']).read_text())
    actual=next(n for n in target.body if getattr(n,'name',None)==manifest['target_function'])
    assert ast.dump(actual)==ast.dump(ast.parse(expected).body[0])


def fixture(monkeypatch,tmp_path,fault=None):
    called,normalized,decisions,tape=fixture_source.synthetic(monkeypatch,tmp_path,fault)
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'BodyProjectedTiledController',run.original.FrozenFootprintAnchoredController)
    monkeypatch.setattr(run,'normalize_candidate',run.original.normalize_candidate)
    monkeypatch.setattr(run,'state_tree',state_tree)
    rows=[]
    for frame,decision in enumerate(decisions[:1428]):
        candidate=deepcopy(decision)|dict(controller=run.original.previous.completed_replay.CONTROLLER,
            **{run.original.previous.completed_replay.FLAG:True})
        rows.append(dict(frame=frame,original_decision_sha256=run.original.reference.saved.identity(decision),
            candidate_decision_sha256=run.original.reference.saved.identity(candidate),
            public_input_sha256=run.original.fingerprint((
                fixture_source.np.array([frame]),fixture_source.np.array([frame]),fixture_source.np.array([frame]),
                fixture_source.np.array([2]),fixture_source.np.array([1]),frame))))
    states=[]
    for frame in run.STATE_FRAMES:
        state=dict(memory=SimpleNamespace(index={'synthetic':True}),floor={},occupied={},
            residual=SimpleNamespace(pending=None),history=list(range(frame+1)))
        states.append(dict(frame=frame,state_sha256=run.original.fingerprint(state_tree(state)),
            retained_observed_state_equal=True))
    return called,normalized,rows,dict(observed_state_checks=states),tape


def test_all_history_decisions_and_seven_independent_state_witnesses_match(monkeypatch,tmp_path):
    called,normalized,rows,prior,_=fixture(monkeypatch,tmp_path)
    hooks=(sys.getprofile(),sys.gettrace())
    report=run.replay(rows,prior)
    assert called==list(range(1428)) and len(normalized)==1428
    assert report['raw_model_forecast_comparisons']==1425
    assert report['observed_state_checks']==prior['observed_state_checks']
    assert report['retained_state_identity_established']
    assert (sys.getprofile(),sys.gettrace())==hooks
    assert report['external_profiler_covers_entire_child'] and not report['external_profile_file_verified']
    assert not report['controller_observe_only_profiled'] and report['sensor_acquisition_profiled']
    assert not report['normalization_outside_profiled_region']
    assert report['normalization_outside_marked_controller_windows']
    assert not report['cprofile_hooks_enabled'] and not report['speedup_established']
    assert list(tmp_path.iterdir())==[tmp_path/'comparison.jsonl']
    assert [len(window['python_stack_markers']) for window in report['windows'].values()]==[10,10,10]


@pytest.mark.parametrize('fault',['receipt','input','metadata','gradient','terminal'])
def test_original_input_model_and_decision_failures_remain_terminal(monkeypatch,tmp_path,fault):
    called,_,rows,prior,_=fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):run.replay(rows,prior)
    assert called==list(range(1428 if fault=='gradient' else 1002))


@pytest.mark.parametrize('fault',['endpoint','prior_state','prior_candidate','prior_input'])
def test_reference_or_endpoint_changes_cannot_pass(monkeypatch,tmp_path,fault):
    _,_,rows,prior,tape=fixture(monkeypatch,tmp_path)
    if fault=='endpoint':tape[1001]['post_sample_index']+=1
    elif fault=='prior_state':prior['observed_state_checks'][-1]['state_sha256']='bad'
    elif fault=='prior_candidate':rows[-1]['candidate_decision_sha256']='bad'
    else:rows[-1]['public_input_sha256']='bad'
    with pytest.raises(ValueError):run.replay(rows,prior)


def test_each_marked_observation_has_one_distinct_live_stack_marker():
    markers=[]
    def observe(value):
        markers.append(sys._getframe(1).f_code.co_name)
        return value
    for frame in range(1428):
        assert run.observe_window(frame,run.MARKER_WINDOWS.get(frame),observe,frame)==frame
        assert markers[-1]==run.MARKERS.get(frame,'observe_window')
    assert len(set(markers)-{'observe_window'})==30
    with pytest.raises(ValueError):run.observe_window(3,'late_navigation',observe,3)
    with pytest.raises(ValueError):run.observe_window(True,None,observe,3)
