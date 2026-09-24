"""Full synthetic history and inherited input/model/state corruption checks."""
from itertools import islice
from types import FunctionType
from scripts import deferred_memo_single_pass_replay_development as run
from lewm import deferred_memo_single_pass_controller_development as candidate
from lewm.tests import test_single_pass_body_projected_replay_development as previous


def fixture(monkeypatch,tmp_path,fault=None):
    rows,prior,tape,calls,models=previous.fixture(monkeypatch,tmp_path,fault)
    fake=previous.run.SinglePassBodyProjectedController
    class Baseline(fake): index=0
    class Candidate(Baseline):
        index=1
        def observe(self,*args,**kwargs):
            return super().observe(*args,**kwargs) | {'controller':candidate.CONTROLLER,candidate.FLAG:True}
    monkeypatch.setattr(run,'SinglePassBodyProjectedController',Baseline)
    monkeypatch.setattr(run,'DeferredMemoSinglePassController',Candidate)
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'normalized_state_tree',run.original.state_tree)
    body=previous.previous
    for i,row in enumerate(islice(run.original.profile.read_rows(None),1428)):
        decision=row['decision'] | {run.original.FLAG:True,'controller':previous.candidate.CONTROLLER,
            previous.candidate.FLAG:True,previous.body.FLAG:True,body.visible.FLAG:True,
            body.progressive.FLAG:True,body.density.FLAG:True,body.visibility.FLAG:True,
            body.previous.fused.FLAG:True,body.previous.combined.FLAG:True,
            body.packed_base.FLAG:True,body.packed.FLAG:True,'batched_retained_floor_queries_enabled':True}
        rows[i]['candidate_decision_sha256']=run.original.profile.reference.saved.identity(decision)
    return rows,prior,tape,calls,models


for _name in ('test_original_corruption_rejections','test_reference_binding_rejections'):
    _function=getattr(previous,_name)
    _clone=FunctionType(_function.__code__,_function.__globals__ | dict(run=run,fixture=fixture),
        _name,_function.__defaults__,_function.__closure__)
    _clone.__kwdefaults__=_function.__kwdefaults__;_clone.__dict__.update(_function.__dict__)
    globals()[_name]=_clone


def test_all_1428_frames_independent_models_and_seven_states(monkeypatch,tmp_path):
    rows,prior,_,calls,models=fixture(monkeypatch,tmp_path)
    report=run.replay(rows,prior)
    assert calls==[(i,j) for i in range(1428) for j in run.original.execution_order(i)]
    assert models[0] is not models[1]
    assert report['frames']==1428 and report['raw_model_forecast_comparisons']==1425
    assert report['observed_state_checks']==prior['observed_state_checks']
    assert report['only_two_pure_footprint_copiers_changed']


def test_original_loop_and_imported_globals_unchanged():
    original=run.original.replay;before=original.__globals__.copy();new=run.isolated_replay()
    assert new.__code__ is original.__code__ and new.__closure__ is original.__closure__
    assert new.__globals__['FrozenFootprintAnchoredController'] is run.SinglePassBodyProjectedController
    assert new.__globals__['ScopedFootprintAnchoredController'] is run.DeferredMemoSinglePassController
    assert new.__globals__['profile'].normalize_candidate is run.previous.normalize_candidate
    assert all(original.__globals__[k] is v for k,v in before.items())
