from itertools import islice
import pytest
from scripts import replay_go2_receipt_copied_footprint_late_history_v1 as run
from lewm import receipt_copied_footprint_development as copied
from lewm import packed_fused_scoped_controller_development as packed
from lewm.tests import test_packed_fused_scoped_replay_development as previous


def fixture(monkeypatch,tmp_path,fault=None):
    rows,prior,tape,calls,models=previous.fixture(monkeypatch,tmp_path,fault)
    fake=previous.replay.PackedFusedScopedController
    class Baseline(fake):index=0
    class Candidate(fake):
        index=1
        def observe(self,*args,**kwargs):
            return super().observe(*args,**kwargs)|{'controller':copied.CONTROLLER,copied.FLAG:True}
    monkeypatch.setattr(run,'PackedFusedScopedController',Baseline)
    monkeypatch.setattr(run,'ReceiptCopiedFootprintController',Candidate)
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'normalized_state_tree',run.original.state_tree)
    for i,row in enumerate(islice(run.original.profile.read_rows(None),1428)):
        decision=row['decision']|{run.original.FLAG:True,'controller':packed.CONTROLLER,
            previous.fused.FLAG:True,previous.combined.FLAG:True,packed.FLAG:True,
            'batched_retained_floor_queries_enabled':True}
        rows[i]['candidate_decision_sha256']=run.original.profile.reference.saved.identity(decision)
    return rows,prior,tape,calls,models


def test_original_loop_and_only_declared_composition_changes():
    old=run.original.replay;before=dict(old.__globals__);new=run.isolated_replay()
    assert new.__code__ is old.__code__ and new.__closure__ is old.__closure__
    assert new.__globals__['FrozenFootprintAnchoredController'] is run.PackedFusedScopedController
    assert new.__globals__['ScopedFootprintAnchoredController'] is run.ReceiptCopiedFootprintController
    assert all(old.__globals__[k] is v for k,v in before.items())


def test_full_history_and_seven_original_state_checks(monkeypatch,tmp_path):
    rows,prior,_,calls,models=fixture(monkeypatch,tmp_path)
    result=run.replay(rows,prior)
    assert calls==[(i,j) for i in range(1428) for j in run.original.execution_order(i)]
    assert models[0] is not models[1]
    assert result['frames']==1428 and result['raw_model_forecast_comparisons']==1425
    assert result['observed_state_checks']==prior['observed_state_checks']
    assert result['baseline']=='PackedFusedScopedController'
    assert result['candidate']=='ReceiptCopiedFootprintController'
    assert result['persistent_memory_type_unchanged']


@pytest.mark.parametrize('fault',['receipt','input','metadata','terminal','state','gradient',
                                  'weight','shared_model','shared_storage'])
def test_original_corruption_rejections(monkeypatch,tmp_path,fault):
    rows,prior,_,_,_=fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):run.replay(rows,prior)


@pytest.mark.parametrize('fault',['endpoint','input_hash','original_hash','baseline_hash','prior_state'])
def test_reference_binding_rejections(monkeypatch,tmp_path,fault):
    rows,prior,tape,_,_=fixture(monkeypatch,tmp_path)
    if fault=='endpoint':tape[1001]['post_sample_index']+=1
    elif fault=='prior_state':prior['observed_state_checks'][-1]['state_sha256']='f'*64
    else:rows[1001][dict(input_hash='public_input_sha256',original_hash='original_decision_sha256',
                         baseline_hash='candidate_decision_sha256')[fault]]='wrong'
    with pytest.raises(ValueError):run.replay(rows,prior)
