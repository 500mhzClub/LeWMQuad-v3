"""Synthetic collection bytes and source-bound receipt/process checks only."""
from copy import deepcopy
import json
import os
import pytest

from scripts import independent_round_trip_collection_handoff_development as handoff
from lewm.tests.test_independent_round_trip_population_case_evidence_development import saved, persist
from lewm.independent_round_trip_comparison_study_development import CASES
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest


@pytest.fixture
def setup(saved,monkeypatch):
    output,collections,reports,_=saved
    launch=dict(synthetic_fixture=True,source_sha256={n:digest(ROOT/n)
        for n in (handoff.SOURCE,handoff.TEST,handoff.PROTOCOL)})
    (output/'launch.json').write_text(json.dumps(launch)+'\n')
    sha=digest(output/'launch.json')
    def checked(root,expected,verifier):
        handoff.verify_artifacts(root,{'launch.json':expected})
        return deepcopy(launch)
    monkeypatch.setattr(handoff.runtime,'checked_launch',checked)
    for case in CASES:
        (output/(case.name+handoff.LOG_SUFFIX)).write_text('synthetic closed collector log\n')
    return output,collections,reports,sha


def publish(setup,index=0,reference=None):
    output,collections,_,sha=setup
    record=handoff.publish(output,CASES[index],collections[index],sha,reference,None)
    return record,digest(output/(CASES[index].name+handoff.SUFFIX))


def consume(setup,sha,*,index=0,reference=None):
    return handoff.read_for_audit(setup[0],CASES[index],sha,setup[3],reference,None)


def test_published_receipt_is_actual_current_process_and_not_audit_completion(setup):
    record,sha=publish(setup)
    assert record['collection_owner']['pid']==os.getpid()
    assert record['status']==handoff.STATUS and record['collection_returned']
    assert not any(record[k] for k in ('raw_audit_completed','audited_episode_complete',
        'native_scene_ownership_released','parent_verified_zero_exit','audit_execution_permitted'))
    with pytest.raises(ValueError,match='collector remains live'):consume(setup,sha)


def test_ended_owner_and_intact_collection_still_require_parent_exit_and_actual_audit(setup,monkeypatch):
    record,sha=publish(setup)
    monkeypatch.setattr(handoff,'owner_live',lambda owner:False)
    result=consume(setup,sha)
    assert result['collection_handoff']==record and result['original_collector_ended']
    assert result['complete_collection_bindings_verified']
    assert not any(result[k] for k in ('raw_audit_completed','audited_episode_complete',
        'parent_verified_zero_exit','audit_execution_permitted'))
    assert not (setup[0]/(CASES[0].name+'_worker_terminal.json')).exists()


def test_live_owner_rejected_before_launch_and_reference_reverification(setup,monkeypatch):
    _,sha=publish(setup)
    monkeypatch.setattr(handoff,'checked_launch',lambda *a:pytest.fail('live owner must reject first'))
    with pytest.raises(ValueError,match='collector remains live'):consume(setup,sha)


def test_changed_process_identity_not_treated_as_completion(setup,monkeypatch):
    _,sha=publish(setup)
    def reused(owner):raise ValueError('original owner process identity changed')
    monkeypatch.setattr(handoff,'owner_live',reused)
    with pytest.raises(ValueError,match='identity changed'):consume(setup,sha)


def test_open_collector_log_cannot_be_bound(setup):
    with (setup[0]/(CASES[0].name+handoff.LOG_SUFFIX)).open('a') as log:
        log.write('still open');log.flush()
        with pytest.raises(ValueError,match='log must be closed'):publish(setup)


@pytest.mark.parametrize('suffix',[handoff.SUFFIX,handoff.FAILURE_SUFFIX,'_worker_terminal.json'])
def test_existing_or_failed_collection_cannot_be_republished(setup,suffix):
    path=setup[0]/(CASES[0].name+suffix);path.write_text('preserve original evidence')
    with pytest.raises(ValueError,match='preserve failure'):publish(setup)
    assert path.read_text()=='preserve original evidence'


@pytest.mark.parametrize('fault',['data','log','receipt','failure','launch'])
def test_post_publication_tampering_or_failure_rejected(setup,monkeypatch,fault):
    _,sha=publish(setup);monkeypatch.setattr(handoff,'owner_live',lambda owner:False)
    output=setup[0];case=CASES[0]
    name={'data':case.name+'/command_tape.json','log':case.name+handoff.LOG_SUFFIX,
        'receipt':case.name+handoff.SUFFIX,'failure':case.name+handoff.FAILURE_SUFFIX,'launch':'launch.json'}[fault]
    (output/name).write_text('changed or failed')
    with pytest.raises(ValueError):consume(setup,sha)


@pytest.mark.parametrize('fault',['missing','extra','collection','source','log_hash','case','reference',
    'boot','audit_claim','permission','exit_claim','released_claim','owner'])
def test_rehashed_receipt_cannot_hide_inconsistent_evidence(setup,monkeypatch,fault):
    record,_=publish(setup);monkeypatch.setattr(handoff,'owner_live',lambda owner:False)
    case=CASES[0]
    if fault=='missing':record['artifact_sha256'].pop(case.name+'/command_tape.json')
    elif fault=='extra':record['artifact_sha256']['unexpected.json']='a'*64
    elif fault=='collection':record['collection']['physical_stop']='different'
    elif fault=='source':record['source_sha256'][handoff.SOURCE]='a'*64
    elif fault=='log_hash':record['collection_log_sha256']='a'*64
    elif fault=='case':record['case']=CASES[1].name
    elif fault=='reference':record['reference_worker_sha256']='a'*64
    elif fault=='boot':record['boot_id']='different'
    elif fault=='audit_claim':record['raw_audit_completed']=True
    elif fault=='permission':record['audit_execution_permitted']=True
    elif fault=='exit_claim':record['parent_verified_zero_exit']=True
    elif fault=='released_claim':record['native_scene_ownership_released']=True
    else:record['collection_owner']['pid']=True
    path=setup[0]/(case.name+handoff.SUFFIX);path.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError):consume(setup,digest(path))


def test_later_arm_requires_completed_original_reference(setup,monkeypatch):
    with pytest.raises(ValueError,match='completed fixed reference'):publish(setup,index=1)
    _,reference=persist(setup,0)
    _,sha=publish(setup,index=1,reference=reference)
    monkeypatch.setattr(handoff,'owner_live',lambda owner:False)
    assert consume(setup,sha,index=1,reference=reference)['complete_collection_bindings_verified']
    (setup[0]/(CASES[0].name+'/command_tape.json')).write_text('changed original reference')
    with pytest.raises(ValueError):consume(setup,sha,index=1,reference=reference)


def test_changed_identity_during_admission_cannot_pass(setup,monkeypatch):
    _,sha=publish(setup);states=iter([False,True])
    monkeypatch.setattr(handoff,'owner_live',lambda owner:next(states))
    with pytest.raises(ValueError,match='ownership changed'):consume(setup,sha)


def test_handoff_requires_frozen_implementation_in_launch(setup,monkeypatch):
    monkeypatch.setattr(handoff.runtime,'checked_launch',lambda *a:{'source_sha256':{}})
    with pytest.raises(ValueError,match='must be frozen'):publish(setup)
