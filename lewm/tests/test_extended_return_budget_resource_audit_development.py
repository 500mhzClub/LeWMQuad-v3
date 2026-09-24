"""Closed resource evidence must match every actual collection/audit boundary."""
import json

import pytest

from scripts import extended_return_budget_resource_audit_development as audit
from lewm.tests.test_extended_return_budget_resource_guard_development import setup


def population(tmp_path,monkeypatch,tail=0):
    setup(tmp_path,monkeypatch);limits=audit.limits
    collection=dict(decisions=3,physical_stop=None,acquisition_stop='PACKET_CONTRACT_STOP' if tail else None)
    for phase in ('collection','audit'):
        guard=limits.ResourceGuard(tmp_path,'synthetic',phase);guard.check('begin')
        if phase=='audit':guard.check('before_sensor_audit');guard.check('after_sensor_audit')
        cycle=('before_packet','after_packet','before_controller','after_controller') if phase=='collection' else ('before_controller','after_controller')
        for frame in range(3):
            for stage in cycle:guard.check(stage,frame)
        if phase=='collection':
            for stage in cycle[:tail]:guard.check(stage,3)
        guard.check('completed');guard.finish()
    return collection


@pytest.mark.parametrize('tail',[0,1,2,3])
def test_complete_and_actual_partially_stopped_acquisition_reconstruct(tmp_path,monkeypatch,tail):
    collection=population(tmp_path,monkeypatch,tail)
    report=audit.check(tmp_path,'synthetic',collection)
    assert report['both_phases_complete'] and not report['between_sample_peak_bounded']
    assert report['collection']['samples']==14+tail and report['audit']['samples']==10
    if tail:
        collection['acquisition_stop']=None
        with pytest.raises(ValueError,match='actual stopped'):audit.check(tmp_path,'synthetic',collection)


@pytest.mark.parametrize('fault',['missing','reordered','frame','boolean_frame','rss','ram','disk','clock',
    'consumption','receipt','count','extra','oversize','initial_admission'])
def test_missing_changed_or_fabricated_closed_resource_evidence_rejects(tmp_path,monkeypatch,fault):
    collection=population(tmp_path,monkeypatch)
    path=tmp_path/'synthetic_collection_resources.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if fault=='missing':rows.pop()
    elif fault=='reordered':rows[1],rows[2]=rows[2],rows[1]
    elif fault=='frame':rows[1]['frame']=1
    elif fault=='boolean_frame':rows[1]['frame']=False
    elif fault=='rss':rows[1]['rss_bytes']=49*audit.limits.GIB
    elif fault=='ram':rows[1]['memory_available_bytes']=15*audit.limits.GIB
    elif fault=='disk':rows[1]['artifact_free_bytes']=39*audit.limits.GIB
    elif fault=='clock':rows[1]['monotonic_s']=-1.
    elif fault=='consumption':rows[1]['disk_consumed_since_phase_start_bytes']=1
    elif fault=='receipt':
        receipt=tmp_path/'synthetic_collection_resource_result.json'
        data=json.loads(receipt.read_text());data['maximum_sampled_rss_bytes']+=1;receipt.write_text(json.dumps(data))
    elif fault=='count':collection['decisions']=4
    elif fault=='extra':rows.insert(-1,dict(rows[-2]))
    elif fault=='oversize':rows[1]['extra']='x'*4096
    else:rows[0]['memory_available_bytes']=63*audit.limits.GIB
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises((ValueError,audit.limits.ResourceLimitError)):audit.check(tmp_path,'synthetic',collection)
