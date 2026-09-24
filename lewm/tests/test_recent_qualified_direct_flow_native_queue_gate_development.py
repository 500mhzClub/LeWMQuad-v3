"""No launch-order bypass through missing, stale or incompatible completion receipts."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import recent_qualified_direct_flow_native_queue_gate_development as gate


@pytest.mark.parametrize('fault',[None,'missing_sha','malformed_sha','failure','sources','status',
    'retry','launch','result_sources','report','saved_native','queue','changed_artifact','original_error'])
def test_completed_predecessor_admission(monkeypatch,tmp_path,fault):
    sources={'fixture.py':'a'*64};report={'queue_result_sha256':'b'*64,'result_sha256':'c'*64}
    queue={'completed':['fixture']}
    result=dict(status='SUPERVISED_COMMITMENT_CONTACT_NATIVE_WAIT_V1_COMPLETE',
        source_sha256=deepcopy(sources),automatic_retry=False,
        artifact_sha256={'launch.json':gate.WAIT_LAUNCH_SHA},report=deepcopy(report))
    saved_native=deepcopy(report);saved_queue=deepcopy(queue);expected='d'*64;calls=[]
    if fault=='missing_sha':expected=None
    elif fault=='malformed_sha':expected='not-a-sha'
    elif fault=='failure':(tmp_path/'failure.json').write_text('{}')
    elif fault=='sources':sources['fixture.py']='0'*64
    elif fault=='status':result['status']='RUNNING'
    elif fault=='retry':result['automatic_retry']=True
    elif fault=='launch':result['artifact_sha256']['launch.json']='0'*64
    elif fault=='result_sources':result['source_sha256']={}
    elif fault=='report':result['report']['changed']=True
    elif fault=='saved_native':saved_native['changed']=True
    elif fault=='queue':saved_queue['changed']=True
    def authenticate(value):
        assert value is sources;calls.append('original')
        if fault=='original_error':raise ValueError('original failed')
        return report
    monkeypatch.setattr(gate,'predecessor',SimpleNamespace(OUTPUT=tmp_path,authenticate_native=authenticate))
    monkeypatch.setattr(gate,'predecessor_sources',lambda:{'fixture.py':'a'*64})
    monkeypatch.setattr(gate,'read_json',lambda root,name:{
        'result.json':result,'native_completion.json':saved_native,'queue_completion.json':saved_queue}[name])
    def artifacts(*args):
        calls.append('artifacts')
        if fault=='changed_artifact':raise ValueError('artifact changed')
    monkeypatch.setattr(gate,'verify_artifacts',artifacts)
    monkeypatch.setattr(gate,'verify',lambda value:calls.append('sources'))
    monkeypatch.setattr(gate,'verify_queue_completion',lambda sha,s:queue)
    if fault is None:
        receipt=gate.verify_completed_predecessor(expected,sources)
        assert receipt['wait_result_sha256']==expected and receipt['native_completion']==report
        assert receipt['queue_completion']==queue and receipt['original_contact_verifier_reexecuted']
        assert calls.count('original')==1 and calls.count('artifacts')==3
    else:
        with pytest.raises(ValueError):gate.verify_completed_predecessor(expected,sources)
