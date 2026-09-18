from types import SimpleNamespace
import pytest
from scripts import auxiliary_prefix_frame_acquisition_development as module


def test_lazy_primary_capture_precedes_auxiliary_and_does_not_advance_physics(monkeypatch):
    events=[]
    session=SimpleNamespace(samples=[None]*750,model_manifest=[])
    def primary():
        index=(len(session.samples)-750)//50
        if len(session.model_manifest)==index:session.model_manifest.append({'frame':index})
        events.append(('primary',index));return index
    session.capture_current=primary
    def auxiliary(s,d,tick):
        assert len(s.model_manifest)==tick+1
        assert len(s.samples)==750+50*tick
        events.append(('auxiliary',tick));return {'frame':tick}
    monkeypatch.setattr(module,'capture',auxiliary)
    for tick in range(20):
        if tick:session.samples.extend([None]*50)
        assert module.capture_frame(session,None,tick)=={'frame':tick}
    assert events==[(kind,tick) for tick in range(20) for kind in ('primary','auxiliary')]


def test_wrong_primary_index_or_physical_clock_prevents_auxiliary(monkeypatch):
    monkeypatch.setattr(module,'capture',lambda *args:pytest.fail('auxiliary must not run'))
    for index,n in ((0,800),(1,799)):
        s=SimpleNamespace(samples=[None]*n,model_manifest=[None]*2,capture_current=lambda:index)
        with pytest.raises(ValueError):module.capture_frame(s,None,1)
