"""Exercise actual exclusive writer across12 updates and checkpoint reload."""
import json
import pytest
from scripts import run_go2_pulse_training_pilot_v2 as pilot
from lewm.tests.test_pulse_timed_learning_development import batch


def test_actual_fit_persists_each_update_once_and_never_reuses_output(tmp_path,monkeypatch):
    b=batch()
    def take(value,i):
        if isinstance(value,dict):return {k:take(v,i) for k,v in value.items()}
        return value[i].clone()
    class Dataset:
        def sample(self,i,readers):return take(b,i)
    monkeypatch.setattr(pilot,'OUTPUT',tmp_path)
    monkeypatch.setattr(pilot,'evaluate',lambda *a,**k:dict(test_only=True))
    schedule=dict(batches=[[0,1]]*12,schedule_sha256='a'*64)
    report=pilot.run_fit(11,'direct',Dataset(),{},schedule)
    directory=tmp_path/'11_direct'
    aggregate=json.loads((directory/'updates.json').read_text())
    assert len(aggregate)==report['optimizer_steps']==12
    for i,row in enumerate(aggregate,1):
        assert json.loads((directory/('update_%04d.json'%i)).read_text())==row
    assert report['checkpoint_roundtrip_verified'] and (directory/'final_checkpoint.pt').is_file()
    before=(directory/'updates.json').read_bytes()
    with pytest.raises(FileExistsError):pilot.run_fit(11,'direct',Dataset(),{},schedule)
    assert (directory/'updates.json').read_bytes()==before
