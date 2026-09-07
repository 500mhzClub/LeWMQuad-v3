from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lewm.causal_sensor_state import SensorContractError
from scripts.run_go2_successive_choice_maze_development_v1 import LiveSession,RouteSession


def fake(tmp_path):
    # No Genesis initialization or world-pose input in this live packet test.
    session=LiveSession.__new__(LiveSession); session.output=tmp_path
    session.last_observed_ns=None; session.observed_indices=[]; session.observe_enabled=True
    pixels=np.zeros((4,4,3),dtype=np.uint8); pixels[0,0]=[13,27,39]; Image.fromarray(pixels).save(tmp_path/'actual.png')
    session.model_manifest=[{'image_ns':1_500_000_000,'rgb_file':'actual.png'}]
    session.capture_current=lambda:0; calls=[]
    session.observations=SimpleNamespace(packet=lambda rgb,ns:{'actual':rgb,'now':ns})
    session.policy=SimpleNamespace(observe=lambda packet,now_ns:calls.append((packet,now_ns)))
    return session,calls,pixels


def test_current_actual_image_once_per_boundary(tmp_path):
    session,calls,pixels=fake(tmp_path)
    assert session.observe_current()==session.observe_current()==0
    assert len(calls)==1 and np.array_equal(calls[0][0]['actual'],pixels)
    assert session.observed_indices==[0] and session.last_observed_ns==1_500_000_000


def test_release_bypasses_failed_adapter_but_dispatches_explicit_zero(tmp_path,monkeypatch):
    session,_,_=fake(tmp_path); sent=[]
    def fail(): raise SensorContractError('failed adapter')
    session.observe_current=fail
    monkeypatch.setattr(RouteSession,'command_tick',lambda self,requested:sent.append(requested))
    with pytest.raises(SensorContractError): session.command_tick([.2,0,0])
    assert not sent
    session.observe_enabled=False; session.command_tick([0,0,0])
    assert sent==[[0,0,0]]


def test_failed_observation_not_marked_ingested(tmp_path):
    session,_,_=fake(tmp_path)
    def fail(*args,**kwargs): raise SensorContractError('rejected packet')
    session.policy.observe=fail
    with pytest.raises(SensorContractError): session.observe_current()
    assert session.last_observed_ns is None and not session.observed_indices
