"""Live identity typing and exact serialized evidence are separate contracts."""
from copy import deepcopy
import json
import pytest
from lewm.causal_sensor_state import _identity, SensorContractError
from lewm import direct_flow_live_replay_validation_development as validation


def decision():
    return dict(original_visual_evidence={'identity':(0,0,0)},evidence={'identity':(0,0,0)},
        new_selection={'prediction':[[1.]],'action':'left_turn'},requested_command=[0.,0.,.45])


def test_actual_tuple_identity_contract_runs_on_live_evidence(monkeypatch):
    seen=[]
    def check(evidence,*args,**kwargs):
        seen.append(_identity(evidence['identity']))
        assert kwargs['now_ns']==22_900_000_000
    monkeypatch.setattr(validation,'current_dual_camera_pose',check)
    monkeypatch.setattr(validation,'current_measured_floor_pose',check)
    live=decision(); recorded=json.loads(json.dumps(live))
    validation.validate_live(live,recorded,{'controller_recovered':True},{},{},{},now_ns=22_900_000_000)
    assert seen==[(0,0,0),(0,0,0)] and live['evidence']['identity']==(0,0,0)
    with pytest.raises(SensorContractError,match='identity must be'):
        validation.validate_live(recorded,recorded,{'controller_recovered':True},{},{},{},now_ns=22_900_000_000)


@pytest.mark.parametrize('field,value',[('requested_command',[.2,0.,0.]),
    ('new_selection',{'prediction':[[2.]],'action':'left_turn'}),('evidence',{'identity':[0,1,0]})])
def test_cannot_validate_one_live_decision_and_save_another(field,value):
    live=decision(); recorded=json.loads(json.dumps(live)); recorded[field]=deepcopy(value)
    with pytest.raises(ValueError,match='exact serialized live decision'):
        validation.validate_live(live,recorded,{'controller_recovered':False},{},{},{},now_ns=1)


def test_negative_controller_result_is_preserved_without_invented_pose(monkeypatch):
    def fail(*args,**kwargs): raise AssertionError('no valid pose claimed')
    monkeypatch.setattr(validation,'current_dual_camera_pose',fail)
    monkeypatch.setattr(validation,'current_measured_floor_pose',fail)
    live={'original_visual_evidence':None,'evidence':None,'terminal':'SENSOR_OR_MODEL_FAILURE'}
    validation.validate_live(live,json.loads(json.dumps(live)),{'controller_recovered':False},{},{},{},now_ns=1)
