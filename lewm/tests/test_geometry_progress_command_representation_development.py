"""Exact recorder precision, isolated source change and old-failure preservation."""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.tests.test_geometry_progress_pilot_development import trace
from scripts import audit_go2_geometry_progress_pilot_v1 as old
from scripts import read_go2_geometry_progress_commands_v1 as new


def recorded_trace(action='left_arc'):
    raw,tape,rows,result,frames=trace(action)
    raw['requested_command']=np.zeros_like(raw['requested_command'],dtype=np.float64)
    raw['applied_command']=np.zeros_like(raw['applied_command'],dtype=np.float64)
    for t in tape:
        a,b=t['pre_sample_index'],t['post_sample_index']
        raw['requested_command'][a+1:b+1]=t['requested_command']
        prior=raw['applied_command'][a].astype(np.float32);delta=np.array([.25,0.,.35],np.float32)
        applied=np.clip(np.asarray(t['requested_command'],np.float32),prior-delta,prior+delta)
        raw['applied_command'][a+1:b+1]=applied
    raw['post_slew_applied_command']=raw['applied_command'].copy()
    return raw,tape,rows,result


@pytest.mark.parametrize('action',ACTIONS)
def test_exact_actual_serialization_including_arc_yaw_slew_and_terminal_brake(action):
    raw,tape,rows,result=recorded_trace(action)
    new.audit_commands(raw,tape,rows,result,action)
    if action!='hold':
        with pytest.raises(AssertionError):old.audit_commands(raw,tape,rows,result,action)


@pytest.mark.parametrize('field',['requested_command','applied_command','post_slew_applied_command'])
def test_single_ulp_command_corruption_is_not_hidden_by_a_tolerance(field):
    raw,tape,rows,result=recorded_trace()
    if field=='requested_command':raw[field][900,2]=np.nextafter(raw[field][900,2],np.inf)
    else:raw[field][900,2]=np.nextafter(np.float32(raw[field][900,2]),np.float32(np.inf))
    with pytest.raises(AssertionError):new.audit_commands(raw,tape,rows,result,'left_arc')


def test_every_other_raw_condition_check_and_progress_score_is_unchanged():
    def function(module,name):
        s=Path(module.__file__).read_text().replace('directory=OUTPUT/trial','directory=INPUT/trial')
        return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_condition','native_horizons','prefix_witness','compare_prefixes'):
        assert function(new,name)==function(old,name)
    assert new.audit_sensors is old.audit_sensors and new.audit_setup is old.audit_setup and new.audit_stops is old.audit_stops
    assert new.INPUT==old.OUTPUT and new.OUTPUT!=old.OUTPUT


def test_separate_readout_refuses_existing_output_before_reading_inputs(monkeypatch,tmp_path):
    monkeypatch.setattr(new,'OUTPUT',tmp_path)
    monkeypatch.setattr(new,'validate_root',lambda *a,**kw:None)
    monkeypatch.setattr(new,'verify_artifacts',lambda *a:pytest.fail('must refuse before reading'))
    with pytest.raises(ValueError,match='exclusive'):new.main()
