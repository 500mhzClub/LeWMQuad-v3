"""Real recorded serialization and rejection of nonbinary coercions."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.recorded_pulse_native_targets_development import RecordedPulseNativeTargets
from lewm.tests.test_pulse_native_targets_development import data
from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,TRIALS
from scripts.startup_raw_sensor_audit_development import read_npz,read_json


@pytest.mark.parametrize('condition',TRIALS)
def test_actual_recorded_trace_and_frozen_window(condition):
    raw=read_npz(INPUT/condition,'physics_trace.npz');model=RecordedPulseNativeTargets(raw)
    windows=read_json(INPUT.parent/'go2_pulse_timed_pairing_diagnostic_v1_attempt_001','windows.json')
    w=next(w for w in windows if w['condition']==condition)
    result=model.labels(w)
    assert result['motion_valid'][:5].all() and result['contact_valid'][:5].all()
    assert raw['physics_contact'].dtype==np.uint8


@pytest.mark.parametrize('value',[np.array([2],np.uint8),np.array([0.],float),np.array([False]),np.array([-1],np.int8)])
def test_ambiguous_contact_encoding_rejected(value):
    raw,w=data();raw['physics_contact']=value
    with pytest.raises(ValueError,match='binary uint8'):RecordedPulseNativeTargets(raw)


def test_decoding_does_not_mutate_raw_bytes():
    raw,w=data();raw['physics_contact']=raw['physics_contact'].astype(np.uint8);saved=deepcopy(raw)
    RecordedPulseNativeTargets(raw).labels(w)
    for k in raw:np.testing.assert_array_equal(raw[k],saved[k])
