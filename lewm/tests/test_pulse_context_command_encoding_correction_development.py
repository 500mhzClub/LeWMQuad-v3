"""Actual float64 requests versus float32 actuator path; no threshold relaxation."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_independent_pulse_context_development import trace_fixture
from scripts.audit_go2_independent_pulse_context_pilot_v1 import audit_commands as failed_audit
from scripts.audit_go2_independent_pulse_context_command_encoding_correction_v1 import audit_commands


@pytest.mark.parametrize('action',range(6))
def test_correct_raw_encoding_passes_without_rounding_requests(action):
    raw,tape,rows,result=trace_fixture(action)
    raw['requested_command']=np.zeros(raw['requested_command'].shape,np.float64)
    for t in tape:
        raw['requested_command'][t['pre_sample_index']+1:t['post_sample_index']+1]=t['requested_command']
    raw['applied_command']=raw['applied_command'].astype(np.float64)
    with pytest.raises(AssertionError):failed_audit(raw,tape,rows,result,action)
    audit_commands(raw,tape,rows,result,action)
    bad=deepcopy(raw);bad['requested_command'][750,0]=np.float32(.12)
    with pytest.raises(AssertionError):audit_commands(bad,tape,rows,result,action)
    bad=deepcopy(raw);bad['applied_command'][750,0]+=.001
    with pytest.raises(AssertionError):audit_commands(bad,tape,rows,result,action)


def test_wrong_raw_dtype_is_not_silently_accepted():
    raw,tape,rows,result=trace_fixture()
    with pytest.raises(AssertionError):audit_commands(raw,tape,rows,result,0)
