"""Reject incomplete or misleading paired registration timing evidence."""
from copy import deepcopy
import pytest
from scripts.verify_go2_density_routed_floor_registration_completion_v1 import validate_row


def row():
    return dict(frame=0,order=['original','candidate'],public_input_sha256='a'*64,
        recorded_evidence_exact=True,complete_registration_state_exact=True,
        public_inputs_unchanged=True,wall_ms=dict(original=100.,candidate=80.))


def test_valid_pair_returns_both_measured_times():
    assert validate_row(row(),0)==dict(original=100.,candidate=80.)


@pytest.mark.parametrize('key,value',[
    ('frame',False),('frame',1),('order',['candidate','original']),
    ('public_input_sha256','unknown'),('recorded_evidence_exact',1),
    ('complete_registration_state_exact',False),('public_inputs_unchanged',None),
    ('wall_ms',dict(original=100.)),('wall_ms',dict(original=float('nan'),candidate=80.)),
    ('wall_ms',dict(original=100.,candidate=0.)),('wall_ms',dict(original=True,candidate=80.)),
])
def test_corrupt_or_incomplete_evidence_rejected(key,value):
    changed=deepcopy(row());changed[key]=value
    with pytest.raises(ValueError):validate_row(changed,0)


def test_odd_frame_requires_reversed_order():
    changed=row();changed['frame']=1
    with pytest.raises(ValueError):validate_row(changed,1)
    changed['order']=['candidate','original']
    assert validate_row(changed,1)==changed['wall_ms']
