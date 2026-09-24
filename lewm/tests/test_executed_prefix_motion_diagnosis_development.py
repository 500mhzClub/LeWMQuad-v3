import numpy as np
import pytest
from lewm.executed_prefix_motion_diagnosis_development import diagnose


def fixture():
    poses=np.zeros((1150,7));poses[:,5]=np.sqrt(.5);poses[:,6]=np.sqrt(.5)
    poses[:,1]=np.arange(1150)/5000
    prediction=np.zeros((8,5));prediction[:,0]=np.arange(1,9)*.01;prediction[:,3]=1.
    tape=[dict(tick=i,pre_sample_index=749+50*i,post_sample_index=799+50*i,
        requested_command=[0.,0.,0.],completed=True) for i in range(8)]
    return prediction,tape,poses


def test_native_target_uses_current_body_and_only_executed_prefix():
    prediction,tape,poses=fixture()
    rows=diagnose(prediction,np.zeros((8,3)),tape,poses,tick=0)
    assert len(rows)==8 and max(r['xy_error_m'] for r in rows)<1e-15
    tape[2]['requested_command']=[.2,0.,0.]
    rows=diagnose(prediction,np.zeros((8,3)),tape,poses,tick=0)
    assert len(rows)==2 and rows[-1]['end_sample']==849


def test_missing_or_incomplete_execution_cannot_make_a_target():
    prediction,tape,poses=fixture();tape[0]['completed']=False
    assert diagnose(prediction,np.zeros((8,3)),tape,poses,tick=0)==[]
    tape[0]['completed']=True
    with pytest.raises(ValueError,match='endpoint'):diagnose(prediction,np.zeros((8,3)),tape,poses[:799],tick=0)
    tape[0]['pre_sample_index']=748
    with pytest.raises(ValueError,match='clock'):diagnose(prediction,np.zeros((8,3)),tape,poses,tick=0)
