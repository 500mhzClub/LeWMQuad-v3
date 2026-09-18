import json
from copy import deepcopy
import numpy as np
from scripts import read_go2_auxiliary_depth_goal_probe_v1 as current
from scripts import read_go2_training_bias_goal_probe_v1 as previous
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_predecessor_and_causal_boundary():
    assert current.PRIOR==previous.INPUT and current.CASES==previous.CASES
    a,b=rows(8),rows(8);ta,tb=tape(7),tape(7);tb[3]['requested_command']=[.2,0,0]
    assert current.common_prefix_length(a,b,ta,tb)==(4,3,None)
    b[2]['decision']['terminal']='STOP'
    assert current.common_prefix_length(a,b,ta,tb)==(3,3,2)


def test_auxiliary_comparison_excludes_affected_future_and_checks_valid_mask(tmp_path,monkeypatch):
    monkeypatch.setattr(current,'read_json',lambda p,n:json.loads((p/n).read_text()))
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    observations=rows(4)
    for row in observations:
        row['decision'].update(evidence={},memory_receipt={},observed_goal_distance_m=1.,new_selection=None)
    commands=tape(3)
    for root in (a,b):
        for name,value in (('context_decisions.json',observations),('command_tape.json',commands),
                ('camera_audit.json',[{'rgb_sha256':'a'*64} for _ in observations])):
            (root/name).write_text(json.dumps(value))
        for name in ('physics_trace.npz','policy_histories.npz','fast_gyro_histories.npz'):
            np.savez(root/name,values=np.arange(1000))
        for i in range(4):
            np.savez(root/f'auxiliary_depth_{i:04d}.npz',depth_m=np.ones((2,2),np.float32),valid=np.ones((2,2),bool))
    commands[2]['requested_command']=[.2,0,0]
    (b/'command_tape.json').write_text(json.dumps(commands))
    np.savez(b/'auxiliary_depth_0003.npz',depth_m=np.zeros((2,2),np.float32),valid=np.zeros((2,2),bool))
    result=current.compare(a,b,auxiliary_pair=True)
    assert result['common_prefix_frames']==3 and result['common_prefix_auxiliary_exact']
    assert len(result['auxiliary_comparison'])==3 and not result['unexecuted_outcomes_inferred']
    np.savez(b/'auxiliary_depth_0002.npz',depth_m=np.ones((2,2),np.float32),valid=np.zeros((2,2),bool))
    assert not current.compare(a,b,auxiliary_pair=True)['common_prefix_auxiliary_exact']
    assert current.compare(a,b)['common_prefix_auxiliary_exact'] is None
