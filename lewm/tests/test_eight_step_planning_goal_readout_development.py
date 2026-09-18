import json
from copy import deepcopy
import numpy as np
import pytest
from scripts import read_go2_eight_step_planning_goal_probe_v1 as current
from scripts import read_go2_observation_horizon_goal_probe_v1 as previous
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_predecessor_and_comparison_boundary_preserve_actual_prefix():
    assert current.PRIOR==previous.INPUT and current.CASES==previous.CASES
    a,b=rows(8),rows(8);ta,tb=tape(7),tape(7);tb[3]['requested_command']=[.2,0,0]
    assert current.common_prefix_length(a,b,ta,tb)==(4,3,None)
    b[2]['decision']['terminal']='STOP'
    assert current.common_prefix_length(a,b,ta,tb)==(3,3,2)


def test_same_model_forecast_guard_excludes_affected_future(tmp_path,monkeypatch):
    # Synthetic temporary reader only; production artifact guards remain intact.
    monkeypatch.setattr(current,'read_json',lambda p,n:json.loads((p/n).read_text()))
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    samples=rows(4)
    for row in samples:
        row['decision'].update(evidence={},memory_receipt={},observed_goal_distance_m=1.,
            new_selection={'prediction':[[[0.,0.,0.,1.,-5.]]]})
    commands=tape(3)
    for root in (a,b):
        for name,value in (('context_decisions.json',samples),('command_tape.json',commands),
                ('camera_audit.json',[{'rgb_sha256':'a'*64} for _ in samples])):
            (root/name).write_text(json.dumps(value))
        for name in ('physics_trace.npz','policy_histories.npz','fast_gyro_histories.npz'):
            np.savez(root/name,values=np.arange(1000))
    commands[2]['requested_command']=[.2,0,0]
    (b/'command_tape.json').write_text(json.dumps(commands))
    changed=deepcopy(samples);changed[3]['decision']['new_selection']['prediction'][0][0][0]=1.
    (b/'context_decisions.json').write_text(json.dumps(changed))
    r=current.compare(a,b,same_model=True)
    assert r['common_prefix_frames']==3 and r['common_prefix_model_forecasts_exact']
    changed[2]['decision']['new_selection']['prediction'][0][0][0]=1.
    (b/'context_decisions.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='unchanged model'):current.compare(a,b,same_model=True)
    assert not current.compare(a,b,same_model=False)['common_prefix_model_forecasts_exact']
