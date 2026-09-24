import json
from copy import deepcopy
import numpy as np
import pytest
from scripts import read_go2_training_bias_goal_probe_v1 as current
from scripts import read_go2_eight_step_planning_goal_probe_v1 as previous
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_predecessor_and_causal_boundary():
    assert current.PRIOR==previous.INPUT and current.CASES==previous.CASES
    a,b=rows(8),rows(8);ta,tb=tape(7),tape(7);tb[3]['requested_command']=[.2,0,0]
    assert current.common_prefix_length(a,b,ta,tb)==(4,3,None)
    b[2]['decision']['terminal']='STOP'
    assert current.common_prefix_length(a,b,ta,tb)==(3,3,2)


def test_exact_correction_excludes_affected_future_and_preserves_other_channels(tmp_path,monkeypatch):
    # Only synthetic temporary files use this reader substitution.
    monkeypatch.setattr(current,'read_json',lambda p,n:json.loads((p/n).read_text()))
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    before=np.arange(240,dtype=np.float32).reshape(6,8,5)/np.float32(1000)
    bias=np.arange(16,dtype=np.float32).reshape(8,2)/np.float32(700)
    after=before.copy();after[...,:2]-=bias
    originals=rows(4)
    for row in originals:
        row['decision'].update(evidence={},memory_receipt={},observed_goal_distance_m=1.,
            new_selection={'prediction':before.tolist()})
    corrected=deepcopy(originals)
    for row in corrected:row['decision']['new_selection']['prediction']=after.tolist()
    commands=tape(3)
    for root,samples in ((a,originals),(b,corrected)):
        for name,value in (('context_decisions.json',samples),('command_tape.json',commands),
                ('camera_audit.json',[{'rgb_sha256':'a'*64} for _ in samples])):
            (root/name).write_text(json.dumps(value))
        for name in ('physics_trace.npz','policy_histories.npz','fast_gyro_histories.npz'):
            np.savez(root/name,values=np.arange(1000))
    commands[2]['requested_command']=[.2,0,0]
    (b/'command_tape.json').write_text(json.dumps(commands))
    corrected[3]['decision']['new_selection']['prediction'][0][0][0]=100.
    (b/'context_decisions.json').write_text(json.dumps(corrected))
    result=current.compare(a,b,translation_bias=bias.tolist())
    assert result['common_prefix_frames']==3 and result['corrected_common_prefix_banks_verified']==3
    assert result['exact_training_correction_verified'] and not result['common_prefix_model_forecasts_exact']
    for channel in (0,1,2,3,4):
        changed=deepcopy(corrected);changed[2]['decision']['new_selection']['prediction'][0][0][channel]+=1.
        (b/'context_decisions.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='exact float32'):current.compare(a,b,translation_bias=bias.tolist())
    changed=deepcopy(corrected);changed[2]['decision']['new_selection']=None
    (b/'context_decisions.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='availability'):current.compare(a,b,translation_bias=bias.tolist())
    assert not current.compare(a,b)['training_correction_expected']
