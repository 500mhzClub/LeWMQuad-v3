import copy

import numpy as np
import pytest
import torch

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.online_rgb_history_development import OnlineRGBHistory
from lewm.tests.test_relative_gyro_turn_development import initialized,append,packet


def stream():
    buffer=initialized(); rows=[packet(buffer,80)]
    for tick in range(1,8):
        for step in range(80+(tick-1)*5+1,80+tick*5+1): append(buffer,step,[0,0,.1*tick])
        p=packet(buffer,80+tick*5); p['image']['rgb'][:]=tick; rows.append(p)
    return rows


def test_sliding_live_history_exactly_matches_offline_causal_tensor_interface():
    rows=stream(); history=OnlineRGBHistory(); history.begin_episode((0,0,0))
    for i,p in enumerate(rows):
        ns=p['image']['measured_ns']; result=history.push(p,now_ns=ns)
        assert result['ready']==(i>=3)
        if i<3:
            with pytest.raises(SensorContractError): history.tensors(now_ns=ns)
        else:
            expected=causal_history_tensors(rows[i-3:i+1],ns); actual=history.tensors(now_ns=ns)
            for key in actual: assert torch.equal(actual[key],expected[key])


def test_frame_gap_requires_four_new_consecutive_frames():
    rows=stream(); history=OnlineRGBHistory(); history.begin_episode((0,0,0))
    for i in range(3): history.push(rows[i],now_ns=rows[i]['image']['measured_ns'])
    result=history.push(rows[4],now_ns=rows[4]['image']['measured_ns'])
    assert result['reset_for_gap'] and result['frames']==1 and not result['ready']
    for i in range(5,8): result=history.push(rows[i],now_ns=rows[i]['image']['measured_ns'])
    assert result['ready']


@pytest.mark.parametrize('fault',['duplicate','identity','privilege','invalid_sensor','rewritten','stale_image'])
def test_bad_input_clears_readiness(fault):
    rows=stream(); history=OnlineRGBHistory(); history.begin_episode((0,0,0))
    for p in rows[:4]: history.push(p,now_ns=p['image']['measured_ns'])
    p=copy.deepcopy(rows[4]); ns=p['image']['measured_ns']
    if fault=='duplicate': p=rows[3]; ns=p['image']['measured_ns']
    if fault=='identity': p['sensor_state']['identity']=(0,0,1)
    if fault=='privilege': p['world_pose']=np.zeros(7)
    if fault=='invalid_sensor':
        p['sensor_state']['sensed']['gyro']['valid'][-1]=False; p['sensor_state']['sensed']['gyro']['values'][-1]=0
    if fault=='rewritten': p['sensor_state']['sensed']['gyro']['values'][0,0]+=1
    if fault=='stale_image':
        p['image']['measured_ns']-=100_000_000; p['sensor_state']['image_ns']=p['image']['measured_ns']
    with pytest.raises(SensorContractError): history.push(p,now_ns=ns)
    with pytest.raises(SensorContractError): history.tensors(now_ns=ns)


def test_external_mutation_and_stale_inference_cannot_change_history():
    rows=stream(); history=OnlineRGBHistory(); history.begin_episode((0,0,0))
    for p in rows[:4]: history.push(p,now_ns=p['image']['measured_ns'])
    ns=rows[3]['image']['measured_ns']; before=history.tensors(now_ns=ns)
    rows[0]['image']['rgb'][:]=255; before['rgb'][:]=255
    assert float(history.tensors(now_ns=ns)['rgb'].max())<1.
    with pytest.raises(SensorContractError): history.tensors(now_ns=ns+100_000_000)
    with pytest.raises(SensorContractError): history.begin_episode((0,0,0))
    history.begin_episode((0,0,1))
    with pytest.raises(SensorContractError): history.tensors(now_ns=ns)
