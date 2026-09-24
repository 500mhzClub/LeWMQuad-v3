from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lewm.observation_horizon_plan_development import plan
from lewm.observation_horizon_view_development import ObservationHorizonView
from lewm.observation_horizon_fit_development import verified_plan,score,train
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
from lewm.tests.test_observation_horizon_stream_development import fixture


def view():
    original,rows=fixture();return ObservationHorizonView(original,rows)


def test_fit_plan_cannot_change_assigned_action_or_temporal_contract():
    v=view();ids=[v.indices('train',source=s)[0] for s in ('family','switch')]
    pairs=[plan(v.rows[i]['action'],offset_ticks=v.rows[i].get('offset_ticks',0)) for i in ids]
    inputs=dict(known_action_blocks=torch.stack([p[0] for p in pairs]),known_action_valid=torch.stack([p[1] for p in pairs]))
    active,offsets=verified_plan(v,ids,inputs)
    assert active.shape==(2,8) and offsets[:,0].tolist()==[100_000_000]*2
    inputs['known_action_blocks'][0,0,0,0]+=.01
    with pytest.raises(ValueError,match='exact'):verified_plan(v,ids,inputs)


def test_scoring_keeps_first_observation_and_shared_half_second_distinct():
    v=view();ids=v.indices('geometry_transfer');prediction=np.zeros((len(ids),8,5),np.float32);prediction[:,:,3]=1.
    arrays=dict(indices=np.array(ids),prediction_valid=np.array([[t['in_plan'] for t in v.rows[i]['targets']] for i in ids]),
        target_offsets_ns=np.array([[t['offset_ns'] for t in v.rows[i]['targets']] for i in ids]),direct_outcomes=prediction)
    result=score(v,arrays,role='geometry_transfer',head='direct_outcomes')
    scopes={r['scope'] for r in result['clusters']}
    assert {'first_observation','half_second'}<=scopes and 'first_half_second' not in scopes
    assert result['planned_family_windows']==384 and result['planned_switch_cells']==72
    arrays['target_offsets_ns'][0,0]=500_000_000
    with pytest.raises(ValueError,match='clocks'):score(v,arrays,role='geometry_transfer',head='direct_outcomes')


def test_changed_schedule_is_rejected_before_target_materialization_or_update():
    v=view();schedule=deepcopy(v.schedule(updates=1200,batch_size=6,seed=2026091001))
    schedule['batches'][0][0]=v.indices('geometry_transfer')[0]
    data=SimpleNamespace(view=v,training_batch=lambda indices:pytest.fail('changed schedule reached training data'))
    trainer=ObservationHorizonTrainer('jepa',seed=2026091001,latent_dim=8)
    with pytest.raises(ValueError,match='schedule'):train(trainer,data,schedule,input_variant='full',on_update=lambda r:None)
    assert trainer.failed and trainer.updates==0
