import numpy as np
import pytest
import torch

from lewm.rgb_body_jepa_reference_development import RGBBodyJEPAReference
from lewm.rgb_body_learning_experiment_development import active_parameters,layout_batches,prediction_metrics,simple_predictions,training_loss


def batch():
    observation={'rgb':torch.rand(2,3,96,128),'body':torch.rand(2,20,63),'control':torch.rand(2,15,7)}
    future={k:v[:,None].repeat(1,8,*([1]*(v.ndim-1))) for k,v in observation.items()}
    valid=torch.ones(2,8,dtype=torch.bool)
    valid[0,3:]=False
    # Missing frames can be poisoned; masking must precede all encoding/arithmetic.
    for v in future.values(): v[~valid]=float('nan')
    motion=torch.zeros(2,8,3); motion[~valid]=float('nan')
    return {'observation':observation,'known_action_blocks':torch.zeros(2,8,5,3),
        'targets':{'future_observations':future,'future_valid':valid,'motion':motion,'motion_valid':valid,
            'contact':torch.zeros(2,8),'contact_valid':torch.ones_like(valid)}}


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
def test_condition_gradients_and_censoring(condition):
    model=RGBBodyJEPAReference(); data=batch()
    loss,parts=training_loss(model,data,condition); loss.backward()
    assert np.isfinite(float(loss.detach()))
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active_parameters(model,condition))
    assert ('latent_prediction' in parts)==(condition=='jepa')
    if condition=='direct': assert all(p.grad is None for p in model.transition.parameters())


def test_direct_condition_receives_future_observation_gradients():
    model=RGBBodyJEPAReference(); data=batch()
    for v in data['targets']['future_observations'].values(): v.requires_grad_()
    training_loss(model,data,'direct')[0].backward()
    for v in data['targets']['future_observations'].values():
        assert v.grad is not None and torch.isfinite(v.grad).all()
        assert v.grad[data['targets']['future_valid']].abs().sum()>0
        assert v.grad[~data['targets']['future_valid']].abs().sum()==0


def test_layout_batches_cover_population_without_duplicate_layout():
    metadata=[{'layout_id':f'l{i}','action_index':a} for i in range(16) for a in range(5)]
    batches=list(layout_batches(metadata,3,17))
    assert sorted(sum(batches,[]))==list(range(80))
    assert all(len({metadata[i]['layout_id'] for i in indices})==16 for indices in batches)
    assert batches==list(layout_batches(metadata,3,17))
    with pytest.raises(ValueError): list(layout_batches(metadata[:-1],3,17))


def test_metrics_ignore_censored_nan_and_reduce_layouts():
    predictions=np.zeros((2,8,5)); predictions[...,3]=1
    targets={'motion':np.zeros((2,8,3)),'contact':np.zeros((2,8)),
        'motion_valid':np.ones((2,8),dtype=bool),'contact_valid':np.ones((2,8),dtype=bool)}
    targets['motion_valid'][0,1:]=False; targets['motion'][0,1:]=np.nan
    predictions[0,:,0]=2.
    result=prediction_metrics(predictions,targets,[{'layout_id':'a'},{'layout_id':'b'}])
    assert result['layout_macro']['position_error_m']==1.
    assert result['layout_macro']['contact_brier']==.25
    assert result['motion_valid']==9
    assert len(result['by_horizon_seconds'])==8


def test_baselines_do_not_consume_validation_outcomes():
    train={'metadata':[{'action_index':a} for a in range(5)],
        'targets':{'motion':np.zeros((5,8,3)),'contact':np.zeros((5,8)),
            'motion_valid':np.ones((5,8),dtype=bool),'contact_valid':np.ones((5,8),dtype=bool)}}
    validation={'metadata':train['metadata'],'known_action_blocks':np.zeros((5,8,5,3))}
    validation['known_action_blocks'][1,:,:,0]=1.
    predictions=simple_predictions(train,validation)
    assert predictions['command_kinematics_no_contact'][1,-1,0]==pytest.approx(1.2)
    assert np.all(predictions['command_kinematics_no_contact'][0,:,:2]==0)
    validation['targets']={'motion':np.full((5,8,3),999.)}
    again=simple_predictions(train,validation)
    assert all(np.array_equal(predictions[k],again[k]) for k in predictions)


def test_known_plan_shuffle_and_tensor_stacking():
    from scripts.run_go2_rgb_body_learning_comparison_development_v1 import permutation,take,tensor_stack
    metadata=[{'layout_id':f'l{i}','action_index':a} for i in range(8) for a in range(5)]
    scene=permutation(metadata,'scene'); action=permutation(metadata,'action')
    assert sorted(scene)==sorted(action)==list(range(40))
    for i in range(40):
        assert metadata[scene[i]]['action_index']==metadata[i]['action_index']
        assert metadata[scene[i]]['layout_id']!=metadata[i]['layout_id']
        assert metadata[action[i]]['layout_id']==metadata[i]['layout_id']
        assert metadata[action[i]]['action_index']!=metadata[i]['action_index']
    stacked=tensor_stack([{'a':torch.tensor(i)} for i in range(3)])
    assert torch.equal(take(stacked,[2,0])['a'],torch.tensor([2,0]))
