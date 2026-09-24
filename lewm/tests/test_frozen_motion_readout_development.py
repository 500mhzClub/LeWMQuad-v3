from copy import deepcopy
import numpy as np
import torch
from lewm.command_history_residual_learning_development import CommandHistoryResidualTrainer,compose
from lewm.frozen_motion_readout_development import fit_readout,predict_readout,features,install_readout
from lewm.tests.test_observation_horizon_learning_development import batch


def test_weighted_fit_matches_augmented_least_squares():
    rng=np.random.default_rng(310);x=rng.normal(size=(40,7));x[:,-1]=2.
    y=rng.normal(size=(40,4));w=rng.integers(1,6,size=40).astype(float)
    fit=fit_readout(x,y,w);z=(x-fit['mean'])/fit['scale']
    expected=np.linalg.lstsq(np.vstack((z*np.sqrt(w[:,None]),np.eye(7))),
        np.vstack(((y-fit['bias'])*np.sqrt(w[:,None]),np.zeros((7,4)))),rcond=None)[0]
    np.testing.assert_allclose(fit['coefficient'],expected,atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.average(predict_readout(x,fit)-y,axis=0,weights=w),0.,atol=1e-12)


@torch.inference_mode()
def test_installed_head_matches_reference_composition_and_keeps_causality():
    inputs=batch()['inputs'];model=CommandHistoryResidualTrainer('jepa',seed=17,latent_dim=16).model.eval()
    frozen={k:v.clone() for k,v in model.state_dict().items()};x=features(model,inputs)
    old=model(**inputs);valid=inputs['known_action_valid'];mask=valid[...,0]
    rng=np.random.default_rng(27);target=rng.normal(scale=.001,size=(int(mask.sum()),4));target[:,3]+=1.
    fit=fit_readout(x.numpy()[mask.numpy()],target,np.ones(int(mask.sum())))
    reference=model.reference_motion(inputs['observation_history'],inputs['known_action_blocks'],valid)
    install_readout(model,fit);out=model(**inputs)
    residual=torch.as_tensor(predict_readout(x.numpy(),fit),dtype=torch.float32)
    expected=compose(torch.cat((residual,torch.zeros_like(residual[...,:1])),dim=-1),reference,valid)
    torch.testing.assert_close(out['rollout_outcomes'][...,:4],expected[...,:4],rtol=0,atol=2e-7)
    torch.testing.assert_close(out['rollout_outcomes'][...,4],old['rollout_outcomes'][...,4],rtol=0,atol=0)
    assert not out['rollout_outcomes'][~mask].any()
    for key,value in frozen.items():
        torch.testing.assert_close(model.state_dict()[key.replace('rollout_decode.2.','rollout_decode.2.original.')],value,rtol=0,atol=0)
    changed=deepcopy(inputs);changed['known_action_blocks'][0,4:,0,2]=-.9
    after=model(**changed)
    torch.testing.assert_close(out['rollout_outcomes'][:,:4],after['rollout_outcomes'][:,:4],rtol=0,atol=0)
