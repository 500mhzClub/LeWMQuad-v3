"""Scientific contract checks for the frozen reference and trained composition."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from lewm.command_history_residual_learning_development import CommandHistoryResidualTrainer,residual_loss,REFERENCE
from lewm.tests.test_observation_horizon_learning_development import batch
from lewm.pulse_position_scale_learning_development import scaled_outcome_loss
from scripts.fit_go2_local_motion_controls_development import nominal


def test_reference_matches_numpy_fit_and_preserves_causal_partial_plan():
    b=batch();inputs=b['inputs'];model=CommandHistoryResidualTrainer('jepa',seed=17,latent_dim=16).model
    with torch.no_grad():
        reference=model.reference_motion(inputs['observation_history'],inputs['known_action_blocks'],inputs['known_action_valid'])
        out=model(**inputs)
    with np.load(REFERENCE,allow_pickle=False) as archive:fit={k:archive[k].copy() for k in archive.files}
    commands=inputs['known_action_blocks'][:,:,0].double().numpy()*[.3,1.,.5]
    for row in range(2):
        base=nominal(commands[row]);history=inputs['observation_history']['control'][row].double().numpy().ravel()
        for h in range(int(inputs['known_action_valid'][row].sum())):
            known=np.zeros((8,3));known[:h+1]=commands[row,:h+1]
            x=np.r_[known.ravel(),base[h],history]
            expected=base[h]+((x-fit['mean'][h])/fit['scale'][h])@fit['coefficient'][h]+fit['bias'][h]
            np.testing.assert_allclose(reference[row,h],expected,rtol=0,atol=1e-10)
    active=inputs['known_action_valid'][...,0]
    for head in ('direct_outcomes','rollout_outcomes'):
        torch.testing.assert_close(out[head][:,:,:2],reference[:,:,:2].float(),rtol=0,atol=0)
        assert not out[head][~active].any()
    changed=deepcopy(inputs);changed['known_action_blocks'][0,4:,0,2]=-.9
    with torch.no_grad():after=model(**changed)
    for head in ('direct_outcomes','rollout_outcomes'):
        torch.testing.assert_close(out[head][:,:4],after[head][:,:4],rtol=0,atol=0)


@pytest.mark.parametrize('condition',('jepa','supervised_rollout'))
def test_training_uses_inference_composition_and_reference_stays_frozen(condition):
    trainer=CommandHistoryResidualTrainer(condition,seed=17,latent_dim=16);b=batch()
    # Check a nonzero learned correction as well as the reference-only initialization.
    with torch.no_grad():trainer.model.rollout_decode[-1].bias[:4].add_(.01)
    out=trainer.model(**b['inputs']);_,parts=residual_loss(trainer.model,b,condition)
    args=[b['targets'][k] for k in ('motion','contact','motion_valid','contact_valid')]
    for head,key in (('direct_outcomes','direct_outcome'),('rollout_outcomes','rollout_outcome')):
        assert parts[key]==float(scaled_outcome_loss(out[head],*args).detach())
    references={k:v.clone() for k,v in trainer.model.state_dict().items() if k.startswith('reference_')}
    result=trainer.step(b)
    assert result['update']==1 and np.isfinite(result['loss'])
    assert ('latent_prediction' in result['parts'])==(condition=='jepa')
    for k,v in references.items():torch.testing.assert_close(trainer.model.state_dict()[k],v,rtol=0,atol=0)
