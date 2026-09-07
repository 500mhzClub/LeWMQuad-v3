"""Optimizer/EMA/checkpoint invariants; synthetic steps are not science results."""
from copy import deepcopy
import io
import pytest
import torch
from lewm.pulse_timed_training_runner_development import PulseTrainer,state_digest
from lewm.tests.test_pulse_timed_learning_development import batch


def test_identical_initialization_across_arms_and_global_rng_isolation():
    state=torch.random.get_rng_state().clone()
    models=[PulseTrainer(c,seed=19,latent_dim=16) for c in ('direct','supervised_rollout','jepa')]
    assert len({m.initial_sha256 for m in models})==1
    assert torch.equal(state,torch.random.get_rng_state())


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
def test_actual_optimizer_step_and_exact_post_step_ema(condition):
    trainer=PulseTrainer(condition,seed=7,latent_dim=16,ema_momentum=.8)
    before=deepcopy(trainer.model.state_dict());result=trainer.step(batch())
    assert result['update']==1 and result['model_sha256']!=trainer.initial_sha256
    assert not trainer.failed and all(p.grad is None for p in trainer.model.target_encoder.parameters())
    for name,p in trainer.model.encoder.named_parameters():
        expected=before['target_encoder.'+name]*.8+p.detach()*.2
        torch.testing.assert_close(trainer.model.target_encoder.state_dict()[name],expected)
    if condition=='direct':
        for name,p in trainer.model.transition.named_parameters():torch.testing.assert_close(p,before['transition.'+name],rtol=0,atol=0)


def test_checkpoint_roundtrip_weights_only_and_snapshot_isolation():
    trainer=PulseTrainer('jepa',seed=3,latent_dim=16);trainer.step(batch());saved=trainer.checkpoint()
    buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0);loaded=torch.load(buffer,weights_only=True)
    clone=PulseTrainer('jepa',seed=3,latent_dim=16)
    clone.model.load_state_dict(loaded['model_state']);clone.optimizer.load_state_dict(loaded['optimizer_state'])
    assert state_digest(clone.model.state_dict())==loaded['model_sha256']
    assert loaded['updates']==1 and len(clone.optimizer.state)>0
    trainer.step(batch());assert state_digest(saved['model_state'])==loaded['model_sha256']


def test_bad_target_latches_without_optimizer_update():
    trainer=PulseTrainer('direct',seed=1,latent_dim=16);b=batch();b['targets']['motion'][0,0]=float('nan')
    with pytest.raises(ValueError):trainer.step(b)
    assert trainer.failed and trainer.updates==0 and state_digest(trainer.model.state_dict())==trainer.initial_sha256
    with pytest.raises(ValueError,match='latched'):trainer.step(batch())


def test_same_seed_and_same_batches_produce_identical_updates():
    b=batch();a=PulseTrainer('supervised_rollout',seed=41,latent_dim=16);c=PulseTrainer('supervised_rollout',seed=41,latent_dim=16)
    assert a.step(b)==c.step(deepcopy(b))


def test_actual_scoring_preserves_model_and_partial_target_times():
    from lewm.pulse_timed_training_runner_development import evaluate
    from lewm.pulse_timed_dataset_development import PulseTimedDataset
    from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
    from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,OUTPUT
    from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
    from scripts.startup_raw_sensor_audit_development import read_json
    windows=read_json(OUTPUT,'windows.json');targets=read_json(LABELS,'targets.json')
    ids=[next(i for i,w in enumerate(windows) if w['condition']==c) for c in ('nominal_left','lower_friction_left')]
    roles={windows[i]['condition']:dict(layout_id='same-room',role='train') for i in ids}
    dataset=PulseTimedDataset([windows[i] for i in ids],[targets[i] for i in ids],roles)
    readers={c:IntentReturnRGBDReplay(INPUT/c) for c in roles}
    trainer=PulseTrainer('direct',seed=7,latent_dim=16);trainer.model.train()
    result=evaluate(trainer,dataset,readers,role='train',batch_size=2)
    assert trainer.model.training and state_digest(trainer.model.state_dict())==trainer.initial_sha256
    assert result['resubstitution'] and not result['independent_generalization_established']
    assert set(result['metrics'])=={'direct_outcomes'}
    assert {'2200000000','2500000000'}<=set(result['metrics']['direct_outcomes']['by_actual_offset_ns'])
    assert result['metrics']['direct_outcomes']['all']['motion_valid']==10
    with pytest.raises(ValueError,match='no eligible'):evaluate(trainer,dataset,readers,role='development_eval')
