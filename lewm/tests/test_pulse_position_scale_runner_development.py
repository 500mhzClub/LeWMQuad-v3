import json
import numpy as np
import pytest
import torch
from lewm.pulse_position_scale_learning_development import PositionScaleTrainer
from lewm.pulse_position_scale_evaluation_development import evaluate
from lewm.pulse_timed_training_runner_development import evaluate as frozen_evaluate,state_digest
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.tests.test_pulse_timed_learning_development import batch


def test_actual_scoring_matches_frozen_reducer_and_preserves_partial_times():
    from scripts.check_go2_pulse_timed_pairing_v1 import OUTPUT,INPUT
    from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
    from scripts.startup_raw_sensor_audit_development import read_json
    windows=read_json(OUTPUT,'windows.json');targets=read_json(LABELS,'targets.json')
    ids=[next(i for i,w in enumerate(windows) if w['condition']==c) for c in ('nominal_left','lower_friction_left')]
    roles={windows[i]['condition']:dict(layout_id='same-room',role='train') for i in ids}
    dataset=PulseTimedDataset([windows[i] for i in ids],[targets[i] for i in ids],roles)
    readers={c:IntentReturnRGBDReplay(INPUT/c) for c in roles}
    trainer=PositionScaleTrainer('jepa',objective='position_6cm',seed=4,latent_dim=16)
    prior=frozen_evaluate(trainer,dataset,readers,role='train');result,arrays=evaluate(trainer,dataset,readers)
    for head in ('direct_outcomes','rollout_outcomes'):
        assert result['metrics'][head]['all']==prior['metrics'][head]['all']
        assert result['metrics'][head]['by_actual_offset_ns']==prior['metrics'][head]['by_actual_offset_ns']
    assert arrays['motion_valid'].sum()==10 and arrays['active'][:,4].all()
    assert set(arrays['offsets_ns'][:,4])=={2_200_000_000,2_500_000_000}
    assert trainer.model.training and state_digest(trainer.model.state_dict())==trainer.initial_sha256


def test_actual_writer_snapshot_optimizer_and_refusal_to_overwrite(tmp_path,monkeypatch):
    import scripts.run_go2_pulse_position_scale_budget_v1 as runner
    monkeypatch.setattr(runner,'UPDATES',2);monkeypatch.setattr(runner,'SNAPSHOTS',(1,2))
    monkeypatch.setattr(runner,'RESERVE',0)
    monkeypatch.setattr(runner,'evaluate',lambda t,d,r:(dict(model_sha256=state_digest(t.model.state_dict())),dict(test=np.zeros(1))))
    seed=42;trainer=PositionScaleTrainer('jepa',objective='position_6cm',seed=seed)
    schedule=dict(batches=[[0,1]],schedule_sha256='synthetic')
    args=(tmp_path/'fit',seed,'jepa','position_6cm',None,None,schedule,[batch()],dict(initial_model_sha256=trainer.initial_sha256))
    result=runner.run_fit(*args)
    assert result['optimizer_steps']==2 and set(result['snapshots'])=={'1','2'}
    rows=json.loads((tmp_path/'fit'/'updates.json').read_text())
    for step in (1,2):
        assert rows[step-1]==json.loads((tmp_path/'fit'/('update_%04d.json'%step)).read_text())
        checkpoint=torch.load(tmp_path/'fit'/('snapshot_%04d.pt'%step),weights_only=True)
        assert checkpoint['updates']==step and checkpoint['objective']=='position_6cm'
        assert all(int(s['step'])==step for s in checkpoint['optimizer_state']['state'].values())
    with pytest.raises(FileExistsError):runner.run_fit(*args)
