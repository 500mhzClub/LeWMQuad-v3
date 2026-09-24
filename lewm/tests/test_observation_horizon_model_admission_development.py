import pytest
from scripts import observation_horizon_model_admission_development as mod


@pytest.mark.parametrize('fault',['missing','duplicate','steps','selected'])
def test_incomplete_or_selected_study_cannot_reach_checkpoint_loading(monkeypatch,fault):
    row=dict(status='OBSERVATION_HORIZON_EIGHTEEN_FITS_COMPLETE',optimizer_updates=21600,scientific_models_trained=18,
        records=[dict(name=n) for n in mod.ROSTER],checkpoint_selection_performed=False,benchmark_weights_reused=False)
    if fault=='missing':row['records'].pop()
    if fault=='duplicate':row['records'][-1]=row['records'][0]
    if fault=='steps':row['optimizer_updates']=21599
    if fault=='selected':row['checkpoint_selection_performed']=True
    reads=[]
    monkeypatch.setattr(mod,'verify_artifacts',lambda *a:None)
    def read(root,name):reads.append(name);return row
    monkeypatch.setattr(mod,'read_json',read)
    monkeypatch.setattr(mod,'load_snapshot',lambda *a,**k:pytest.fail('partial study reached a model'))
    with pytest.raises(ValueError,match='all eighteen'):mod.admit('0'*64)
    assert reads==['result.json']


def test_assigned_loader_requires_complete_all_model_admission_before_file_access(monkeypatch):
    admission=dict(snapshots={n:{} for n in mod.ROSTER},all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed=False)
    monkeypatch.setattr(mod,'verify_artifacts',lambda *a:pytest.fail('unadmitted model reached artifacts'))
    with pytest.raises(ValueError,match='complete eighteen'):mod.load_assigned(admission,mod.ROSTER[0])


def test_snapshot_reloads_only_with_exact_new_temporal_contract():
    from copy import deepcopy
    from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
    from scripts.observation_horizon_snapshot_development import SCHEMA,config,validate_payload
    trainer=ObservationHorizonTrainer('direct',seed=17,latent_dim=8)
    binding=dict(experiment_sha256='a'*64,dataset_sha256='b'*64,schedule_sha256='c'*64,input_variant='full')
    payload=dict(schema=SCHEMA,binding=binding,trainer=trainer.checkpoint())
    loaded=validate_payload(payload,binding,config(trainer))
    assert loaded.initial_sha256==trainer.initial_sha256 and loaded.updates==0
    with pytest.raises(ValueError,match='evaluation-only'):loaded.step(None)
    for key,value in [('target_cadence_ns',500_000_000),('maximum_horizon_ns',4_000_000_000),('action_ticks_per_block',5)]:
        changed=deepcopy(payload);changed['trainer'][key]=value
        with pytest.raises(ValueError):validate_payload(changed,binding,config(trainer))
