import pytest
from scripts import augmented_family_switch_model_admission_development as mod


@pytest.mark.parametrize('fault',['missing','duplicate','steps','selected'])
def test_incomplete_or_selected_study_cannot_reach_checkpoint_loading(monkeypatch,fault):
    row=dict(status='AUGMENTED_FAMILY_SWITCH_EIGHTEEN_FITS_COMPLETE',optimizer_updates=21600,scientific_models_trained=18,
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
