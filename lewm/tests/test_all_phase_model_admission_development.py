from copy import deepcopy
import pytest
from scripts.all_phase_model_admission_development import verify_ledger, load_assigned


def ledger():
    schedule = dict(schedule_sha256='fixed', batches=[[i%4010]*6 for i in range(1200)])
    rows = [dict(update=i+1, sample_indices=batch, schedule_sha256='fixed', input_variant='full',
        model_sha256='final' if i==1199 else 'intermediate') for i,batch in enumerate(schedule['batches'])]
    return rows, schedule


def test_complete_expanded_optimizer_ledger_is_accepted():
    rows, schedule = ledger()
    verify_ledger(iter(rows), schedule, variant='full', model_sha256='final')


@pytest.mark.parametrize('fault',['omitted','extra','transfer','order','variant','schedule','model','empty'])
def test_incomplete_changed_or_transfer_leaking_ledger_is_rejected(fault):
    rows, schedule = ledger(); rows = deepcopy(rows)
    if fault=='omitted':rows.pop(600)
    elif fault=='extra':rows.append(deepcopy(rows[-1]))
    elif fault=='transfer':rows[500]['sample_indices'][0]=4800
    elif fault=='order':rows[500]['update']=499
    elif fault=='variant':rows[700]['input_variant']='no_rgb'
    elif fault=='schedule':rows[800]['schedule_sha256']='different'
    elif fault=='model':rows[-1]['model_sha256']='wrong'
    elif fault=='empty':rows=[]
    with pytest.raises(ValueError):verify_ledger(iter(rows), schedule, variant='full', model_sha256='final')


def test_partial_model_roster_cannot_be_loaded():
    with pytest.raises(ValueError,match='complete'):
        load_assigned(dict(snapshots={},all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed=True),'seed_2026091001_full_jepa')
