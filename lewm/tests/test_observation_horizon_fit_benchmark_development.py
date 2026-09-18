from copy import deepcopy
import pytest
from scripts.run_go2_observation_horizon_fits_v1 import ROSTER,OPTIMIZATION_SEEDS,benchmark_decision


def phase(name,wall):
    return dict(wall_s=wall,records=[dict(name=f'{name}_{i}',status='OBSERVATION_HORIZON_WORKER_COMPLETE',
        actual_updates=20,fit=dict(seed=2026091610+i,model_sha256=str(i)*64),ledger_sha256=str(i)*64,
        peak_rss_bytes=2*1024**3) for i in range(4)])


def test_roster_is_three_seeds_and_six_conditions_per_seed():
    assert len(ROSTER)==len(set(ROSTER))==18
    for seed in OPTIMIZATION_SEEDS:assert sum(n.startswith(f'seed_{seed}_') for n in ROSTER)==6


def test_parallel_choice_requires_exact_all_updates_models_speed_and_memory():
    a,b=phase('serial',100),phase('parallel',40)
    assert benchmark_decision(a,b)['selected_workers']==4
    b['wall_s']=90;assert benchmark_decision(a,b)['selected_workers']==1
    b['wall_s']=40;b['records'][0]['peak_rss_bytes']=9*1024**3
    assert benchmark_decision(a,b)['selected_workers']==1
    b['records'][0]['fit']['model_sha256']='different'
    with pytest.raises(ValueError,match='exactly'):benchmark_decision(a,b)
