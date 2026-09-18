from copy import deepcopy
import pytest
from scripts.all_phase_fit_execution_development import (assignments, make_request, benchmark_decision,
    ROSTER, CACHE_BYTES, WORKER_RAM, NATIVE_RAM, PARENT_RAM, RESERVE, OUTPUT_ALLOWANCE)
from scripts.fit_go2_all_phase_models_v1 import capacity, SOURCE
from scripts.run_go2_prepared_native_queue_v1 import competing_command


def phase(name, wall):
    records = []
    for slot in range(3):
        jobs = []
        for row in assignments(name, slot, 1 if name=='serial' else 3):
            fit = dict(seed=row['seed'], condition='jepa', input_variant='full', benchmark=True,
                updates=20, model_sha256=str(row['seed']), initial_sha256='initial', schedule_sha256='schedule')
            jobs.append(dict(name=row['name'], fit=fit, actual_updates=20, ledger_sha256=str(row['seed'])))
        records.append(dict(name=f'{name}_{slot}', status='ALL_PHASE_FIT_WORKER_COMPLETE',
            warmed_contexts=4010, cache_bytes=CACHE_BYTES, peak_rss_bytes=8*1024**3, jobs=jobs))
    return dict(wall_s=wall, records=records)


def test_fixed_roster_partition_has_all_eighteen_fresh_fits_once():
    assert len(ROSTER)==18 and len({r['name'] for r in ROSTER})==18
    for workers in (1,3):
        jobs = [job for slot in range(workers) for job in assignments('fits',slot,workers)]
        assert sorted(j['name'] for j in jobs)==sorted(j['name'] for j in ROSTER)
        assert len(jobs)==18


@pytest.mark.parametrize('args', [('serial',0,3),('parallel',0,1),('fits',3,3),('fits',True,1),('other',0,1)])
def test_unassigned_process_or_phase_is_rejected(args):
    with pytest.raises(ValueError): assignments(*args)


def test_parallel_choice_requires_full_cache_equal_ledgers_and_measured_speedup():
    a,b = phase('serial',100),phase('parallel',40)
    assert benchmark_decision(a,b)['selected_workers']==3
    b['wall_s']=90
    assert benchmark_decision(a,b)['selected_workers']==1
    b['wall_s']=40; b['records'][0]['peak_rss_bytes']=WORKER_RAM+1
    assert benchmark_decision(a,b)['selected_workers']==1


@pytest.mark.parametrize('fault', ['model','ledger','updates','seed','cache','contexts','status','jobs','nan','zero'])
def test_incomplete_or_unequal_benchmark_cannot_admit_fitting(fault):
    a,b=phase('serial',100),phase('parallel',40); row=b['records'][0]; job=row['jobs'][0]
    if fault=='model':job['fit']['model_sha256']='changed'
    elif fault=='ledger':job['ledger_sha256']='changed'
    elif fault=='updates':job['actual_updates']=19
    elif fault=='seed':job['fit']['seed']+=1
    elif fault=='cache':row['cache_bytes']-=1
    elif fault=='contexts':row['warmed_contexts']-=1
    elif fault=='status':row['status']='FAILED'
    elif fault=='jobs':row['jobs'].pop()
    elif fault=='nan':b['wall_s']=float('nan')
    elif fault=='zero':b['wall_s']=0
    with pytest.raises(ValueError):benchmark_decision(a,b)


def test_resource_gate_accounts_for_concurrent_native_and_parent():
    resources=dict(memory_available_bytes=3*WORKER_RAM+NATIVE_RAM+PARENT_RAM,
        artifact_free_bytes=RESERVE+OUTPUT_ALLOWANCE)
    capacity(resources,3)
    with pytest.raises(ValueError):capacity(resources|dict(memory_available_bytes=resources['memory_available_bytes']-1),3)
    with pytest.raises(ValueError):capacity(resources|dict(artifact_free_bytes=RESERVE),1)
    capacity(resources|dict(memory_available_bytes=WORKER_RAM+NATIVE_RAM+PARENT_RAM),1)


def test_training_subprocess_does_not_claim_a_native_scene_worker():
    request=make_request('parallel',0,3,'a'*64)
    command=['python',SOURCE,'--phase','benchmark','--worker-request',request['name']+'_request.json']
    assert not competing_command(command)
    assert competing_command(['python','scripts/run_go2_recent_qualified_anchor_maze01_pilot_v1.py'])
    assert competing_command(['python','-c','from multiprocessing.spawn import spawn_main'])
