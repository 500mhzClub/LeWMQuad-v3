"""All-eighteen ledger, raw-score and snapshot admission for expanded fits.

This reader grants no fitting resume or native execution. It only reloads a
named final model after reconstructing the complete fixed study accounting.
"""
import json
import numpy as np
from lewm.all_phase_training_schedule_development import schedule as build_schedule, SEEDS
from lewm.observation_horizon_fit_development import score
from scripts.all_phase_study_inputs_development import authenticate, stream
from scripts.all_phase_fit_execution_development import OUTPUT, BENCH, ROSTER, science, make_request, benchmark_decision
from scripts.observation_horizon_snapshot_development import load_snapshot
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch


def verify_ledger(rows, schedule, *, variant, model_sha256):
    count = 0; last = None
    for count, row in enumerate(rows, 1):
        if (count > 1200 or row['update'] != count
                or row['sample_indices'] != schedule['batches'][count-1]
                or row['schedule_sha256'] != schedule['schedule_sha256']
                or row['input_variant'] != variant):
            raise ValueError('every exact expanded-context optimizer receipt required')
        last = row
    if count != 1200 or last['model_sha256'] != model_sha256:
        raise ValueError('complete final ledger/model identity required')


def admit(result_sha256):
    verify_artifacts(OUTPUT, {'result.json':result_sha256}); result = read_json(OUTPUT, 'result.json')
    if (result['status'] != 'ALL_PHASE_EIGHTEEN_FITS_COMPLETE' or result['optimizer_updates'] != 21600
            or result['scientific_models_trained'] != 18 or result['checkpoint_selection_performed'] is not False
            or result['benchmark_weights_reused'] is not False or result['native_execution'] is not False):
        raise ValueError('complete eighteen fresh unselected expanded-data fits required')
    verify_artifacts(OUTPUT, result['artifact_sha256'])
    launch = read_json(OUTPUT, 'launch.json'); verify_ordered_launch(launch)
    if (launch['source_sha256'] != result['source_sha256'] or launch['science'] != result['science']
            or launch['selected_workers'] != result['selected_workers'] or launch['phase'] != 'fits'):
        raise ValueError('unchanged source/science/worker assignment required')
    verify_artifacts(BENCH, launch['benchmark_artifact_sha256'])
    bench = read_json(BENCH, 'result.json'); verify_ordered_launch(read_json(BENCH, 'launch.json'))
    if (launch['benchmark_artifact_sha256']['result.json'] != launch['benchmark_result_sha256']
            or bench['status'] != 'ALL_PHASE_FIT_BENCHMARK_COMPLETE'
            or bench['source_sha256'] != result['source_sha256'] or bench['science'] != result['science']
            or bench['decision'] != benchmark_decision(bench['serial'], bench['parallel'])
            or result['selected_workers'] != bench['decision']['selected_workers']):
        raise ValueError('exact successful full-cache execution benchmark required')
    authenticate(); data = stream(maximum_cache_bytes=0)
    schedules = {str(seed):build_schedule(data.view, seed=seed) for seed in SEEDS}
    if read_json(OUTPUT, 'training_schedules.json') != schedules or launch['science'] != science(schedules):
        raise ValueError('all exact admitted expanded-context schedules required')
    workers = result['selected_workers']; records = result['records']
    if workers not in (1,3) or len(records) != workers:
        raise ValueError('complete selected process assignment required')
    launch_sha = result['artifact_sha256']['launch.json']; snapshots = {}; fitted_files = {}
    initial = {seed:set() for seed in SEEDS}; total = 0
    for slot, record in enumerate(records):
        request = make_request('fits', slot, workers, launch_sha)
        if (record['name'] != request['name'] or record['status'] != 'ALL_PHASE_FIT_WORKER_COMPLETE'
                or read_json(OUTPUT, request['name']+'_request.json') != request
                or read_json(OUTPUT, request['name']+'_terminal.json') != record
                or len(record['jobs']) != len(request['jobs'])
                or record['actual_updates'] != 1200*len(request['jobs'])):
            raise ValueError('exact complete original subprocess requests and terminals required')
        verify_artifacts(OUTPUT, record['artifact_sha256'])
        for job, assignment in zip(record['jobs'], request['jobs'], strict=True):
            name = assignment['name']; prefix = request['name']+'_'+name
            if name in snapshots or job['name'] != name or job['actual_updates'] != 1200:
                raise ValueError('every assigned fresh model exactly once required')
            fitted = read_json(OUTPUT, prefix+'_fit.json'); fit = fitted['fit']; snapshot = fitted['snapshot']
            seed, variant, condition = (assignment[k] for k in ('seed','variant','condition'))
            schedule = schedules[str(seed)]
            expected_binding = dict(experiment_sha256=launch_sha,
                dataset_sha256=launch['science']['dataset_pair_sha256'],
                schedule_sha256=schedule['schedule_sha256'], input_variant=variant)
            expected_config = dict(condition=condition, seed=seed, latent_dim=32,
                learning_rate=.001, ema_momentum=.99, updates=1200)
            if (fit != job['fit'] or fit['condition'] != condition or fit['seed'] != seed
                    or fit['input_variant'] != variant or fit['updates'] != 1200 or fit['benchmark'] is not False
                    or fit['schedule_sha256'] != schedule['schedule_sha256']
                    or snapshot['binding'] != expected_binding or snapshot['configuration'] != expected_config
                    or snapshot['filename'] != prefix+'.pt' or snapshot['model_sha256'] != fit['model_sha256']
                    or snapshot['sha256'] != result['artifact_sha256'][prefix+'.pt']):
                raise ValueError('exact assigned model/treatment/schedule/snapshot configuration required')
            ledger_name = prefix+'_updates.jsonl'
            if (fitted['ledger_sha256'] != job['ledger_sha256']
                    or job['ledger_sha256'] != result['artifact_sha256'][ledger_name]):
                raise ValueError('bound complete optimizer ledger required')
            with (OUTPUT/ledger_name).open() as ledger:
                verify_ledger((json.loads(line) for line in ledger), schedule,
                    variant=variant, model_sha256=fit['model_sha256'])
            complete = read_json(OUTPUT, prefix+'_prediction_phase_complete.json')
            if (complete['model_sha256'] != fit['model_sha256']
                    or set(complete['prediction_sha256']) != {'train','geometry_transfer'}):
                raise ValueError('both complete prediction roles required before scoring')
            for role in ('train','geometry_transfer'):
                filename = prefix+'_'+role+'.npz'
                if complete['prediction_sha256'][role] != result['artifact_sha256'][filename]:
                    raise ValueError('exact raw prediction identity required')
                with np.load(OUTPUT/filename, allow_pickle=False) as saved: arrays = {k:saved[k] for k in saved.files}
                head = 'direct_outcomes' if condition=='direct' else 'rollout_outcomes'
                if score(data.view, arrays, role=role, head=head) != read_json(OUTPUT, prefix+'_'+role+'_scores.json'):
                    raise ValueError('complete expanded-training/original-transfer raw score reproduction required')
            clone = load_snapshot(OUTPUT, snapshot['filename'], sha256=snapshot['sha256'],
                expected_binding=expected_binding, expected_config=expected_config)
            del clone
            initial[seed].add(fit['initial_sha256']); total += 1200
            snapshots[name] = snapshot; fitted_files[name] = prefix+'_fit.json'
    if (set(snapshots) != {r['name'] for r in ROSTER} or total != 21600
            or any(len(v)!=1 for v in initial.values()) or len(set.union(*initial.values())) != 3):
        raise ValueError('all eighteen models, all21600 updates and three paired initial states required')
    authenticate(); verify_ordered_launch(launch)
    verify_artifacts(OUTPUT, result['artifact_sha256'] | {'result.json':result_sha256})
    return dict(study_result_sha256=result_sha256, fit_artifact_sha256=result['artifact_sha256'],
        snapshots=snapshots, fitted_files=fitted_files,
        all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed=True, optimizer_updates=21600,
        primary_native_candidate=launch['science']['primary_native_candidate'],
        checkpoint_selection_performed=False, final_evaluation=False, navigation_qualified=False)


def load_assigned(admission, name):
    if (set(admission['snapshots']) != {r['name'] for r in ROSTER} or name not in admission['snapshots']
            or admission['all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed'] is not True):
        raise ValueError('complete expanded-data model admission and explicit assigned name required')
    verify_artifacts(OUTPUT, admission['fit_artifact_sha256'] | {'result.json':admission['study_result_sha256']})
    snapshot = read_json(OUTPUT, admission['fitted_files'][name])['snapshot']
    assigned = next(row for row in ROSTER if row['name']==name)
    if (snapshot != admission['snapshots'][name] or snapshot['configuration']['condition'] != assigned['condition']
            or snapshot['configuration']['seed'] != assigned['seed'] or snapshot['binding']['input_variant'] != assigned['variant']):
        raise ValueError('assigned final model identity changed')
    clone = load_snapshot(OUTPUT, snapshot['filename'], sha256=snapshot['sha256'],
        expected_binding=snapshot['binding'], expected_config=snapshot['configuration'])
    return clone.model, assigned['condition'], assigned['variant']
