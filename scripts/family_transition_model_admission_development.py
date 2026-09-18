"""Admit the fixed full-JEPA final model only from all six completed family fits."""
import json
from lewm.family_transition_fit_development import score
from scripts.run_go2_family_transition_fits_v1 import OUTPUT, ROSTER
from scripts.family_transition_fit_inputs_development import authenticate, stream, CHECK_SHA
from scripts.cumulative_pulse_snapshot_development import load_snapshot
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
import numpy as np

NAME = 'seed_2026091001_full_jepa'


def admit(result_sha256):
    verify_artifacts(OUTPUT, {'result.json': result_sha256})
    result = read_json(OUTPUT, 'result.json')
    if (result['status'] != 'FAMILY_TRANSITION_SIX_FITS_COMPLETE' or result['optimizer_updates'] != 7200
            or set(r['name'] for r in result['records']) != set(ROSTER) or len(result['records']) != 6
            or result['checkpoint_selection_performed'] is not False or result['benchmark_weights_reused'] is not False):
        raise ValueError('all six complete unselected fresh family fits required')
    verify_artifacts(OUTPUT, result['artifact_sha256'])
    launch = read_json(OUTPUT, 'launch.json'); verify(launch)
    _, _, schedule = authenticate()
    view = stream().view
    initial = set()
    for record in result['records']:
        name = record['name']; request = read_json(OUTPUT, name+'_request.json')
        fitted = read_json(OUTPUT, name+'_fit.json'); snapshot = fitted['snapshot']
        if (record['status'] != 'FAMILY_TRANSITION_WORKER_COMPLETE' or record['actual_updates'] != 1200
                or request['benchmark'] is not False or request['seed'] != 2026091001
                or request['science'] != launch['science'] or fitted['fit'] != record['fit']
                or snapshot['binding']['dataset_sha256'] != CHECK_SHA
                or snapshot['binding']['experiment_sha256'] != result['artifact_sha256']['launch.json']
                or snapshot['binding']['schedule_sha256'] != schedule['schedule_sha256']
                or snapshot['binding']['input_variant'] != request['variant']
                or snapshot['configuration'] != dict(condition=request['condition'], seed=2026091001,
                    latent_dim=32, learning_rate=.001, ema_momentum=.99, updates=1200)):
            raise ValueError('complete matched fixed-configuration fit accounting required')
        initial.add(fitted['fit']['initial_sha256'])
        count = 0
        with (OUTPUT/(name+'_updates.jsonl')).open() as ledger:
            for count, line in enumerate(ledger, 1):
                row = json.loads(line)
                if (count > 1200 or row['update'] != count or row['sample_indices'] != schedule['batches'][count-1]
                        or row['schedule_sha256'] != schedule['schedule_sha256'] or row['input_variant'] != request['variant']):
                    raise ValueError('exact every-step sample/treatment ledger required')
        if count != 1200 or row['model_sha256'] != fitted['fit']['model_sha256'] or snapshot['model_sha256'] != row['model_sha256']:
            raise ValueError('final ledger/snapshot model identity mismatch')
        for role in ('train', 'geometry_transfer'):
            with np.load(OUTPUT/(name+'_'+role+'.npz'), allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in saved.files}
            head = 'direct_outcomes' if request['condition'] == 'direct' else 'rollout_outcomes'
            if score(view, arrays, role=role, head=head) != read_json(OUTPUT, name+'_'+role+'_scores.json'):
                raise ValueError('whole-role raw score reconstruction failed')
    if len(initial) != 1: raise ValueError('paired initial model identity required')
    snapshot = read_json(OUTPUT, NAME+'_fit.json')['snapshot']
    clone = load_snapshot(OUTPUT, snapshot['filename'], sha256=snapshot['sha256'],
        expected_binding=snapshot['binding'], expected_config=snapshot['configuration'])
    verify_artifacts(OUTPUT, {'result.json': result_sha256, **result['artifact_sha256']})
    return clone.model, dict(study_result_sha256=result_sha256, selected_name=NAME, snapshot=snapshot,
        selection_rule='fixed first full-JEPA seed before fit results', all_six_ledgers_and_raw_scores_reconstructed=True,
        final_evaluation=False, navigation_qualified=False)
