"""Reject confounded objective assignments and preserve input evidence."""
from copy import deepcopy
import pytest
from lewm import matched_rollout_objective_admission_development as pair


def evidence():
    snapshots = {}; coefficients = {}; fitted = {}
    for i, (name, condition) in enumerate(zip(pair.NAMES, pair.CONDITIONS, strict=True)):
        snapshot = dict(filename=name+'.pt', sha256=pair.SNAPSHOT_SHA[i], model_sha256=pair.BASE_STATE[i],
            binding=deepcopy(pair.COMMON_BINDING), configuration=pair.COMMON_CONFIG | dict(condition=condition),
            evaluation_only_reload_verified=True, training_resume_authorized=False,
            checkpoint_selection_performed=False)
        fit = dict(updates=1200, initial_sha256=pair.INITIAL_STATE, model_sha256=pair.BASE_STATE[i],
            benchmark=False, input_variant='full', condition=condition, seed=pair.SEED)
        fitted[name] = dict(snapshot=deepcopy(snapshot), fit=fit)
        snapshots[name] = snapshot
        heads = {}
        for head in pair.HEADS:
            heads[head] = dict(schema='training_translation_bias.v1', head=head,
                estimator='draw_count_divided_by_window_motion_count_weighted_mean',
                training_examples=408, training_draws=7200, motion_counts=[408]*8,
                effective_horizon_weights=[900.]*8, target_offsets_ns=[h*100_000_000 for h in range(1,9)],
                fitted_scalar_parameters=16, yaw_changed=False, contact_changed=False,
                probability_calibrated=False, native_data_used=False,
                applied_bias_xy_m=[[.01*i, -.001*i] for _ in range(8)])
        coefficients[name] = dict(base_model_sha256=pair.BASE_STATE[i], condition=condition,
            variant='full', seed=pair.SEED, heads=heads)
    # The pair helper supplements, rather than re-runs, the eighteen-fit admission.
    for i in range(16): snapshots[f'other_{i}'] = {}; coefficients[f'other_{i}'] = {}
    admission = dict(correction_result_sha256=pair.CORRECTION_RESULT,
        all_coefficients_reconstructed=True, all_models=18, all_trained_heads=30,
        coefficients=coefficients, base_admission=dict(study_result_sha256=pair.FIT_RESULT,
            all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed=True,
            optimizer_updates=21600, checkpoint_selection_performed=False, snapshots=snapshots))
    return admission, fitted


def test_same_rollout_head_with_distinct_fixed_weights_and_training_intercepts():
    admission, fitted = evidence(); original = deepcopy((admission, fitted))
    result = pair.admit_objective_pair(admission, fitted)
    assert (admission, fitted) == original
    assert [r['condition'] for r in result['models']] == ['jepa', 'supervised_rollout']
    assert {r['head'] for r in result['models']} == {'rollout_outcomes'}
    assert result['models'][0]['correction'] != result['models'][1]['correction']
    assert result['native_execution'] is False and result['jepa_advantage_established'] is False
    assert result['artifact_authentication_still_required'] is True
    result['models'][0]['snapshot']['binding']['input_variant'] = 'changed'
    assert (admission, fitted) == original


@pytest.mark.parametrize('fault', ['missing_model', 'extra_model', 'incomplete_ledger',
    'changed_fit_result', 'selected_checkpoint', 'different_initialization', 'different_schedule',
    'different_rgb', 'different_updates', 'direct_head_model', 'wrong_snapshot', 'wrong_base',
    'wrong_correction_assignment', 'native_fitted_bias', 'changed_contact', 'different_correction_population',
    'different_correction_weights', 'nonfinite_bias', 'wrong_horizon', 'missing_trained_head'])
def test_rejects_confounding_or_unauthenticated_pair_metadata(fault):
    admission, fitted = evidence(); name = pair.NAMES[1]
    base = admission['base_admission']; snap = base['snapshots'][name]
    fit = fitted[name]['fit']; correction = admission['coefficients'][name]
    head = correction['heads']['rollout_outcomes']
    if fault == 'missing_model': fitted.pop(name)
    elif fault == 'extra_model': fitted['chosen_third_model'] = deepcopy(fitted[name])
    elif fault == 'incomplete_ledger': base['all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed'] = False
    elif fault == 'changed_fit_result': base['study_result_sha256'] = '0'*64
    elif fault == 'selected_checkpoint': base['checkpoint_selection_performed'] = True
    elif fault == 'different_initialization': fit['initial_sha256'] = '0'*64
    elif fault == 'different_schedule': snap['binding']['schedule_sha256'] = '0'*64
    elif fault == 'different_rgb': snap['binding']['input_variant'] = 'no_rgb'
    elif fault == 'different_updates': snap['configuration']['updates'] = 1199
    elif fault == 'direct_head_model': fit['condition'] = 'direct'
    elif fault == 'wrong_snapshot': snap['sha256'] = pair.SNAPSHOT_SHA[0]
    elif fault == 'wrong_base': snap['model_sha256'] = pair.BASE_STATE[0]
    elif fault == 'wrong_correction_assignment': correction['base_model_sha256'] = pair.BASE_STATE[0]
    elif fault == 'native_fitted_bias': head['native_data_used'] = True
    elif fault == 'changed_contact': head['contact_changed'] = True
    elif fault == 'different_correction_population': head['motion_counts'][0] -= 1
    elif fault == 'different_correction_weights': head['effective_horizon_weights'][0] += 1
    elif fault == 'nonfinite_bias': head['applied_bias_xy_m'][0][0] = float('nan')
    elif fault == 'wrong_horizon': head['target_offsets_ns'][0] = 500_000_000
    elif fault == 'missing_trained_head': correction['heads'].pop('direct_outcomes')
    if name in fitted: fitted[name]['snapshot'] = deepcopy(snap)
    with pytest.raises(ValueError): pair.admit_objective_pair(admission, fitted)
