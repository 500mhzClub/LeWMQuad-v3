"""Match two already authenticated final fits for a prospective JEPA comparison.

This pure metadata check supplements the existing eighteen-fit/correction
artifact admission. It never replaces byte verification, loads a checkpoint,
executes a controller, or establishes a navigation advantage.
"""
from copy import deepcopy
import math

SEED = 2026091001
CONDITIONS = ('jepa', 'supervised_rollout')
NAMES = tuple(f'seed_{SEED}_full_{c}' for c in CONDITIONS)
FIT_RESULT = '45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418'
CORRECTION_RESULT = 'a425d3ab1398df9312663e35dee665e5800153bd60d29334ae71a1d95e4f21d5'
INITIAL_STATE = 'ed2c1f096b430b424cf6e381047eeee7672934bcac2fdbe198d5f85c2cead607'
SNAPSHOT_SHA = (
    'abf5272ab9cde930dd5408a3ae46a426bcf8453a896013c17de9d3c87d9091e8',
    '1c3eb0b1b02f6c594cccf68f68e9f56490d6c126bec496fcdff9ff004393b9bb')
BASE_STATE = (
    '859d691240da9cb0314cf3501f960f32c67293949a18f2af796b734c34363b95',
    '8ff02dc34f6af9cc34b9a25f17d49aad9bfa821dabc15861c7ae0d6da10853e4')
COMMON_BINDING = dict(
    experiment_sha256='58ef5b311211edc3a0bead90d0399f941553eb3d348386b44103a7c2a512f710',
    dataset_sha256='a592deeea6493b43e90ebeb8914d0c73bb2eb70c80865e6411384692ddc53a08',
    schedule_sha256='73592f86381cd0195c9596732e7226502fe1c72005436820b7b577a4b175b0bb',
    input_variant='full')
COMMON_CONFIG = dict(seed=SEED, latent_dim=32, learning_rate=.001,
                     ema_momentum=.99, updates=1200)
HEADS = ('direct_outcomes', 'rollout_outcomes')
CORRECTION_DESIGN = ('estimator', 'training_examples', 'training_draws',
                     'motion_counts', 'effective_horizon_weights', 'target_offsets_ns')


def admit_objective_pair(admission, fitted):
    """Require the fixed first-seed full-RGB JEPA/supervised-rollout pair.

Callers must authenticate admission and each fitted JSON against their frozen
artifact bindings before this check, then reload the assigned snapshots through
the existing evaluation-only loader. Result records are private copies.
"""
    base = admission['base_admission']
    if (admission['correction_result_sha256'] != CORRECTION_RESULT
            or base['study_result_sha256'] != FIT_RESULT
            or admission['all_coefficients_reconstructed'] is not True
            or admission['all_models'] != 18 or admission['all_trained_heads'] != 30
            or base['all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed'] is not True
            or base['optimizer_updates'] != 21600
            or base['checkpoint_selection_performed'] is not False
            or len(base['snapshots']) != 18 or len(admission['coefficients']) != 18
            or set(fitted) != set(NAMES)):
        raise ValueError('complete frozen fit/correction admission and fixed objective pair required')
    records = []
    designs = []
    for i, (name, condition) in enumerate(zip(NAMES, CONDITIONS, strict=True)):
        snapshot = base['snapshots'][name]
        fit = fitted[name]
        if (fit['snapshot'] != snapshot
                or snapshot['filename'] != name+'.pt' or snapshot['sha256'] != SNAPSHOT_SHA[i]
                or snapshot['model_sha256'] != BASE_STATE[i]
                or snapshot['binding'] != COMMON_BINDING
                or snapshot['configuration'] != COMMON_CONFIG | dict(condition=condition)
                or snapshot['evaluation_only_reload_verified'] is not True
                or snapshot['training_resume_authorized'] is not False
                or snapshot['checkpoint_selection_performed'] is not False
                or fit['fit'] != dict(updates=1200, initial_sha256=INITIAL_STATE,
                    model_sha256=BASE_STATE[i], benchmark=False, input_variant='full',
                    condition=condition, seed=SEED)):
            raise ValueError('same initialization/data/schedule/configuration and exact final snapshot required')
        correction = admission['coefficients'][name]
        if (correction['base_model_sha256'] != BASE_STATE[i]
                or correction['condition'] != condition or correction['variant'] != 'full'
                or correction['seed'] != SEED or set(correction['heads']) != set(HEADS)):
            raise ValueError('same trained heads and correctly assigned training-only correction required')
        design = {}
        for head in HEADS:
            row = correction['heads'][head]
            bias = row['applied_bias_xy_m']
            if (row['schema'] != 'training_translation_bias.v1' or row['head'] != head
                    or row['estimator'] != 'draw_count_divided_by_window_motion_count_weighted_mean'
                    or row['training_examples'] != 408 or row['training_draws'] != 7200
                    or row['fitted_scalar_parameters'] != 16
                    or row['target_offsets_ns'] != [100_000_000*h for h in range(1, 9)]
                    or any(row[k] is not False for k in (
                        'yaw_changed', 'contact_changed', 'probability_calibrated', 'native_data_used'))
                    or not isinstance(bias, list) or len(bias) != 8
                    or any(not isinstance(xy, list) or len(xy) != 2 or any(
                        type(x) not in (int, float) or not math.isfinite(x) for x in xy) for xy in bias)):
                raise ValueError('finite eight-horizon training-only XY intercepts required')
            design[head] = {k:deepcopy(row[k]) for k in CORRECTION_DESIGN}
        designs.append(design)
        records.append(dict(name=name, condition=condition, variant='full', head='rollout_outcomes',
            snapshot=deepcopy(snapshot), correction=deepcopy(correction)))
    if designs[0] != designs[1]:
        raise ValueError('identical correction estimator, training population and weighting required')
    return dict(models=records, same_initial_state_sha256=INITIAL_STATE,
        shared_binding=deepcopy(COMMON_BINDING), shared_configuration=deepcopy(COMMON_CONFIG),
        comparison='jepa_vs_supervised_rollout_with_same_rollout_head',
        per_model_training_only_intercepts_retained=True, correction_values_required_equal=False,
        native_execution=False, checkpoint_loaded=False, model_training=False,
        model_selected_using_new_maze_outcomes=False, jepa_advantage_established=False,
        generalization_or_optimization_seed_variation_established=False,
        artifact_authentication_still_required=True)
