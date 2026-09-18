"""Explicit new transition-prediction scope, preserving the failed task assay.

This scope uses independently verified motion/contact measurements as training
targets. It does not admit the old task-discrimination experiment or grant a
navigation, calibration, independent-maze or final-evaluation claim.
"""


def admit_scope(collection, derivation):
    if (collection['status'] != 'GEOMETRY_PROGRESS_FAMILY_COLLECTION_AND_AUDIT_COMPLETE'
            or collection['episodes'] != 96 or collection['all_measurement_gates_pass'] is not True
            or collection['training_design_informative'] is not False
            or collection['episode_exclusions'] != []
            or collection['mirrored_siblings_are_independent'] is not False
            or collection['independent_maze_evaluation_layouts'] != 0):
        raise ValueError('complete measured population with preserved failed task-design scope required')
    if (derivation['status'] != 'GEOMETRY_PROGRESS_FAMILY_CAUSAL_READOUT_COMPLETE'
            or derivation['ready_for_separately_frozen_learning'] is not False
            or derivation['total_planned_windows'] != 768 or derivation['total_materialized_windows'] != 684
            or derivation['initial_tensor_equivalence_checked'] != 96
            or derivation['moving_contexts_do_not_prove_replanning'] is not True):
        raise ValueError('complete causal population with unchanged old readiness result required')
    raw = derivation['raw_accounting']
    if (raw['frames'] != 3740 or raw['physics_samples'] != 254726
            or raw['footprint_frames'] != 3740 or raw['stable_interior_failed_frames'] != 0
            or raw['near_occlusion_failed_frames'] != 0):
        raise ValueError('complete passing physical sensor measurements required')
    for result in (collection, derivation):
        if any(result[k] is not False for k in ('model_trained', 'navigation_qualified', 'goal_achieved')):
            raise ValueError('unchanged untrained and unpromoted predecessor required')
    return dict(scope='family_transition_prediction_bootstrap_v1', measured_transitions_admitted=True,
        original_task_design_pass=False, original_learning_readiness=False,
        original_task_scope_reclassified=False, outcome_conditioned_sampling=False,
        fit_role='train', development_scoring_role='geometry_transfer',
        independent_maze_evaluation=False, final_evaluation=False, navigation_qualified=False)
