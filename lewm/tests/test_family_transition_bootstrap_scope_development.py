"""New scope retains scientific failures and rejects invalid measurement input."""
from copy import deepcopy
import pytest
from lewm.family_transition_bootstrap_scope_development import admit_scope


def fixture():
    c = dict(status='GEOMETRY_PROGRESS_FAMILY_COLLECTION_AND_AUDIT_COMPLETE', episodes=96,
        all_measurement_gates_pass=True, training_design_informative=False, episode_exclusions=[],
        mirrored_siblings_are_independent=False, independent_maze_evaluation_layouts=0,
        model_trained=False, navigation_qualified=False, goal_achieved=False)
    d = dict(status='GEOMETRY_PROGRESS_FAMILY_CAUSAL_READOUT_COMPLETE',
        ready_for_separately_frozen_learning=False, total_planned_windows=768, total_materialized_windows=684,
        initial_tensor_equivalence_checked=96, moving_contexts_do_not_prove_replanning=True,
        raw_accounting=dict(frames=3740, physics_samples=254726, footprint_frames=3740,
            stable_interior_failed_frames=0, near_occlusion_failed_frames=0),
        model_trained=False, navigation_qualified=False, goal_achieved=False)
    return c, d


def test_new_scope_preserves_old_failure_and_does_not_mutate_evidence():
    c, d = fixture(); before = deepcopy((c, d))
    scope = admit_scope(c, d)
    assert (c, d) == before
    assert scope['measured_transitions_admitted']
    for key in ('original_task_design_pass', 'original_learning_readiness', 'original_task_scope_reclassified',
            'navigation_qualified', 'independent_maze_evaluation', 'final_evaluation'):
        assert scope[key] is False


@pytest.mark.parametrize('fault', ['measurements', 'old_design', 'old_readiness', 'exclusion',
    'missing_context', 'raw_population', 'near_occlusion', 'promotion'])
def test_invalid_measurements_or_relabelled_science_are_rejected(fault):
    c, d = fixture()
    if fault == 'measurements': c['all_measurement_gates_pass'] = False
    if fault == 'old_design': c['training_design_informative'] = True
    if fault == 'old_readiness': d['ready_for_separately_frozen_learning'] = True
    if fault == 'exclusion': c['episode_exclusions'] = ['a']
    if fault == 'missing_context': d['total_materialized_windows'] -= 1
    if fault == 'raw_population': d['raw_accounting']['frames'] -= 1
    if fault == 'near_occlusion': d['raw_accounting']['near_occlusion_failed_frames'] = 1
    if fault == 'promotion': d['navigation_qualified'] = True
    with pytest.raises(ValueError): admit_scope(c, d)
