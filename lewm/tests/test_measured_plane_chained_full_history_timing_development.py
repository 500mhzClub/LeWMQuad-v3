"""Chained timing accounting retains complete history and exact tracking state."""
from copy import deepcopy

import pytest

from scripts import measured_plane_chained_full_history_timing_development as comparison
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from lewm.tests.test_measured_plane_full_history_timing_development import population


def test_full_population_including_original_failure_region_and_final_state():
    rows, states = population(3124)
    report = comparison.summarize(rows, states, frames=3124,
        model_sha='a'*64, input_result_sha='b'*64)
    assert report['baseline'] == 'MeasuredPlaneChainedAnchorController'
    assert report['candidate'] == 'MeasuredPlaneChainedSinglePassController'
    assert report['frames'] == 3124
    assert report['observed_state_checks'][-1]['frame'] == 3123
    assert report['actual_model_forward_calls'] == [3121, 3121]
    assert not report['motion_and_mission_types_normalized']
    assert not report['chained_tracking_state_normalized']
    assert not report['navigation_outcomes_inferred']
    assert not report['native_adoption_performed']
    assert not report['real_time_qualified']
    for modified in (rows[:3113], rows[:3113]+rows[3114:]):
        with pytest.raises(ValueError, match='all admitted observations'):
            comparison.summarize(modified, states, frames=3124,
                model_sha='a'*64, input_result_sha='b'*64)


@pytest.mark.parametrize('fault', ['changed_decision', 'changed_calls', 'missing_final_state'])
def test_chained_accounting_rejects_incomplete_or_changed_evidence(fault):
    rows, states = population()
    if fault == 'changed_decision': rows[5]['normalized_candidate_decision_sha256'] = 'changed'
    elif fault == 'changed_calls': rows[5]['actual_model_forward_calls'] = [1, 0]
    else: states.pop()
    with pytest.raises(ValueError):
        comparison.summarize(rows, states, frames=8,
            model_sha='a'*64, input_result_sha='b'*64)


def test_exact_chained_types_and_complete_motion_fields_are_preserved():
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorController
    from lewm.measured_plane_chained_single_pass_controller_development import MeasuredPlaneChainedSinglePassController
    from lewm.measured_plane_single_pass_controller_development import MeasuredPlaneSinglePassController
    from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
    from scripts.analyze_go2_ground_plane_development_v1 import URDF

    def make(cls):
        return cls(FixedHeadModel('direct'), ArticulatedCollisionGeometry(URDF),
            public_mission=deepcopy(MISSION), navigation_ticks=40,
            condition='direct', variant='no_rgb', persistent=True)

    baseline, candidate = [make(cls) for cls in (
        MeasuredPlaneChainedAnchorController, MeasuredPlaneChainedSinglePassController)]
    before = fingerprint(comparison.observed_state(baseline))
    assert fingerprint(comparison.observed_state(candidate)) == before
    candidate.motion.synthetic_changed_tracking_witness = {'accepted': True, 'reference': 7}
    assert fingerprint(comparison.observed_state(candidate)) != before
    del candidate.motion.synthetic_changed_tracking_witness
    assert fingerprint(comparison.observed_state(candidate)) == before
    unchained = make(MeasuredPlaneSinglePassController)
    with pytest.raises(ValueError, match='exact chained'):
        comparison.observed_state(unchained)
    candidate.motion = unchained.motion
    with pytest.raises(ValueError, match='exact chained'):
        comparison.observed_state(candidate)
