"""Real observer bridge/rejoin rows on synthetic sensor packets; no native run."""
from copy import deepcopy
from functools import lru_cache
import numpy as np
import pytest

from lewm.independent_tracking_continuity_verification_development import verify_continuity
from lewm.tests.test_temporal_anchor_continuity_replay_development import bridge_factory
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
from lewm.tests.test_independent_tracking_stress_cohort_development import shifted
from scripts.replay_go2_temporal_anchor_continuity_v1 import ContinuityAudit, serial


@lru_cache(maxsize=8)
def actual_rows(ending):
    count = {'rejoin': 5, 'recording_end': 4, 'terminal': 14, 'multiple': 8}[ending]
    missing = {2, 3} if ending == 'rejoin' else {2, 3, 5, 6} if ending == 'multiple' else set(range(2, count))
    model = bridge_factory(missing)(); rows = []; audit = ContinuityAudit()
    for i, item in enumerate(packets([texture()] * count)):
        p, d, f, now = shifted(item); r = model.observe(p, d, f, now_ns=now)
        row = dict(frame=i, measured_ns=now, arms={'temporal_anchor': dict(
            pose=serial(r['current_pose']), selection=serial(r['reference_selection']),
            failure=serial(r['terminal_failure']), continuity=serial(r['continuity_evidence']))})
        rows.append(row); audit.observe(row)
    return rows, audit.summary()


@pytest.mark.parametrize('ending', ['rejoin', 'recording_end', 'terminal', 'multiple'])
def test_actual_observer_histories_match_independent_reconstruction(ending):
    rows, summary = deepcopy(actual_rows(ending)); before = deepcopy((rows, summary))
    result = verify_continuity(iter(rows), summary)
    assert (rows, summary) == before and result['summary'] == summary
    assert result['available_bridge_and_rejoin_history_verified']
    assert not result['rotation_disagreements_independently_recomputed']
    assert not result['shared_gyro_bias_correction_established'] and not result['full_challenge_pass']
    if ending == 'terminal': assert summary['unavailable_frames'] == 2 and summary['total_bridged_frames'] == 10
    if ending == 'multiple': assert len(summary['bridge_spans']) == 2 and summary['total_bridged_frames'] == 4


@pytest.mark.parametrize('fault', ['clock', 'promotion', 'reference', 'selection', 'count', 'total', 'path',
    'position', 'reset', 'new_initial', 'independence', 'error_bound', 'retained_count', 'keyframe_count',
    'unavailable_agreement', 'rejoin_distance', 'rejoin_angle', 'rejoin_history', 'rejoin_path',
    'rejoin_reference', 'summary', 'missing_row'])
def test_modified_bridge_or_rejoin_evidence_is_rejected(fault):
    rows, summary = deepcopy(actual_rows('rejoin'))
    arm = rows[2]['arms']['temporal_anchor']; e = arm['continuity']; p = arm['pose']
    anchor = rows[4]['arms']['temporal_anchor']; a = anchor['continuity']
    if fault == 'clock': rows[2]['measured_ns'] += 1
    elif fault == 'promotion': p['promoted_keyframe'] = True
    elif fault == 'reference': p['reference_frame'] = 0
    elif fault == 'selection': arm['selection']['selected_reference_retained_anchor'] = True
    elif fault == 'count': e['bridge_frames'] += 1
    elif fault == 'total': e['total_bridge_frames'] += 1
    elif fault == 'path': e['bridge_path_m'] += .01
    elif fault == 'position': e['incremental_position_initial_body_m'][0] += .01
    elif fault == 'reset': p['global_history_reset'] = True
    elif fault == 'new_initial': e['status'] = 'INITIAL_REFERENCE'
    elif fault == 'independence': e['measurements_independent'] = True
    elif fault == 'error_bound': e['error_bound_m'] = .001
    elif fault == 'retained_count': arm['selection']['retained_references'] += 1
    elif fault == 'keyframe_count': p['keyframe_count'] += 1
    elif fault == 'unavailable_agreement': e['disagreement_m'] = 0.
    elif fault == 'rejoin_distance': a['disagreement_m'] += .001
    elif fault == 'rejoin_angle': a['disagreement_rad'] = .101
    elif fault == 'rejoin_history': a['preceding_bridge_frames'] = 0
    elif fault == 'rejoin_path': a['preceding_bridge_path_m'] += .01
    elif fault == 'rejoin_reference': anchor['pose']['reference_frame'] = 3; anchor['selection']['selected_reference'] = 3
    elif fault == 'summary': summary['bridge_spans'][0]['outcome'] = 'END_OF_RECORDING'
    elif fault == 'missing_row': rows.pop()
    with pytest.raises(ValueError): verify_continuity(rows, summary)


@pytest.mark.parametrize('fault', ['resurrection', 'rewritten_reason', 'false_budget', 'missing_reason'])
def test_terminal_failure_is_retained_without_recovery(fault):
    rows, summary = deepcopy(actual_rows('terminal'))
    if fault == 'resurrection': rows[13]['arms']['temporal_anchor']['pose'] = rows[11]['arms']['temporal_anchor']['pose']
    elif fault == 'rewritten_reason': rows[13]['arms']['temporal_anchor']['failure'] = 'different failure'
    elif fault == 'false_budget': rows[2]['arms']['temporal_anchor']['pose'] = None; rows[2]['arms']['temporal_anchor']['failure'] = 'failure'; rows[2]['arms']['temporal_anchor']['continuity']['status'] = 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    elif fault == 'missing_reason': rows[12]['arms']['temporal_anchor']['failure'] = None
    with pytest.raises(ValueError): verify_continuity(rows, summary)


def test_no_rows_has_no_success_or_bridge_claim():
    summary = ContinuityAudit().summary()
    report = verify_continuity([], summary)
    assert report['summary']['frames'] == report['summary']['total_bridged_frames'] == 0
    assert not report['full_challenge_pass']


def test_shared_gyro_rotation_agreement_is_algebraic_not_independent_heading_evidence():
    from scipy.spatial.transform import Rotation
    # Both retained anchors and the preceding frame use the same gyro trajectory.
    # Different noncommuting reference orientations still lead to the same G_now.
    now = Rotation.from_rotvec([.04, -.03, .08]).as_matrix()
    candidates = []
    for reference in (np.eye(3), Rotation.from_rotvec([.1, .2, -.3]).as_matrix()):
        relative_gyro = reference.T @ now
        candidates.append(reference @ relative_gyro)
    np.testing.assert_allclose(candidates[0], candidates[1], atol=1e-14)
    assert Rotation.from_matrix(candidates[0].T @ candidates[1]).magnitude() < 1e-14
    assert Rotation.from_matrix(candidates[0]).magnitude() > .09  # Could be wrong versus static truth.


def test_rotation_scalar_inside_gate_is_explicitly_not_independently_verified():
    rows, summary = deepcopy(actual_rows('rejoin'))
    rows[4]['arms']['temporal_anchor']['continuity']['disagreement_rad'] = .05
    report = verify_continuity(rows, summary)
    assert report['range_checked_reported_rotation_disagreements'] > 0
    assert not report['rotation_disagreements_independently_recomputed']
    assert not report['incremental_rotation_witness_saved']


def test_actual_observer_promotion_and_eight_reference_retention(monkeypatch):
    import lewm.temporal_anchor_continuity_development as model_module
    # Synthetic intervention on promotion choice only; no frozen source edit.
    monkeypatch.setattr(model_module, 'support_near_limit', lambda registration: True)
    model = model_module.TemporalAnchorVisualLedMotion(); audit = ContinuityAudit(); rows = []
    for i, item in enumerate(packets([texture()] * 12)):
        p, d, f, now = shifted(item); r = model.observe(p, d, f, now_ns=now)
        row = dict(frame=i, measured_ns=now, arms={'temporal_anchor': dict(
            pose=serial(r['current_pose']), selection=serial(r['reference_selection']),
            failure=serial(r['terminal_failure']), continuity=serial(r['continuity_evidence']))})
        rows.append(row); audit.observe(row)
    report = verify_continuity(rows, audit.summary())
    assert rows[-1]['arms']['temporal_anchor']['pose']['keyframe_count'] == 12
    assert rows[-1]['arms']['temporal_anchor']['selection']['retained_references'] == 8
    assert report['summary']['total_bridged_frames'] == 0


def test_failure_before_initial_reference_retains_all_unavailable_rows():
    rows = [dict(frame=i, measured_ns=1_500_000_000 + i * 100_000_000,
        arms={'temporal_anchor': dict(pose=None, failure='missing initial RGB',
            continuity=dict(status='VALIDATING_CURRENT_INPUT'))}) for i in range(3)]
    audit = ContinuityAudit()
    for row in rows: audit.observe(row)
    report = verify_continuity(rows, audit.summary())
    assert report['summary']['unavailable_frames'] == 3
    assert report['summary']['first_failure']['frame'] == 0
    assert not report['failure_causes_independently_reconstructed']
