"""Actual 96-stream scoring/verification on synthetic packets, not native runs."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from lewm.independent_tracking_challenge_development import TRIALS, specification
from lewm.independent_tracking_coverage_development import measured_coverage
from lewm.tests.test_independent_tracking_stress_cohort_development import prepared, template
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts import independent_tracking_evaluation_development as scoring
from scripts import independent_tracking_scored_population_verification_development as check


@pytest.fixture
def scored(prepared, monkeypatch):
    store, c, b, s, phase = prepared
    # Explicit synthetic stand-ins: this component authenticates their bytes,
    # but deliberately does not claim worker/launch semantic verification.
    extras = check.expected_names() - store.allowed
    store.allowed |= extras
    store.save('launch.json', dict(synthetic_fixture=True))
    for name in sorted(extras):
        store.save(name, dict(synthetic_fixture=True))
    calls = []

    def audit(output, trial, result, protocol):
        calls.append(trial)
        n = result['physics_samples']; pose = np.zeros((n, 7)); pose[:, 6] = 1.
        raw = dict(timestamp_s=np.arange(1, n + 1) * .002,
            base_pose_world=pose, base_twist_world=np.zeros((n, 6)))
        coverage = measured_coverage(specification(trial)['direction'], *raw.values(),
            **{k: result[k] for k in ('completed_ticks', 'schedule_complete', 'physical_stop', 'acquisition_stop')})
        return raw, dict(coverage=coverage, sensors=dict(depth_checks=[]), synthetic_raw_auditor=True)

    monkeypatch.setattr(scoring, '_raw_audit', audit)
    result = stress.evaluate_complete_population(store, c, b, s, 'a' * 64)
    calls.clear()
    return store, result, calls


def replace_json(store, name, value):
    payload = base.encode(value)
    (store.output / name).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_complete_96_stream_reconstruction_retains_negative_coverage_and_unexercised_stress(scored):
    store, result, calls = scored
    before = dict(store.hashes)
    out = check.verify_scored_population(store.output, store.hashes['result.json'], 'a' * 64)
    assert calls == list(TRIALS)  # Exactly eight audits, not 96.
    assert out['pose_stream_count'] == 96
    assert not out['verified_aggregate_claims']['all_intended_motion_covered']
    assert not out['verified_aggregate_claims']['strict_depth_visibility_pass']
    for trial in TRIALS:
        report = out['reports'][trial]
        assert len(report['pose_streams']) == 12
        assert all(v['numerical_pose_score_reconstruction_verified'] for v in report['pose_streams'].values())
        assert all(v['available_bridge_and_rejoin_history_verified'] for v in report['continuity'].values())
        assert all(not v['onset_recorded'] for v in report['stress_exposure'].values())
    for key in ('source_and_launch_authority_verified', 'outside_keeper_terminal_verified',
                'predecessor_comparison_verified', 'full_challenge_pass', 'navigation_qualified',
                'sensor_transform_and_raw_audit_algorithms_independent', 'goal_achieved'):
        assert out[key] is False
    base.verify_artifacts(store.output, before)  # Reader did not mutate outputs.


@pytest.mark.parametrize('fault', ['missing_stream', 'phase_binding', 'promoted', 'protocol'])
def test_population_and_authority_mismatch_reject_before_raw_access(scored, fault):
    store, result, calls = scored
    bad = deepcopy(result)
    if fault == 'missing_stream': del bad['output_sha256'][TRIALS[-1] + '_evaluation.jsonl']
    elif fault == 'phase_binding': bad['stress_phase_sha256'] = 'f' * 64
    elif fault == 'promoted': bad['navigation_qualified'] = True
    else: bad['protocol_sha256'] = 'b' * 64
    sha = replace_json(store, 'result.json', bad)
    with pytest.raises(ValueError): check.verify_scored_population(store.output, sha, 'a' * 64)
    assert calls == []


@pytest.mark.parametrize('fault', ['pose_score', 'evaluation_row', 'raw_audit', 'aggregate'])
def test_rebound_scientific_corruption_is_not_hidden_by_valid_hashes(scored, fault):
    store, result, calls = scored
    bad = deepcopy(result); trial = TRIALS[0]
    if fault == 'aggregate': bad['all_intended_motion_covered'] = True
    elif fault == 'evaluation_row':
        name = trial + '_evaluation.jsonl'
        rows = list(stress.rows(store.output, name)); rows[0]['frame'] = 1
        payload = b''.join(base.encode(r) for r in rows)
        (store.output / name).write_bytes(payload)
        bad['output_sha256'][name] = hashlib.sha256(payload).hexdigest()
    else:
        name = trial + '_audit.json'; audit = base.read(store.output, name)
        if fault == 'raw_audit': audit['raw_audit']['coverage']['intended_motion_covered'] = True
        else:
            bad['scores'][trial]['arms']['temporal_anchor']['position_m']['maximum'] += .01
            audit['base_pose_score'] = deepcopy(bad['scores'][trial])
        bad['output_sha256'][name] = replace_json(store, name, audit)
    sha = replace_json(store, 'result.json', bad)
    with pytest.raises(ValueError): check.verify_scored_population(store.output, sha, 'a' * 64)
    assert calls  # Actual separately coded numerical/raw comparison was reached.
