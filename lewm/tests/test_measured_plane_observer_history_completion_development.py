"""Serialized real synthetic poses, tampering, exact boundaries and live-owner rejection."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from scripts import verify_go2_measured_plane_observer_history_v1 as check
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin


def serialized(value):
    return json.loads(check.run.canonical(value))


@pytest.fixture(scope='module')
def actual_rows():
    run = check.run
    baseline, candidate = run.DualCameraVisualMotion(), run.MeasuredPlaneVisualMotion()
    old_reg, new_reg = run.MeasuredFloorTransportRegistration(), run.MeasuredFloorTransportRegistration()
    result = []
    for frame, item in enumerate(sequence(2)):
        policy, depth, fast, options = [move_test_origin(deepcopy(v)) for v in item]
        image, auxiliary, now = options['auxiliary_rgb'], options['auxiliary_depth'], options['now_ns']
        old = baseline.observe(policy, depth, fast, **options)
        registered, error = run.floor_observe(old_reg, policy, depth, auxiliary, old, now)
        live = candidate.observe(policy, depth, fast, **options)
        candidate_floor, failure = run.floor_observe(new_reg, policy, depth, auxiliary, live, now)
        assert error is None and failure is None
        public = policy, depth, fast, image, auxiliary
        recorded = dict(tick=frame, decision=dict(original_visual_evidence=old, evidence=registered,
            failure=None, requested_command=[0.,0.,0.]))
        comparison = dict(frame=frame, original_visual_exact=True, original_floor_exact_or_terminal_reproduced=True,
            candidate_visual_failure=None, candidate_floor_failure=None, stop_reason=None,
            raw_packet_sha256=run.fingerprint(public), original_requested_command=[0.,0.,0.], command_selected=False)
        row = dict(tick=frame, original=old, candidate=live, original_floor=registered,
            original_floor_error=None, candidate_floor=candidate_floor, comparison=comparison)
        result.append((serialized(row), serialized(recorded), public))
    return result


def test_complete_serialized_candidate_and_floor_witnesses_reconstruct(actual_rows):
    for frame, (row, recorded, public) in enumerate(actual_rows):
        before = deepcopy(row)
        assert check.check_row(row, recorded, frame=frame, public=public) == row['comparison']
        assert row == before


@pytest.mark.parametrize('fault', ['original_pose','candidate_pose','floor_correction','raw_hash','clock',
    'missing_floor','scope','row_identity','missing_field','extra_field'])
def test_tampered_complete_evidence_is_rejected(actual_rows, fault):
    original, recorded, public = actual_rows[1]
    row = deepcopy(original)
    if fault == 'original_pose': row['original']['current_pose']['position_initial_body_m'][0] += .01
    elif fault == 'candidate_pose': row['candidate']['current_pose']['position_initial_body_m'][0] += .01
    elif fault == 'floor_correction': row['candidate_floor']['floor_registration']['correction']['normal_translation_correction_m'] += .01
    elif fault == 'raw_hash': row['comparison']['raw_packet_sha256'] = '0'*64
    elif fault == 'clock': row['candidate']['decision_ns'] += 1
    elif fault == 'missing_floor': row['candidate_floor'] = None
    elif fault == 'scope': row['candidate']['reference_history_reset'] = True
    elif fault == 'row_identity': row['tick'] = True
    elif fault == 'missing_field': del row['comparison']['command_selected']
    elif fault == 'extra_field': row['unreviewed'] = True
    with pytest.raises((ValueError, KeyError)):
        check.check_row(row, recorded, frame=1, public=public)


def test_identity_restoration_is_narrow_and_does_not_mutate_nested_evidence():
    visual = dict(identity=[0,0,0], payload=['keep','list'])
    anchor = dict(identity=[0,0,0], original_visual_evidence=deepcopy(visual))
    floor = dict(identity=[0,0,0], original_visual_evidence=deepcopy(visual), floor_transport=dict(anchor=anchor))
    before = deepcopy(floor)
    restored = check.floor_identity(floor)
    assert restored['identity'] == (0,0,0)
    assert restored['floor_transport']['anchor']['original_visual_evidence']['identity'] == (0,0,0)
    assert restored['original_visual_evidence']['payload'] == ['keep','list']
    assert floor == before
    with pytest.raises(ValueError): check.visual_identity(dict(identity=[0,False,0]))
    with pytest.raises(ValueError): check.visual_identity(dict(identity=(0,0,0)))


@pytest.mark.parametrize('fault', ['live', 'wrong_boot'])
def test_original_owner_must_have_ended_on_the_recorded_boot(monkeypatch, fault):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    launch = dict(boot_id=boot if fault == 'live' else 'another boot', owner={'synthetic':'owner'})
    monkeypatch.setattr(check.run, 'owner_live', lambda owner: fault == 'live')
    with pytest.raises(ValueError, match='owner must be ended'):
        check.original_owner_ended(launch)


def test_report_preserves_negative_result_and_rejects_abbreviated_positive_history(actual_rows):
    row = deepcopy(actual_rows[1][0])
    row['comparison']['stop_reason'] = 'CANDIDATE_FLOOR_FAILURE'
    negative = check.reconstructed_report(2, 1, 0, row)
    assert negative['candidate_failure_preserved'] and not negative['complete_planned_history']
    assert not negative['navigation_recovered'] and not negative['native_completion_admitted']
    row['comparison']['stop_reason'] = 'FIXED_HISTORY_END'
    with pytest.raises(ValueError): check.reconstructed_report(2, 1, 0, row)
    row['comparison']['stop_reason'] = None
    with pytest.raises(ValueError): check.reconstructed_report(2, 1, 0, row)
