"""Boundary checks prevent a diagnostic from claiming an unexecuted future."""
from copy import deepcopy
import pytest
from scripts.replay_go2_no_rgb_jepa_direct_flow_observer_prefix_v1 import compare


def evidence(frame=0):
    return dict(decision_ns=1_500_000_000+frame*100_000_000, status='CURRENT_VISUAL_POSE',
                current_pose={'position': [0., 0., 0.]}, identity=(0,0,0))


def test_original_receipt_mismatch_rejected_even_when_candidate_matches_recording():
    recorded = evidence(); original = deepcopy(recorded); original['current_pose']['position'][0] = .01
    with pytest.raises(ValueError, match='did not reproduce'): compare(recorded, original, recorded, frame=0)


def test_changed_pose_stops_immediately_without_mutating_receipts():
    original = evidence(); candidate = deepcopy(original)
    candidate['current_pose']['position'][0] = .01
    candidate['direct_corner_flow_fallback'] = {'accepted': True}
    before = deepcopy(candidate)
    check = compare(original, original, candidate, frame=0)
    assert check['stop'] and check['stop_reason'] == 'FIRST_CHANGED_OBSERVER_EVIDENCE'
    assert candidate == before


def test_unexplained_change_cannot_be_metadata_normalized():
    original = evidence(); candidate = deepcopy(original); candidate['new_unreviewed_flag'] = True
    with pytest.raises(ValueError, match='unexplained'): compare(original, original, candidate, frame=0)


@pytest.mark.parametrize('which', ['original', 'candidate'])
def test_either_terminal_stops(which):
    original = evidence(); candidate = deepcopy(original)
    target = original if which == 'original' else candidate
    target['status'] = 'VISUAL_TERMINAL_FAILURE'; target['current_pose'] = None
    candidate['direct_corner_flow_fallback'] = {'accepted': False}
    assert compare(original, original, candidate, frame=0)['stop']


def test_json_tuple_normalization_retains_all_other_fields():
    original = evidence(); recorded = deepcopy(original); recorded['identity'] = [0,0,0]
    check = compare(recorded, original, original, frame=0)
    assert check['candidate_original_fields_exact'] and not check['stop']


@pytest.mark.parametrize('which', [0, 1, 2])
def test_each_receipt_clock_is_checked(which):
    values = [evidence() for _ in range(3)]; values[which]['decision_ns'] += 1
    with pytest.raises(ValueError, match='clock'): compare(*values, frame=0)


def test_fixed_end_is_terminal_for_replay_even_with_unchanged_pose():
    value = evidence(859)
    check = compare(value, value, value, frame=859)
    assert check['stop'] and check['stop_reason'] == 'FIXED_PREFIX_LIMIT'
