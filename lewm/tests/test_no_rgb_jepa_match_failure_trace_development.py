"""Original early registration failure and trace ownership; no native inputs."""
import sys
import numpy as np
import pytest
from scripts import diagnose_go2_no_rgb_jepa_maze02_matches_v1 as diagnosis


def test_original_insufficient_matches_reproduced_without_mutating_arrays():
    points = (np.ones((2,3)),np.ones((2,3)),np.ones((2,2)),np.ones((2,2)))
    before = [x.tobytes() for x in points]
    result = diagnosis.early_failure_trace(points,'insufficient rigid-pose matches')
    assert result['lifted_matches'] == 2 and result['valid_proposals'] == 0
    assert result['initial_consensus_points'] is None and result['final_consensus_points'] is None
    assert not result['converged_pose_available'] and not result['actual_gyro_gate_reexecuted']
    assert before == [x.tobytes() for x in points] and sys.gettrace() is None


def test_different_failure_is_not_relabelled_and_trace_is_restored():
    points = (np.ones((2,3)),np.ones((2,3)),np.ones((2,2)),np.ones((2,2)))
    with pytest.raises(ValueError,match='did not reproduce'):
        diagnosis.early_failure_trace(points,'insufficient rigid consensus after pruning')
    assert sys.gettrace() is None


def test_later_gyro_gate_cannot_be_claimed_with_identity_placeholder():
    with pytest.raises(ValueError,match='pre-gyro'):
        diagnosis.early_failure_trace((),'image and gyro reference rotations disagree beyond diagnostic envelope')


def test_existing_tracer_is_not_replaced():
    def tracer(*args):return tracer
    sys.settrace(tracer)
    try:
        with pytest.raises(ValueError,match='untraced'):
            diagnosis.early_failure_trace((),'insufficient rigid-pose matches')
        assert sys.gettrace() is tracer
    finally:sys.settrace(None)
