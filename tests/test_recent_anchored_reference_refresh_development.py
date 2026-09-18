import numpy as np
import pytest

from lewm.multi_reference_rgbd_pose_development import Reference
from lewm.recent_anchored_reference_refresh_development import RecentAnchoredReferenceRefreshPose


def fixture(*, anchored=True, promoted=False, age_ns=400_000_000):
    pose = RecentAnchoredReferenceRefreshPose()
    R = np.eye(3); p = np.zeros(3); features = object()
    ref = Reference(0, 0, object(), R, R, p)
    pose.references = [ref]; pose.stable_reference = ref; pose.stable_active = True
    pose.previous = Reference(4, age_ns, features, R, R, p)
    pose.frame = 4; pose._refresh_anchored_measurement = anchored
    pose.last_selection = {'selected_reference': 0}
    # Isolate reference-retention mechanics from actual depth-plane fitting.
    pose._prepare_plane = lambda *args: {'test_only': True}
    row = dict(frame=4, measured_ns=age_ns, promoted_keyframe=promoted,
        reference_frame=0, position_initial_body_m=p.tolist(),
        rotation_initial_body_from_current_body=R.tolist(), promotion_reason=None)
    return pose, row, ref


def test_accepted_anchored_refresh_retains_stable_reference_and_same_pose():
    pose, row, ref = fixture()
    result = pose._refresh_accepted_reference(row)
    assert result['promotion_reason'] == 'accepted_anchor_recent_reference_age'
    assert result['position_initial_body_m'] == row['position_initial_body_m']
    assert pose.stable_reference is ref and pose.references[-1].frame == 4
    assert pose.references[-1].features is pose.previous.features
    assert pose.nodes[-1]['parent_frame'] == 0


@pytest.mark.parametrize('kwargs', [dict(anchored=False), dict(promoted=True),
    dict(age_ns=399_999_999)])
def test_no_bridge_duplicate_or_early_promotion(kwargs):
    pose, row, ref = fixture(**kwargs)
    assert pose._refresh_accepted_reference(row) is row
    assert pose.references == [ref] and not pose.nodes


def test_existing_eight_reference_limit_is_preserved():
    pose, row, stable = fixture(age_ns=2_000_000_000)
    pose.references += [Reference(i, i*100_000_000, object(), np.eye(3), np.eye(3),
        np.zeros(3)) for i in range(1,8)]
    pose.frame = 20; pose.previous = Reference(20, 2_000_000_000, object(),
        np.eye(3), np.eye(3), np.zeros(3))
    row['frame'] = 20
    pose._refresh_accepted_reference(row)
    assert len(pose.references) == 8 and pose.stable_reference is stable
    assert [r.frame for r in pose.references] == [0,2,3,4,5,6,7,20]
