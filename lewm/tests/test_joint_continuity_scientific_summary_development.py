"""Complete denominator and violation localization on synthetic scored rows."""
from copy import deepcopy
import pytest
from scripts.read_go2_joint_continuity_comparison_science_v1 import violations


def data():
    return [dict(frame=i, errors=dict(
        original=None if i>=100 else dict(position_m=.01,orientation_rad=.001),
        temporal_anchor=None if i>=200 else dict(position_m=.03 if i>=150 else .01,orientation_rad=.001)))
        for i in range(443)]


def test_accepted_violations_and_terminal_missing_rows_remain_distinct():
    result=violations(data())
    assert result['original']['available']==100
    assert result['original']['position_violations']==0
    assert result['temporal_anchor']['available']==200
    assert result['temporal_anchor']['position_violations']==50
    assert result['temporal_anchor']['first_position_violation']==dict(frame=150,error=.03)
    assert result['temporal_anchor']['first_orientation_violation'] is None


@pytest.mark.parametrize('fault',['truncated','extra','reordered'])
def test_no_partial_stream_can_appear_complete(fault):
    rows=data()
    if fault=='truncated':rows.pop()
    elif fault=='extra':rows.append(deepcopy(rows[-1]))
    else:rows[2],rows[3]=rows[3],rows[2]
    with pytest.raises(ValueError):violations(rows)
