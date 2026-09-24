import pytest
from scripts.read_go2_augmented_family_switch_fits_v1 import summarize_cells


def cell(n,error,*,undefined=0):
    return dict(motion_targets=n,contact_targets=3,contact_positives=1,undefined_yaw=undefined,
        position_error_m=error,yaw_error_rad=None if undefined or error is None else error/2,contact_brier=.2)


def test_weighted_denominators_and_undefined_yaw_are_not_filtered():
    a,b=cell(2,.1),cell(1,.4);r=summarize_cells([a,b])
    assert r['motion_targets']==3 and r['contact_targets']==6 and r['contact_positives']==2
    assert r['position_error_m']==pytest.approx(.2) and r['yaw_error_rad']==pytest.approx(.1)
    b=cell(1,.4,undefined=1);r=summarize_cells([a,b])
    assert r['motion_targets']==3 and r['undefined_yaw']==1 and r['yaw_error_rad'] is None
    assert r['position_error_m']==pytest.approx(.2)


def test_no_motion_target_stays_unknown_with_contact_denominator_intact():
    r=summarize_cells([cell(0,None)])
    assert r['position_error_m'] is None and r['yaw_error_rad'] is None and r['contact_targets']==3
