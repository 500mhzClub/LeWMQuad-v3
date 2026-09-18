from scripts.read_go2_observation_horizon_fits_v1 import baseline_scope
from scripts.read_go2_augmented_family_switch_fits_v1 import summarize_cells


def test_only_matching_physical_horizons_are_compared():
    assert baseline_scope('half_second')=='first_half_second'
    assert baseline_scope('switch_half_second')=='switch_first_half_second'
    for scope in ('all','first_observation','moving_first_observation','unknown_half_second'):
        assert baseline_scope(scope) is None


def test_undefined_yaw_remains_unknown_in_shared_summary():
    r=summarize_cells([dict(motion_targets=1,contact_targets=2,contact_positives=1,
        undefined_yaw=1,position_error_m=.01,yaw_error_rad=None,contact_brier=.2)])
    assert r['yaw_error_rad'] is None and r['motion_targets']==1 and r['contact_targets']==2
