from scripts import read_go2_auxiliary_downward45_goal_probe_v1 as current
from scripts import read_go2_auxiliary_depth_reobserve_goal_probe_v1 as prior
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_calibration_intervention_comparison_stops_at_first_actual_command_difference():
    assert current.PRIOR==prior.INPUT and current.CASES==prior.CASES
    a,b=rows(30),rows(40);ta,tb=tape(29),tape(39)
    tb[17]['requested_command']=[0.,0.,-.45]
    assert current.common_prefix_length(a,b,ta,tb)==(18,17,None)


def test_terminal_intervention_can_precede_command_difference():
    a,b=rows(8),rows(12);ta,tb=tape(7),tape(11)
    for r in a[3:]:r['decision']['terminal']='NO_FEASIBLE'
    assert current.common_prefix_length(a,b,ta,tb)==(4,7,3)
