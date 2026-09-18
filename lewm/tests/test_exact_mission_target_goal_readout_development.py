from scripts import read_go2_exact_mission_target_goal_probe_v1 as current
from scripts import read_go2_observed_floor_contact_goal_probe_v1 as prior
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_changed_target_command_limits_causal_comparison():
    assert current.PRIOR==prior.INPUT and current.CASES==prior.CASES
    a,b=rows(237),rows(220);ta,tb=tape(236),tape(219)
    tb[170]['requested_command']=[.16,0.,-.45]
    assert current.common_prefix_length(a,b,ta,tb)==(171,170,None)


def test_arrival_terminal_difference_also_limits_comparison():
    a,b=rows(237),rows(220);ta,tb=tape(236),tape(219)
    for r in b[205:]:r['decision']['terminal']='OBSERVED_GOAL_CANDIDATE'
    assert current.common_prefix_length(a,b,ta,tb)==(206,219,205)
