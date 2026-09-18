from scripts import read_go2_observed_floor_contact_goal_probe_v1 as current
from scripts import read_go2_auxiliary_downward45_goal_probe_v1 as prior
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_same_model_comparison_ends_at_contact_policy_command_intervention():
    assert current.PRIOR==prior.INPUT and current.CASES==prior.CASES
    a,b=rows(63),rows(90);ta,tb=tape(62),tape(89)
    tb[42]['requested_command']=[.16,0.,-.45]
    assert current.common_prefix_length(a,b,ta,tb)==(43,42,None)


def test_terminal_boundary_is_preserved_when_zero_commands_match():
    a,b=rows(8),rows(12);ta,tb=tape(7),tape(11)
    for r in a[3:]:r['decision']['terminal']='NO_FEASIBLE'
    assert current.common_prefix_length(a,b,ta,tb)==(4,7,3)
