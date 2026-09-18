from scripts import read_go2_auxiliary_depth_reobserve_goal_probe_v1 as current
from scripts import read_go2_auxiliary_depth_goal_probe_v1 as prior
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_active_wait_intervention_precedes_same_zero_command_tape_end():
    assert current.PRIOR==prior.INPUT and current.CASES==prior.CASES
    a,b=rows(8),rows(12);ta,tb=tape(7),tape(11)
    for r in a[3:]:r['decision']['terminal']='NO_FEASIBLE'
    # Commands still agree. The older terminal at tick 3 ends the shared
    # controller-state prefix before the tape lengths first differ at tick 7.
    assert current.common_prefix_length(a,b,ta,tb)==(4,7,3)
