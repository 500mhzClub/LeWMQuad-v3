from scripts import read_go2_causal_residual_final_goal_probe_v1 as current
from scripts import read_go2_executed_horizon_final_goal_probe_v1 as prior
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows, tape


def test_causal_correction_command_limits_comparison():
    assert current.PRIOR == prior.INPUT and current.CASES == prior.CASES
    a, b = rows(254), rows(220); ta, tb = tape(253), tape(219)
    tb[171]['requested_command'] = [.16, 0., -.45]
    assert current.common_prefix_length(a, b, ta, tb) == (172, 171, None)


def test_arrival_terminal_difference_limits_comparison_before_tape_end():
    a, b = rows(254), rows(220); ta, tb = tape(253), tape(219)
    for r in b[205:]: r['decision']['terminal'] = 'OBSERVED_GOAL_CANDIDATE'
    assert current.common_prefix_length(a, b, ta, tb) == (206, 219, 205)
