from copy import deepcopy
from lewm.full_reserve_heading_release_development import release_heading_turn
from lewm.matched_heading_release_development import compare_heading_release
from lewm.tests.test_no_early_heading_release_development import releasable


def test_matched_control_reproduces_original_and_ablation_retains_recovery():
    selected, state = releasable()
    before, previous = deepcopy(selected), deepcopy(state)
    expected, cleared = release_heading_turn(selected, state)
    control, control_state = compare_heading_release(selected, state, suppress=False)
    variant, variant_state = compare_heading_release(selected, state, suppress=True)
    assert {k:v for k,v in control.items() if k != 'heading_release_comparison'} == expected
    assert control_state is cleared is None
    assert {k:v for k,v in variant.items() if k != 'heading_release_comparison'} == selected
    assert variant_state is state
    assert control['heading_release_comparison']['proposed'] == variant['heading_release_comparison']['proposed']
    assert not control['heading_release_comparison']['suppressed']
    assert variant['heading_release_comparison']['suppressed']
    assert selected == before and state == previous


def test_both_modes_are_identical_when_no_recovery_is_active():
    selected, _ = releasable()
    for suppress in (False, True):
        result, state = compare_heading_release(selected, None, suppress=suppress)
        assert result is selected and state is None
        assert 'heading_release_comparison' not in result
