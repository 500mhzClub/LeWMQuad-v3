from lewm.current_reserve_terminal_feedback_development import select_reserved_terminal
from lewm.rollout_selection_off_development import select_current_clearance


def test_same_reserve_applies_to_turns_and_translations_but_not_hold():
    for view in (None,.8):
        result=select_reserved_terminal([1.,.3],scan_error=view,clearance_m=.479)
        assert result['current_nominal_disk_clear']
        assert not result['current_action_reserve_clear']
        assert result['action']=='hold'
        assert [r['action'] for r in result['candidates'] if r['eligible']]==['hold']
        assert all(r['required_current_clearance_m']==.48 for r in result['candidates'][1:])


def test_terminal_positive_distance_progress_replaces_heading_turn():
    args=dict(pulse=True,clearance_m=.6)
    old=select_current_clearance([.01,.03],**args)
    new=select_reserved_terminal([.01,.03],**args)
    assert old['action']=='left_turn'
    assert new['action']=='forward'
    assert new['current_reserve_terminal']['position_priority_changed']
    assert not new['learned_candidate_rollouts_used_for_selection']


def test_terminal_behind_uses_heading_and_views_never_translate():
    behind=select_reserved_terminal([-.03,.01],pulse=True,clearance_m=.6)
    assert behind['action']=='left_turn'
    assert not behind['current_reserve_terminal']['position_priority_changed']
    view=select_reserved_terminal([.01,.03],scan_error=.5,pulse=True,clearance_m=.6)
    assert view['action']=='left_turn'
    assert not view['current_reserve_terminal']['terminal_position_priority_active']


def test_no_terminal_override_outside_approach_and_no_reserve_bypass():
    assert select_reserved_terminal([.01,.03],pulse=False,clearance_m=.6)['action']==select_current_clearance([.01,.03],clearance_m=.6)['action']
    assert select_reserved_terminal([.01,.03],pulse=True,clearance_m=.479)['action']=='hold'
