from scripts.read_go2_augmented_family_switch_goal_probe_v1 import common_prefix_length


def rows(n):return [dict(decision=dict(terminal=None)) for _ in range(n)]
def tape(n):return [dict(requested_command=[0,0,0]) for _ in range(n)]


def test_comparison_includes_precommand_observation_but_no_affected_future():
    a,b=rows(8),rows(8);ta,tb=tape(7),tape(7);tb[3]['requested_command']=[.2,0,0]
    assert common_prefix_length(a,b,ta,tb)==(4,3,None)
    b[2]['decision']['terminal']='STOP'
    assert common_prefix_length(a,b,ta,tb)==(3,3,2)


def test_shorter_terminal_tape_cannot_compare_unexecuted_future():
    assert common_prefix_length(rows(5),rows(8),tape(4),tape(7))==(5,4,None)
    assert common_prefix_length(rows(8),rows(8),tape(7),tape(7))==(8,None,None)
