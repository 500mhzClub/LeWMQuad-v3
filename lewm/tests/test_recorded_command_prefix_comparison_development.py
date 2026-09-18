from copy import deepcopy
import pytest
from lewm.recorded_command_prefix_comparison_development import compare


def rows():return [dict(requested_command=[0.,0.,.45],terminal=None) for _ in range(4)]


def test_last_proposal_is_not_counted_as_an_executed_difference():
    original=rows();shadow=deepcopy(original);shadow[-1]['requested_command']=[.2,0.,0.]
    r=compare(original,shadow,[r['requested_command'] for r in original[:-1]])
    assert r['common_executed_prefix_observations']==4 and r['all_recorded_commands_match']
    assert not r['last_observation_proposal_matches'] and not r['last_observation_command_was_executed']


def test_difference_or_terminal_limits_causal_prefix_before_affected_observation():
    original=rows();shadow=deepcopy(original);shadow[1]['requested_command']=[0.,0.,0.]
    tape=[r['requested_command'] for r in original[:-1]]
    assert compare(original,shadow,tape)['common_executed_prefix_observations']==2
    shadow[0]['terminal']='STOP'
    assert compare(original,shadow,tape)['common_executed_prefix_observations']==1
    tape[0]=[.2,0.,0.]
    with pytest.raises(ValueError,match='actual recorded'):compare(original,shadow,tape)
