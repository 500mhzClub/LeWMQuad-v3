import numpy as np
import pytest

from lewm.counterfactual_learning_data_development import prospective_plan


def test_known_plan_reconstructs_slew_without_reading_future_commands():
    result=prospective_plan([.3,0,.5],[0,0,0])
    assert result.shape==(40,3)
    assert result[0]==pytest.approx([.25,0,.35])
    assert result[1]==pytest.approx([.3,0,.5])
    assert np.allclose(result[1:],result[1])


def test_previous_command_is_required_for_correct_candidate_execution():
    result=prospective_plan([.3,0,.5],[-.2,0,-.5])
    assert result[0]==pytest.approx([.05,0,-.15])
    assert result[1]==pytest.approx([.3,0,.2])


@pytest.mark.parametrize('command',[[1,0,0],[0,.1,0],[0,0,float('nan')],[0,0]])
def test_invalid_candidate_is_rejected(command):
    with pytest.raises(ValueError): prospective_plan(command,[0,0,0])
