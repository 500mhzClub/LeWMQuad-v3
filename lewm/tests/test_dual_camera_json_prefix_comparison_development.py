from copy import deepcopy
import json
import pytest
from lewm.dual_camera_json_prefix_comparison_development import compare_json_primary_decision
from lewm.tests.test_dual_camera_controller_prefix_comparison_development import initial


def serialized(initial):
    original,candidate,p,image,a,now=initial
    return json.loads(json.dumps(original)),json.loads(json.dumps(candidate)),p,image,a,now


def test_actual_serialized_controller_decisions_pass_without_mutating_identity(initial):
    original,candidate,p,image,a,now=serialized(initial)
    before=deepcopy((original,candidate))
    result=compare_json_primary_decision(original,candidate,p,image,a,now_ns=now)
    assert result['complete_decision_exact_outside_added_auxiliary_metadata']
    assert (original,candidate)==before
    assert type(candidate['original_visual_evidence']['identity']) is list


@pytest.mark.parametrize('identity',[[True,0,0],[-1,0,0],[0,0],[0,0,1]])
def test_json_restoration_does_not_admit_malformed_or_changed_episode(initial,identity):
    original,candidate,p,image,a,now=serialized(initial)
    candidate['original_visual_evidence']['identity']=identity
    with pytest.raises(ValueError):compare_json_primary_decision(original,candidate,p,image,a,now_ns=now)


def test_json_command_difference_still_rejected(initial):
    original,candidate,p,image,a,now=serialized(initial)
    candidate['requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError):compare_json_primary_decision(original,candidate,p,image,a,now_ns=now)
