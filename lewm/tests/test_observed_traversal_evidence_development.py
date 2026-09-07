import copy
import json

import pytest

from lewm.observed_traversal_controller_development import ObservedTraversalController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.online_temporal_choice_development import OnlineTemporalChoice
from lewm.tests.test_observed_traversal_controller_development import Stream
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.audit_go2_observed_traversal_development_v1 import scientific_decision


@pytest.mark.parametrize('method', ['direct_direct', 'supervised_rollout', 'jepa_rollout'])
def test_actual_frozen_models_consume_four_current_frames_and_hold_selected_tape(method):
    template = OnlineTemporalChoice.from_completed_study(method)
    controller = ObservedTraversalController(method, ArticulatedCollisionGeometry(URDF), template)
    stream, command, selections = Stream(), [0., 0., 0.], []
    for tick in range(9):
        packet, fast, now = stream.frame(tick, command)
        result = controller.observe(packet, fast, now_ns=now)
        assert not result['terminal']
        assert json.loads(json.dumps(result, allow_nan=False)) == result
        if result['selection'] is not None:
            selection = result['selection']
            assert selection['method'] == method
            assert len(selection['input_images']) == 4 and len(selection['member_predictions']) == 3
            assert selection['input_images'][-1]['measured_ns'] == now
            assert selection['model_bindings'] == template.bindings
            command = selection['requested_command_tape'][0]
            selections.append(selection)
        assert result['requested_command'] == command
    assert len(selections) == 2
    assert selections[1]['decision_ns']-selections[0]['decision_ns'] == 500_000_000


def test_replay_excludes_only_two_wall_clock_measurements_not_predictions():
    value = {'status': 'TRAVERSING', 'selection': {'inference_ms': 1., 'adapter_ms': 2.,
              'member_predictions': [[[.1]]], 'selected_action_name': 'forward'}}
    changed = copy.deepcopy(value)
    changed['selection']['inference_ms'] = 7.
    assert scientific_decision(changed) == scientific_decision(value)
    changed['selection']['member_predictions'][0][0][0] += .00001
    assert scientific_decision(changed) != scientific_decision(value)
    assert 'inference_ms' in value['selection']
    changed['selection']['adapter_ms'] = -1.
    with pytest.raises(ValueError): scientific_decision(changed)
