import copy
import json

import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.observed_continuation_development import ObservedContinuation
from lewm.online_temporal_choice_development import OnlineTemporalChoice
from lewm.tests.test_observed_continuation_development import MovingStream
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.audit_go2_observed_continuation_development_v1 import scientific_decision


@pytest.mark.parametrize('method', ['direct_direct', 'supervised_rollout', 'jepa_rollout'])
def test_frozen_ensembles_execute_inside_continuation_without_new_model_inputs(method):
    template = OnlineTemporalChoice.from_completed_study(method)
    controller = ObservedContinuation(method, ArticulatedCollisionGeometry(URDF), template)
    stream, command, selected = MovingStream(), [0., 0., 0.], []
    for tick in range(9):
        packet, fast, now = stream.frame(tick, command)
        result = controller.observe(packet, fast, now_ns=now)
        assert result['stage'] == 'FIRST' and not result['terminal']
        assert json.loads(json.dumps(result, allow_nan=False)) == result
        command = result['requested_command']
        selection = result['child']['selection']
        if selection is not None:
            selected.append(selection)
            assert selection['model_bindings'] == template.bindings
            assert selection['input_images'][-1]['measured_ns'] == now
            assert len(selection['member_predictions']) == 3
    assert len(selected) == 2
    assert selected[1]['decision_ns']-selected[0]['decision_ns'] == 500_000_000


def test_nested_replay_never_discards_scientific_or_primitive_fields():
    result = {'status': 'RUNNING', 'child': {'selection': {'method': 'jepa_rollout',
        'inference_ms': 1., 'adapter_ms': 2., 'member_predictions': [1., 2.]}}}
    changed = copy.deepcopy(result)
    changed['child']['selection']['adapter_ms'] = 3.
    assert scientific_decision(changed) == scientific_decision(result)
    changed['child']['selection']['member_predictions'][0] = 99.
    assert scientific_decision(changed) != scientific_decision(result)
    primitive = {'child': {'selection': {'method': 'fixed_forward', 'learned_prediction_used': False,
                                      'requested_command_tape': [[.3, 0., 0.]]*5}}}
    assert scientific_decision(primitive) == primitive
    primitive['child']['selection']['learned_prediction_used'] = True
    with pytest.raises(ValueError): scientific_decision(primitive)
