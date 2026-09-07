import ast
import copy
import inspect
import json

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_rgb_dataset_development import load_continuation_observation
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.tests.test_causal_rgb_dataset_development import create
from scripts.run_go2_observed_continuation_development_v1 import OUTPUT
import scripts.audit_go2_observed_continuation_development_v1 as original
import scripts.audit_go2_observed_continuation_development_v2 as corrected


def test_reader_body_differs_only_in_name_and_declared_population_bound():
    old = ast.parse(inspect.getsource(load_route_observation))
    new = ast.parse(inspect.getsource(load_continuation_observation))
    new.body[0].name = old.body[0].name
    changes = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Constant) and node.value == 806:
            node.value = 341
            changes += 1
    assert changes == 1
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_full_trial_auditor_differs_only_in_reader_call_not_scientific_checks():
    old = ast.parse(inspect.getsource(original.audit_trial))
    new = ast.parse(inspect.getsource(corrected.audit_trial))
    for node in ast.walk(new):
        if isinstance(node, ast.Name) and node.id == 'load_continuation_observation':
            node.id = 'load_route_observation'
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    assert inspect.getsource(original.scientific_decision) == inspect.getsource(corrected.scientific_decision)


@pytest.mark.parametrize('count', [342, 431, 806])
def test_long_population_uses_unchanged_packet_contract_and_legacy_stays_strict(tmp_path, count):
    manifest, arrays = create(tmp_path)
    manifest['schema'] = 'causal_rgb_body_routes_development.v1'
    manifest['frames'] = [{**manifest['frames'][0], 'rgb_file': f'rgb_{i:04d}.png'} for i in range(count)]
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    np.savez_compressed(tmp_path/'policy_histories.npz', **{k: np.repeat(v, count, axis=0) for k, v in arrays.items()})
    assert load_continuation_observation(tmp_path, 0)['image']['rgb'].shape == (480, 640, 3)
    with pytest.raises(SensorContractError, match='population'): load_route_observation(tmp_path, 0)


@pytest.mark.parametrize('fault', ['excess_frames', 'oracle_field', 'future_tensor', 'path_escape', 'history_path', 'boolean_index'])
def test_larger_episode_budget_does_not_relax_policy_boundary(tmp_path, fault):
    manifest, arrays = create(tmp_path)
    manifest['schema'] = 'causal_rgb_body_routes_development.v1'
    if fault == 'excess_frames': manifest['frames'] *= 807
    elif fault == 'oracle_field': manifest['world_pose'] = [0.]*7
    elif fault == 'future_tensor': arrays['future_contact'] = np.zeros(1)
    elif fault == 'path_escape': manifest['frames'][0]['rgb_file'] = '../rgb_0000.png'
    elif fault == 'history_path': manifest['history_file'] = 'physics_trace.npz'
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    np.savez_compressed(tmp_path/'policy_histories.npz', **arrays)
    with pytest.raises(SensorContractError): load_continuation_observation(tmp_path, True if fault == 'boolean_index' else 0)


def assert_equal(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: assert_equal(a[key], b[key])
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    else: assert a == b


@pytest.mark.parametrize('index', [0, 40, 91])
def test_actual_short_episode_packets_identical_between_readers(index):
    directory = OUTPUT/'observed-continuation-development-v1-00-fixed_forward'
    assert_equal(load_route_observation(directory, index), load_continuation_observation(directory, index))


@pytest.mark.parametrize('index', [0, 340, 341, 430])
def test_actual_long_episode_access_reaches_all_previously_rejected_boundaries(index):
    directory = OUTPUT/'observed-continuation-development-v1-01-fixed_forward'
    value = load_continuation_observation(directory, index)
    manifest = json.loads((directory/'policy_observations.json').read_text())
    assert len(manifest['frames']) == 431
    assert value['image']['measured_ns'] == manifest['frames'][index]['image_ns']
    assert set(value) == {'image', 'sensor_state'}
    with pytest.raises(SensorContractError, match='population'): load_route_observation(directory, index)
