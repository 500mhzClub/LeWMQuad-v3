import ast
import inspect
import json

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_rgb_dataset_development import load_continuation_observation
from lewm.whole_task_rgb_dataset_development import load_whole_task_observation
from lewm.tests.test_causal_rgb_dataset_development import create


def test_only_reader_name_and_episode_population_cap_change():
    old = ast.parse(inspect.getsource(load_continuation_observation))
    new = ast.parse(inspect.getsource(load_whole_task_observation))
    new.body[0].name = old.body[0].name
    count = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Constant) and node.value == 3606:
            node.value = 806; count += 1
    assert count == 1 and ast.dump(new) == ast.dump(old)


@pytest.mark.parametrize('count', [807, 2406, 3606, 3607])
def test_fixed_whole_task_frame_budget(tmp_path, count):
    manifest, arrays = create(tmp_path)
    manifest['schema'] = 'causal_rgb_body_routes_development.v1'
    manifest['frames'] = [{**manifest['frames'][0], 'rgb_file': f'rgb_{i:04d}.png'} for i in range(count)]
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    np.savez_compressed(tmp_path/'policy_histories.npz', **{k: np.repeat(v, count, axis=0) for k, v in arrays.items()})
    if count > 3606:
        with pytest.raises(SensorContractError, match='population'): load_whole_task_observation(tmp_path, 0)
    else:
        assert load_whole_task_observation(tmp_path, 0)['image']['rgb'].shape == (480, 640, 3)
    with pytest.raises(SensorContractError, match='population'): load_continuation_observation(tmp_path, 0)
