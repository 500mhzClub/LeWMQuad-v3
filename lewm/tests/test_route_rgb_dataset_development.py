import json
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_rgb_dataset_development import load_policy_observation
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.tests.test_causal_rgb_dataset_development import create


def test_route_schema_is_explicit_and_old_reader_remains_strict(tmp_path):
    manifest,_=create(tmp_path)
    with pytest.raises(SensorContractError): load_route_observation(tmp_path,0)
    manifest['schema']='causal_rgb_body_routes_development.v1'
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    assert load_route_observation(tmp_path,0)['image']['rgb'].shape==(480,640,3)
    with pytest.raises(SensorContractError): load_policy_observation(tmp_path,0)


@pytest.mark.parametrize('fault',['route_truth','future_tensor_path','rgb_path','excess_frames'])
def test_route_reader_does_not_relax_policy_boundary(tmp_path,fault):
    manifest,_=create(tmp_path)
    manifest['schema']='causal_rgb_body_routes_development.v1'
    if fault=='route_truth': manifest['true_route']=[[0,0],[1,0]]
    elif fault=='future_tensor_path': manifest['history_file']='physics_trace.npz'
    elif fault=='rgb_path': manifest['frames'][0]['rgb_file']='../rgb_0000.png'
    else: manifest['frames']*=342
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    with pytest.raises(SensorContractError): load_route_observation(tmp_path,0)
