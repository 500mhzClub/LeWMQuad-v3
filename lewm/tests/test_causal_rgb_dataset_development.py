import json

import numpy as np
from PIL import Image
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_rgb_dataset_development import load_policy_observation,schema_metadata
from lewm.simulated_body_observation_development import SCHEMAS,CAMERA_CALIBRATION
from lewm.tests.test_simulated_body_observation_development import packet


def create(directory):
    value=packet()
    state=value['sensor_state']
    arrays={'image_ns':np.array([500_000_000]),'decision_ns':np.array([500_000_000])}
    for schema in SCHEMAS:
        for field in ('values','valid','measured_ns','available_ns'):
            arrays[f'{schema.name}_{field}']=state[schema.role][schema.name][field][None]
    np.savez_compressed(directory/'policy_histories.npz',**arrays)
    manifest={'schema':'causal_rgb_body_development.v1','camera_calibration_id':CAMERA_CALIBRATION,
        'sensor_assumption':'ideal_simulated_body_origin_50hz_zero_latency','history_file':'policy_histories.npz',
        'sensor_schemas':schema_metadata(),'frames':[{'rgb_file':'rgb_0000.png','image_ns':500_000_000,'decision_ns':500_000_000}]}
    (directory/'policy_observations.json').write_text(json.dumps(manifest))
    Image.fromarray(value['image']['rgb']).save(directory/'rgb_0000.png')
    return manifest,arrays


def test_policy_reader_does_not_require_any_oracle_artifact(tmp_path):
    create(tmp_path)
    value=load_policy_observation(tmp_path,0)
    assert set(value)=={'image','sensor_state'}
    assert value['sensor_state']['sensed']['joints']['values'].shape==(20,24)


@pytest.mark.parametrize('fault',['oracle_metadata','future_tensor','path_escape','history_path','future_time','wrong_schema'])
def test_loader_rejects_privileged_or_corrupt_artifacts(tmp_path,fault):
    manifest,arrays=create(tmp_path)
    if fault=='oracle_metadata': manifest['global_pose']=[0]*7
    elif fault=='future_tensor': arrays['future_contact']=np.zeros(1)
    elif fault=='path_escape': manifest['frames'][0]['rgb_file']='../rgb_0000.png'
    elif fault=='history_path': manifest['history_file']='physics_trace.npz'
    elif fault=='future_time': arrays['gyro_available_ns'][0,-1]+=1
    else: manifest['sensor_schemas'][0]['channels'][0]='global_x'
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    np.savez_compressed(tmp_path/'policy_histories.npz',**arrays)
    with pytest.raises(SensorContractError): load_policy_observation(tmp_path,0)


def test_missing_frame_and_boolean_index_are_rejected(tmp_path):
    create(tmp_path)
    for index in (1,True,-1):
        with pytest.raises(SensorContractError): load_policy_observation(tmp_path,index)


def test_symlink_cannot_escape_trial_directory(tmp_path):
    directory=tmp_path/'trial'
    directory.mkdir()
    create(directory)
    external=tmp_path/'external.png'
    (directory/'rgb_0000.png').replace(external)
    (directory/'rgb_0000.png').symlink_to(external)
    with pytest.raises(SensorContractError,match='escapes'): load_policy_observation(directory,0)
