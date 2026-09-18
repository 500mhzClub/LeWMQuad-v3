"""Replay captured auxiliary RGB and depth through distinct public contracts."""
import hashlib
import numpy as np
from PIL import Image
from lewm.causal_rgb_dataset_development import _leaf
from lewm.novel_maze_round_trip_contract_development import MAX_OBSERVATIONS
from lewm.auxiliary_downward45_depth_geometry_development import CALIBRATION_ID
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from scripts.novel_maze_auxiliary_packet_development import packet as depth_packet

PUBLIC_FIELDS = {'frame', 'measured_ns', 'native_depth_sha256', 'calibration_id', 'rgb_sha256'}


def public_acquisition(row):
    return {k: row[k] for k in PUBLIC_FIELDS}


def packet(directory, index, policy, acquisition, *, now_ns):
    if type(index) is not int or not 0 <= index < MAX_OBSERVATIONS:
        raise ValueError('bounded actual auxiliary RGB frame required')
    if (not isinstance(acquisition, dict) or set(acquisition) != PUBLIC_FIELDS
            or type(acquisition['frame']) is not int or acquisition['frame'] != index
            or acquisition['calibration_id'] != CALIBRATION_ID):
        raise ValueError('exact public auxiliary RGB acquisition identity required')
    depth = depth_packet(directory, index, policy,
        {k: acquisition[k] for k in PUBLIC_FIELDS-{'rgb_sha256'}}, now_ns=now_ns)
    with Image.open(_leaf(directory, f'auxiliary_rgb_{index:04d}.png')) as image:
        rgb = np.array(image)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != acquisition['rgb_sha256']:
        raise ValueError('auxiliary RGB pixels differ from actual acquisition identity')
    image = from_captured_rgb(rgb, depth, policy, measured_ns=acquisition['measured_ns'],
        available_ns=acquisition['measured_ns'], now_ns=now_ns)
    return image, depth
