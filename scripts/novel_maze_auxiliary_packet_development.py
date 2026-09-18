"""Read only captured depth arrays into the strict auxiliary public packet."""
from lewm.novel_maze_round_trip_contract_development import MAX_OBSERVATIONS
import hashlib
import numpy as np
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth
from lewm.auxiliary_downward45_depth_geometry_development import CALIBRATION_ID as CAPTURE_CALIBRATION
from lewm.causal_rgb_dataset_development import _leaf


def public_acquisition(row):
    return {k:row[k] for k in ('frame','measured_ns','native_depth_sha256','calibration_id')}


def packet(directory,index,policy,acquisition,*,now_ns):
    if type(index) is not int or not 0<=index<MAX_OBSERVATIONS:raise ValueError('bounded actual auxiliary frame required')
    if (not isinstance(acquisition,dict) or set(acquisition)!={'frame','measured_ns','native_depth_sha256','calibration_id'}
            or acquisition['frame']!=index or acquisition['calibration_id']!=CAPTURE_CALIBRATION):
        raise ValueError('exact public auxiliary acquisition identity required')
    with np.load(_leaf(directory,f'auxiliary_depth_{index:04d}.npz'),allow_pickle=False) as archive:
        native=archive['native_optical_depth_m'];depth=archive['depth_m'];valid=archive['valid']
    if hashlib.sha256(native.tobytes()).hexdigest()!=acquisition['native_depth_sha256']:
        raise ValueError('auxiliary raw array differs from actual acquisition identity')
    result=from_native_depth(native,policy,measured_ns=acquisition['measured_ns'],available_ns=acquisition['measured_ns'],now_ns=now_ns)
    if not np.array_equal(result['depth_m'],depth) or not np.array_equal(result['valid'],valid):
        raise ValueError('recorded auxiliary mask must reconstruct exactly from native depth')
    return result


