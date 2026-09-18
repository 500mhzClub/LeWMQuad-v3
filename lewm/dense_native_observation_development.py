"""Full native RGB and causal command history for the dense world model."""
import numpy as np
from PIL import Image
import torch

from lewm.simulated_body_observation_development import validate_policy_packet
from scripts.dev_frozen_dense_representation_encoders_v1 import _normalise, _to_chw


def dense_native_context(packets, *, observed_ns):
    """Consume exactly three acquired packets at -1000/-500/0 ms.

    The caller supplies retained observations, not resized legacy model tensors
    or future packets. The returned commands are raw applied native units;
    normalization belongs to the predictor's existing training statistics.
    """
    if len(packets)!=3:
        raise ValueError('three native packets at 500-ms spacing required')
    expected=[observed_ns-1_000_000_000,observed_ns-500_000_000,observed_ns]
    identity=packets[-1]['sensor_state']['identity'];pixels=[]
    for packet,now in zip(packets,expected,strict=True):
        image,state=packet['image'],packet['sensor_state']
        if (image['measured_ns']!=now or state['image_ns']!=now or state['decision_ns']!=now
                or image['available_ns']>observed_ns or state['identity']!=identity):
            raise ValueError('exact causal packet times and common episode required')
        validate_policy_packet(packet)
        rgb=np.asarray(image['rgb'])
        if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:
            raise ValueError('full native 640x480 uint8 RGB required; do not upsample legacy tensors')
        resized=Image.fromarray(rgb).resize((512,384),Image.Resampling.BICUBIC)
        pixels.append(_normalise(_to_chw(resized)))
    prior=packets[-1]['sensor_state']['control']['applied_command']
    values=np.asarray(prior['values'],np.float32)
    measured=np.asarray(prior['measured_ns'],np.int64)
    available=np.asarray(prior['available_ns'],np.int64)
    if (values.shape!=(15,3) or measured.shape!=(15,) or available.shape!=(15,)
            or not np.asarray(prior['valid']).all() or not np.isfinite(values).all()
            or not np.array_equal(measured[[4,9,14]],expected)
            or not np.all(np.diff(measured)==100_000_000)
            or np.any(measured>observed_ns) or np.any(available>observed_ns)
            or np.any(values[:,1]!=0)):
        raise ValueError('complete causal native command history required')
    return dict(pixels=torch.stack(pixels),context_times_ns=torch.tensor(expected,dtype=torch.int64),
        past_applied_commands=torch.from_numpy(values.copy()),
        past_measured_ns=torch.from_numpy(measured.copy()),
        past_available_ns=torch.from_numpy(available.copy()))
