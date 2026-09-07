"""RGB-only visible-floor evidence for the current simulator palette.

This intentionally simple appearance baseline is NOT learned semantics, metric
clearance, traversability, exit identity or place association. Neutral walls and
the renderer's default `floor` palette are distinguishable by chroma; shadows or
appearance shifts can remove that cue. Negative output means unknown, not wall.
No image-to-ground transform, body/world pose or layout enters this runtime code.
"""
import numpy as np

from lewm.causal_sensor_state import SensorContractError,_ns
from lewm.simulated_body_observation_development import validate_policy_packet


def palette_floor_mask(rgb):
    image=np.asarray(rgb)
    if image.shape!=(480,640,3) or image.dtype!=np.uint8: raise SensorContractError('native calibrated RGB required')
    r,g,b=np.moveaxis(image.astype(np.int16),-1,0)
    # Fixed low chroma margins from the declared greenish simulator floor
    # palette, not thresholds fitted on validation images or inferred depth.
    return (g-r>=2)&(g-b>=4)&(r-b>=1)


def bottom_connected_envelope(mask):
    mask=np.asarray(mask)
    if mask.shape!=(480,640) or mask.dtype!=bool: raise ValueError('native boolean evidence mask required')
    count=np.logical_and.accumulate(mask[::-1,:],axis=0).sum(axis=0)
    # Twelve contiguous pixels suppress tiny lower-image specks. This is image
    # evidence extent only; no pixel count is converted to metres or body width.
    valid=count>=12
    return {'valid_columns':valid,'first_floor_row':np.where(valid,480-count,-1).astype(np.int32),
        'contiguous_floor_rows':count.astype(np.int32)}


def observe_floor(packet,*,now_ns):
    validate_policy_packet(packet); now_ns=_ns(now_ns,'floor observation clock')
    if packet['image']['measured_ns']!=now_ns or packet['sensor_state']['decision_ns']!=now_ns:
        raise SensorContractError('current RGB evidence required')
    mask=palette_floor_mask(packet['image']['rgb'])
    return {'decision_ns':now_ns,'floor_evidence_mask':mask,**bottom_connected_envelope(mask),
        'metric_clearance_qualified':False,'place_or_exit_identity':None,
        'scope':'palette-specific visible-floor pixels only; negative means unknown, not blocked or safe'}
