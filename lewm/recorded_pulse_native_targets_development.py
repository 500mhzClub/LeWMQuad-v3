"""Decode the collector's exact binary uint8 contact serialization."""
import numpy as np
from lewm.pulse_native_targets_development import PulseNativeTargets


class RecordedPulseNativeTargets(PulseNativeTargets):
    def __init__(self,raw):
        contact=np.asarray(raw['physics_contact'])
        if contact.dtype!=np.uint8 or contact.ndim!=1 or not np.isin(contact,[0,1]).all():
            raise ValueError('recorded contact must be exact binary uint8')
        super().__init__(raw|{'physics_contact':contact.astype(bool)})
