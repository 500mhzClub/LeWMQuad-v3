"""Typed digest of the captured public depth packet, including its arrays."""
import hashlib
import json
import numpy as np


SCHEMA = 'in_memory_paired_camera_development.v2_raw_depth'


def packet_digest(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return dict(array_dtype=item.dtype.str, array_shape=list(item.shape),
                array_sha256=hashlib.sha256(item.tobytes()).hexdigest())
        if isinstance(item, np.generic): return encode(item.item())
        if isinstance(item, dict): return {k:encode(v) for k,v in item.items()}
        if isinstance(item, (tuple, list)): return [encode(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
