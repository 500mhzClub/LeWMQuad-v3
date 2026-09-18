"""Acquire the current primary observation before its paired auxiliary view."""
from scripts.auxiliary_tilted_depth_capture_integrity_development import capture


def capture_frame(session,directory,tick):
    index=session.capture_current()
    if (type(tick) is not int or not 0<=tick<=19 or index!=tick
            or len(session.samples)!=750+50*tick or len(session.model_manifest)!=tick+1):
        raise ValueError('complete current primary observation at exact physical sample required')
    return capture(session,directory,tick)
