"""Use 128-frame projection batches in the original chronological coverage loop."""
from types import FunctionType
from lewm.visibility_batched_retained_floor_patch_development import VisibilityBatchedRetainedFloorPatches

FRAME_BATCH=128
_original=VisibilityBatchedRetainedFloorPatches.coverage
if _original.__closure__ is not None:
    raise ValueError('closure-free original coverage body required')


class WideBatchedRetainedFloorPatches(VisibilityBatchedRetainedFloorPatches):
    coverage=FunctionType(_original.__code__,_original.__globals__|{'FRAME_BATCH':FRAME_BATCH},
        _original.__name__,_original.__defaults__)
    coverage.__kwdefaults__=_original.__kwdefaults__
