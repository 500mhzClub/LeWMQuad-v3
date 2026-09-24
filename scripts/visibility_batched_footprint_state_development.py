"""Normalize only the existing two patch type paths for the new implementation."""
from types import FunctionType
from lewm.visibility_batched_footprint_controller_development import PATCH_FIELDS
from lewm.visibility_batched_retained_floor_patch_development import VisibilityBatchedRetainedFloorPatches
from scripts import packed_fused_state_development as packed

STATE_TYPE_PATHS=packed.STATE_TYPE_PATHS


def fork(function,**bindings):
    result=FunctionType(function.__code__,function.__globals__|bindings,
        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


_patch_state=fork(packed.patch_state_tree,BatchedRetainedFloorPatches=VisibilityBatchedRetainedFloorPatches)
_candidate_state=fork(packed.normalized_state_tree,patch_state_tree=_patch_state)


def normalized_state_tree(value):
    memory=value['memory'];patches=[getattr(memory,name) for name in PATCH_FIELDS]
    if any(type(p) is VisibilityBatchedRetainedFloorPatches for p in patches):
        return _candidate_state(value)
    return packed.normalized_state_tree(value)
