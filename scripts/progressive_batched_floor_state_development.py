"""Normalize the existing patch type paths without changing retained state."""
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
from scripts import visibility_batched_footprint_state_development as previous

STATE_TYPE_PATHS=previous.STATE_TYPE_PATHS
_patch_state=previous.fork(previous.packed.patch_state_tree,
    BatchedRetainedFloorPatches=ProgressiveBatchedRetainedFloorPatches)
_candidate_state=previous.fork(previous.packed.normalized_state_tree,patch_state_tree=_patch_state)


def normalized_state_tree(value):
    patches=[getattr(value['memory'],name) for name in previous.PATCH_FIELDS]
    if any(type(p) is ProgressiveBatchedRetainedFloorPatches for p in patches):
        return _candidate_state(value)
    return previous.normalized_state_tree(value)
