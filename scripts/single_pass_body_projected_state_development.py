"""Compare full retained evidence with the existing ten declared type paths."""
from lewm.packed_fused_scoped_controller_development import index_owners
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from scripts import progressive_batched_floor_state_development as previous

STATE_TYPE_PATHS = previous.STATE_TYPE_PATHS
_candidate_state = previous.previous.fork(
    previous.previous.packed.normalized_state_tree,
    PackedOwnedMeasuredSampleBoundsIndex=SinglePassMeasuredSampleBoundsIndex,
    patch_state_tree=previous._patch_state)


def normalized_state_tree(value):
    indices = [getattr(owner, name) for owner, name in index_owners(value['memory'])]
    if any(type(index) is SinglePassMeasuredSampleBoundsIndex for index in indices):
        return _candidate_state(value)
    return previous.normalized_state_tree(value)
