"""Compare all retained state with only eight additional exact type tags."""
from lewm.packed_fused_scoped_controller_development import INDEX_PATHS, INDEX_FIELDS, index_owners
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex
from scripts.replay_go2_batched_patch_anchored_prefix_v1 import normalized_state_tree as patch_state_tree

STATE_TYPE_PATHS = ['memory.fields.patches.type', 'memory.fields.auxiliary_patches.type'] + [
    'memory.fields.'+'.fields.'.join(path)+'.type' for path in INDEX_PATHS]


def normalized_state_tree(value):
    result = patch_state_tree(value)
    indices = [getattr(owner, name) for owner, name in index_owners(value['memory'])]
    kind = type(indices[0])
    if (kind not in (MeasuredSampleBoundsIndex, PackedOwnedMeasuredSampleBoundsIndex)
            or len({id(index) for index in indices}) != 8
            or any(type(index) is not kind or set(vars(index)) != INDEX_FIELDS for index in indices)):
        raise ValueError('eight distinct matched original or packed-owned bound indices required')
    expected = kind.__module__+'.'+kind.__name__
    original = MeasuredSampleBoundsIndex.__module__+'.'+MeasuredSampleBoundsIndex.__name__
    for path in INDEX_PATHS:
        node = result['memory']
        for name in path:
            node = node['fields'][name]
        if set(node) != {'type', 'fields'} or node['type'] != expected:
            raise ValueError('exact bound-index type tag required')
        node['type'] = original
    return result
