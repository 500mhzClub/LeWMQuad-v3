"""Original controller ownership, storage wiring and failure-path equivalence."""
import hashlib
from pathlib import Path
from lewm.batched_patch_anchored_controller_development import (
    BatchedPatchAnchoredController, PATCH_FIELDS)
from lewm.batched_retained_floor_patch_development import BatchedRetainedFloorPatches
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.tests.test_batched_retained_floor_patch_development import frame, GOOD, BAD
from scripts.replay_go2_batched_patch_anchored_prefix_v1 import normalize_candidate, normalized_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def controllers():
    options = dict(public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
        require_return_after_goal=True), navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    model, geometry = object(), object()
    return [kind(model, geometry, **options)
        for kind in (ResidualAnchoredContinuationController, BatchedPatchAnchoredController)]


def retained(c):
    return dict(memory=c.memory, floor=c.mapper.floor, occupied=c.mapper.occupied,
        residual=c.residual, history=c.history)


def test_frozen_implementations_and_inherited_policy():
    bindings = {
        'lewm/residual_anchored_continuation_controller_development.py':
            'f35ff18c78b4db81c0c3c766eed1015823bf972d484600907954941ef7fb946a',
        'lewm/batched_retained_floor_patch_development.py':
            'd16244f88818ade6dd9a07f2aeebadb9fadc99f6f2a303c8d9cd5fcf64121e5d',
    }
    for path, expected in bindings.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for name in ('observe', 'advance'):
        assert getattr(BatchedPatchAnchoredController, name) is getattr(ResidualAnchoredContinuationController, name)
    assert BatchedRetainedFloorPatches.append is RetainedFloorPatches.append


def test_only_fresh_patch_stores_change_and_all_existing_aliases_remain():
    old, new = controllers()
    assert type(new.mapper) is type(old.mapper)
    assert type(new.memory) is type(old.memory)
    assert type(new.selector) is type(old.selector)
    assert new.selector.residual is new.residual
    assert new.memory is new.mapper.surface
    assert new.memory.patches.frames is not new.memory.auxiliary_patches.frames
    for name in PATCH_FIELDS:
        assert type(getattr(old.memory, name)) is RetainedFloorPatches
        assert type(getattr(new.memory, name)) is BatchedRetainedFloorPatches
    assert fingerprint(normalized_state_tree(retained(old))) == fingerprint(normalized_state_tree(retained(new)))


def test_both_camera_stores_are_used_without_changing_coverage_or_witnesses():
    old, new = controllers()
    for name in PATCH_FIELDS:
        a, b = getattr(old.memory, name), getattr(new.memory, name)
        for i in range(35):
            a.frames.append(frame(i, GOOD if i == 34 else BAD))
            b.frames.append(frame(i, GOOD if i == 34 else BAD))
        assert a.coverage([[1., 0.], [0., 0.]]) == b.coverage([[1., 0.], [0., 0.]])
        assert b.coverage([[1., 0.]])[0]['coverage_witness']['witness']['frame'] == 34
    assert fingerprint(normalized_state_tree(retained(old))) == fingerprint(normalized_state_tree(retained(new)))


def test_bad_sensor_packet_preserves_full_original_stop_decision():
    old, new = controllers()
    expected = old.observe({}, {}, {}, now_ns=1)
    actual = new.observe({}, {}, {}, now_ns=1)
    assert expected['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert expected['requested_command'] == [0., 0., 0.]
    assert normalize_candidate(actual) == expected
