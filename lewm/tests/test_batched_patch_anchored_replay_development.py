"""Reject evidence changes while normalizing exact implementation identities."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from lewm.batched_patch_anchored_controller_development import CONTROLLER, FLAG, PATCH_FIELDS
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.batched_retained_floor_patch_development import BatchedRetainedFloorPatches
from lewm.tests.test_batched_patch_anchored_controller_development import controllers, retained
from lewm.tests.test_batched_retained_floor_patch_development import frame
from scripts import replay_go2_batched_patch_anchored_prefix_v1 as candidate
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def test_original_replay_code_and_all_undeclared_global_bindings_are_retained():
    assert hashlib.sha256(Path('scripts/replay_go2_receipt_copied_anchored_prefix_v1.py').read_bytes()).hexdigest() == (
        '0fdc4ac3493ce8c4744cab4a7270c3ff8b41163fe9575988a2111fbbe5bb1a7d')
    original = candidate.preceding.replay
    before = original.__globals__.copy()
    new = candidate.isolated_replay()
    overrides = dict(ReceiptCopiedAnchoredController=candidate.BatchedPatchAnchoredController,
        normalize_candidate=candidate.normalize_candidate, state_tree=candidate.normalized_state_tree,
        OUTPUT=candidate.OUTPUT, print=candidate.progress)
    assert new.__code__ is original.__code__
    assert new.__defaults__ is original.__defaults__
    assert new.__kwdefaults__ is original.__kwdefaults__
    assert new.__closure__ is original.__closure__ is None
    assert new.__globals__ is not original.__globals__
    assert new.__globals__.keys() == before.keys() | overrides.keys()
    for name, value in new.__globals__.items():
        assert value is overrides[name] if name in overrides else value is before[name]
    assert original.__globals__.keys() == before.keys()
    assert all(original.__globals__[name] is value for name, value in before.items())


def test_decision_normalization_preserves_all_nonmetadata_fields():
    decision = dict(controller=CONTROLLER, **{FLAG: True}, requested_command=[0., 0., .45],
        new_selection={'witness': {'type': CONTROLLER, FLAG: True}, 'score': -.2})
    before = deepcopy(decision)
    expected = decision.copy(); expected.pop(FLAG)
    expected['controller'] = 'residual_anchored_continuation_controller_v1'
    assert candidate.normalize_candidate(decision) == expected
    assert decision == before
    for update in ({'controller': 'other'}, {FLAG: False}, {FLAG: 1}):
        with pytest.raises(ValueError): candidate.normalize_candidate(decision | update)


def test_only_two_structural_type_tags_change_and_witness_strings_are_untouched():
    old, new = controllers()
    tag = BatchedRetainedFloorPatches.__module__+'.'+BatchedRetainedFloorPatches.__name__
    for name in PATCH_FIELDS:
        f = frame(0); f['witness']['type'] = tag
        getattr(old.memory, name).frames.append(deepcopy(f))
        getattr(new.memory, name).frames.append(deepcopy(f))
    before = fingerprint(candidate.state_tree(retained(new)))
    normalized = candidate.normalized_state_tree(retained(new))
    assert fingerprint(candidate.normalized_state_tree(retained(old))) == fingerprint(normalized)
    for name in PATCH_FIELDS:
        node = normalized['memory']['fields'][name]
        assert node['type'] == RetainedFloorPatches.__module__+'.'+RetainedFloorPatches.__name__
        assert node['fields']['frames'][0]['witness']['type'] == tag
    assert fingerprint(candidate.state_tree(retained(new))) == before


@pytest.mark.parametrize('change', ['primary_witness', 'auxiliary_pose', 'floor', 'occupied', 'residual', 'history'])
def test_real_retained_state_changes_remain_detectable(change):
    old, new = controllers()
    for c in (old, new):
        for name in PATCH_FIELDS: getattr(c.memory, name).frames.append(frame(0))
    if change == 'primary_witness': new.memory.patches.frames[0]['witness']['frame'] = 1
    elif change == 'auxiliary_pose': new.memory.auxiliary_patches.frames[0]['p'][0] += .001
    elif change == 'floor': new.mapper.floor[(1, 2)] = 3
    elif change == 'occupied': new.mapper.occupied[(1, 2)] = 3
    elif change == 'residual': new.residual.pending = {'changed': True}
    else: new.history.append({'changed': True})
    assert fingerprint(candidate.normalized_state_tree(retained(old))) != fingerprint(candidate.normalized_state_tree(retained(new)))


@pytest.mark.parametrize('fault', ['missing_population', 'extra_population', 'memory_class', 'mixed_stores', 'extra_patch_field'])
def test_unreviewed_state_implementations_and_partial_populations_are_rejected(fault):
    _, new = controllers(); value = retained(new)
    if fault == 'missing_population': value.pop('history')
    elif fault == 'extra_population': value['extra'] = []
    elif fault == 'memory_class': value['memory'] = object()
    elif fault == 'mixed_stores': new.memory.auxiliary_patches = RetainedFloorPatches()
    else: new.memory.patches.extra = []
    with pytest.raises(ValueError): candidate.normalized_state_tree(value)


def test_private_progress_changes_only_the_old_progress_label(capsys):
    candidate.progress('RECEIPT_COPIED_ANCHORED_RAW_FRAME', 50, flush=True)
    assert capsys.readouterr().out == 'BATCHED_PATCH_ANCHORED_RAW_FRAME 50\n'
