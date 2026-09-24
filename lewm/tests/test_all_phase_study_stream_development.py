"""Complete synthetic populations exercise transfer isolation and study mapping."""
from copy import deepcopy
from functools import lru_cache
import pytest
import torch
from lewm.observation_horizon_view_development import ObservationHorizonView
from lewm.all_phase_training_view_development import AllPhaseTrainingView
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.tests.test_observation_horizon_stream_development import fixture as original_fixture
from scripts import all_phase_study_stream_development as mod
from scripts.all_phase_training_policy_stream_development import AllPhaseTrainingStream
from scripts.observation_horizon_fit_inputs_development import CheckedObservationHorizonStream


@lru_cache(maxsize=1)
def population():
    original, old_rows = original_fixture(); old = ObservationHorizonView(original, old_rows)
    trials = {}
    for row in old.rows:
        if row['data_role'] == 'train':
            offset = row['offset_ticks'] if row['source'] == 'family' else 0
            trials.setdefault((row['source'], row['trial']), {})[offset] = row
    expanded = []
    for (source, trial), rows in sorted(trials.items()):
        first = rows[0]
        identity = {k: first[k] for k in ('source', 'trial', 'action', 'data_role',
            'geometry', 'cluster', 'opening', 'appearance_seed') if k in first}
        for offset in range(40):
            frame = (3 if source == 'family' else 13)+offset
            now = 1_500_000_000+100_000_000*frame; past = list(range(frame-3, frame+1))
            available = rows[offset]['available'] if offset in rows else True
            reason = None if available else rows[offset]['reason']
            targets = [] if available else None
            if available:
                for h in range(1, 9):
                    active = h <= min(8, 40-offset)
                    targets.append(dict(in_plan=active, offset_ns=h*100_000_000 if active else 0,
                        motion_valid=active, motion=[0., 0., 0.] if active else None,
                        contact_valid=active, contact=0. if active else None,
                        future_image_valid=active, future_observation_index=frame+h if active else None))
            if offset in rows: targets = deepcopy(rows[offset]['targets'])
            expanded.append(deepcopy(identity) | dict(offset_ticks=offset, remaining_ticks=40-offset,
                sample_id=f'all_phase_train/{source}/{trial}/offset_{offset:02d}',
                original_sample_id=rows[offset]['sample_id'] if offset in rows else None,
                decision_ns=now, history_observation_indices=past, control_phase_modulo_five=frame%5,
                known_commands=candidate_commands(first['action'])[offset:offset+8],
                available=available, reason=reason, targets=targets,
                observation_horizon_receipt=dict(target_only=True, departure_tick=frame, departure_ns=now,
                    history_observation_indices=past, target_cadence_ns=100_000_000,
                    maximum_horizon_ns=800_000_000, available=available, reason=reason),
                native_labels_are_target_only=True, source_role_changed=False, future_rgb_materialized=False))
    return old, expanded


def test_every_transfer_slot_preserved_and_maps_to_original_global_index():
    old, rows = population(); view = AllPhaseTrainingView(old, rows)
    assert len(view.rows) == 5256
    for combined in range(4800, 5256):
        original = view.transfer_index(combined)
        assert view.rows[combined] == old.rows[original]
        assert old.rows[original]['data_role'] == 'geometry_transfer'
    for source in ('family', 'switch'):
        assert [view.transfer_index(i) for i in view.indices('geometry_transfer', source=source)] == old.indices('geometry_transfer', source=source)
    assert all(i < 4800 for i in view.indices('train'))
    assert all(i >= 4800 for i in view.indices('geometry_transfer'))
    switch = next(r for r in view.rows if r['source'] == 'switch' and r['offset_ticks'] == 39)
    witness = next(r for r in old.rows if r['source'] == 'switch' and r['trial'] == switch['trial'])
    assert switch['stratum'] == witness['stratum'] and switch['prefix_action'] == witness['prefix_action']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'role', 'trial', 'target', 'availability',
    'clock', 'phase', 'witness', 'action', 'stratum', 'offset', 'receipt'])
def test_changed_population_or_shared_witness_rejected(fault):
    old, rows = population(); rows = deepcopy(rows); row = rows[0]
    if fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows[1] = deepcopy(row)
    elif fault == 'role': row['data_role'] = 'geometry_transfer'
    elif fault == 'trial': row['trial'] = 'unassigned'
    elif fault == 'target': row['targets'][0]['motion'] = [1., 0., 0.]
    elif fault == 'availability': row['available'] = not row['available']
    elif fault == 'clock': row['decision_ns'] += 1
    elif fault == 'phase': row['control_phase_modulo_five'] = -1
    elif fault == 'witness': row['original_sample_id'] = None
    elif fault == 'action': row['action'] = 'unassigned'
    elif fault == 'stratum': row['stratum'] = 'invented'
    elif fault == 'offset': row['offset_ticks'] = True
    elif fault == 'receipt': row['observation_horizon_receipt']['target_only'] = False
    with pytest.raises(ValueError): AllPhaseTrainingView(old, rows)


def routed_stream():
    old, rows = population(); calls = []
    training = object.__new__(mod.CheckedAllPhaseTrainingStream)
    training.rows = rows; training.failed = False
    original = object.__new__(CheckedObservationHorizonStream)
    original.view = old; original.failed = False
    def record(name, indices, role=None):
        calls.append((name, indices, role)); return name
    training.training_batch = lambda ids: record('training', ids)
    training.inference_batch = lambda ids, role: record('train_inference', ids, role)
    original.inference_batch = lambda ids, role: record('transfer_inference', ids, role)
    original.training_batch = lambda *a, **k: pytest.fail('original future-training reader called')
    return mod.AllPhaseStudyStream(training, original), calls


def test_transfer_inference_routes_every_available_original_index_without_training_calls():
    stream, calls = routed_stream(); ids = stream.view.indices('geometry_transfer')
    for start in range(0, len(ids), 16):
        batch = ids[start:start+16]
        assert stream.inference_batch(batch, role='geometry_transfer') == 'transfer_inference'
        assert calls[-1] == ('transfer_inference', [stream.view.transfer_index(i) for i in batch], 'geometry_transfer')
    train = stream.view.indices('train')[:2]
    assert stream.training_batch(train) == 'training'
    assert stream.inference_batch(train, role='train') == 'train_inference'


@pytest.mark.parametrize('fault', ['transfer_training', 'mixed', 'wrong_role', 'negative', 'boolean', 'empty', 'large'])
def test_invalid_batch_rejected_before_either_reader_and_failure_latched(fault):
    stream, calls = routed_stream(); train = stream.view.indices('train')[0]; transfer = stream.view.indices('geometry_transfer')[0]
    ids = dict(transfer_training=[transfer], mixed=[train, transfer], wrong_role=[train],
        negative=[-1], boolean=[True], empty=[], large=[train]*17)[fault]
    with pytest.raises(ValueError):
        if fault == 'wrong_role': stream.inference_batch(ids, role='geometry_transfer')
        else: stream.training_batch(ids)
    assert stream.failed and calls == []
    with pytest.raises(ValueError, match='latched'): stream.inference_batch([transfer], role='geometry_transfer')


def test_exact_training_suffix_and_original_transfer_plan_in_one_view():
    old, rows = population(); view = AllPhaseTrainingView(old, rows)
    ids = [next(i for i in view.indices('train', source=s) if view.rows[i]['offset_ticks'] == offset)
        for s in ('family', 'switch') for offset in (1, 36, 39)]
    ids += [view.indices('geometry_transfer', source=s)[0] for s in ('family', 'switch')]
    pairs = [mod.plan(view.rows[i]) if i < 4800 else mod.original_plan(view.rows[i]['action'],
        offset_ticks=view.rows[i]['offset_ticks'] if view.rows[i]['source'] == 'family' else 0) for i in ids]
    inputs = dict(known_action_blocks=torch.stack([p[0] for p in pairs]), known_action_valid=torch.stack([p[1] for p in pairs]))
    active, offsets = mod.verified_plan(view, ids, inputs)
    assert active.sum(1)[:6].tolist() == [8, 4, 1, 8, 4, 1]
    assert offsets[2].tolist() == [100_000_000]+[0]*7
    inputs['known_action_blocks'][0, 0, 0, 0] += .01
    with pytest.raises(ValueError, match='exact assigned'): mod.verified_plan(view, ids, inputs)


@pytest.mark.parametrize('fault', ['inputs', 'scope', 'future', 'targets'])
def test_materialization_must_reproduce_completed_witness_and_latch_failure(monkeypatch, fault):
    obj = object.__new__(mod.CheckedAllPhaseTrainingStream); obj.failed = False
    sample = dict(inputs={'placeholder': 1}, targets=dict(future_observations={'rgb': torch.tensor([1.])}, motion=torch.tensor([2.])))
    witness = dict(materialized=True, inference_and_training_inputs_exact=True, identity='bound',
        training_access={'scope': 'bound'}, past_access={'scope': 'bound'},
        future_tensor_sha256={'rgb': mod.fingerprint(sample['targets']['future_observations']['rgb'].numpy())},
        target_tensor_sha256={'motion': mod.fingerprint(sample['targets']['motion'].numpy())})
    obj.index = [witness]; obj.last_access = {'scope': 'bound'}
    monkeypatch.setattr(AllPhaseTrainingStream, 'materialize', lambda *a, **k: sample)
    monkeypatch.setattr(mod, 'input_identity', lambda _: {'identity': 'bound'})
    if fault == 'inputs': witness['identity'] = 'changed'
    elif fault == 'scope': obj.last_access['scope'] = 'changed'
    elif fault == 'future': witness['future_tensor_sha256']['rgb'] = 'changed'
    else: witness['target_tensor_sha256']['motion'] = 'changed'
    with pytest.raises(ValueError): obj.materialize(0, training=True)
    assert obj.failed
