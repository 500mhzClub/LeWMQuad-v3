"""Read-only complete transfer routing and bounded expanded-witness validation.

No optimizer, parameter update, model scoring, cache or native scene is created.
Full predecessor authentication executes before and after this preparation.
"""
import json
import time
import numpy as np
import torch
from lewm.observation_horizon_fit_development import score
from scripts.all_phase_study_inputs_development import authenticate, stream, CORRECTION_SHA, CHECK_SHA
from scripts.all_phase_study_stream_development import verified_plan
from scripts.navigation_artifact_root_development import BASE
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import verify as verify_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

SOURCE = 'scripts/check_go2_all_phase_study_stream_preparation_v1.py'
TEST = 'lewm/tests/test_all_phase_study_stream_development.py'


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    before = hardware()
    if before['memory_available_bytes'] < 40*1024**3:
        raise ValueError('8GiB preparation plus32GiB concurrent native memory headroom required')
    print('STUDY_PREPARATION_AUTHENTICATING', flush=True)
    launch, _, _ = authenticate()
    sources = discover_sources((SOURCE, TEST), launch['source_sha256'])
    verify_sources(sources)
    owner_sources = {}
    for root in ('go2_prepared_native_queue_v1_attempt_001', 'go2_supervised_commitment_contact_native_wait_v1_attempt_001'):
        for path, sha in read_json(BASE/root, 'launch.json')['source_sha256'].items():
            if path in owner_sources and owner_sources[path] != sha:
                raise ValueError('live original owner source identities disagree')
            owner_sources[path] = sha
    verify_sources(owner_sources)
    data = stream(maximum_cache_bytes=0); started = time.perf_counter()
    examples = []
    for source in ('family', 'switch'):
        for offset in (0, 1, 2, 3, 4, 36, 39):
            i = next(i for i in data.view.indices('train', source=source)
                if data.view.rows[i]['offset_ticks'] == offset)
            batch = data.training_batch([i]); verified_plan(data.view, [i], batch['inputs'])
            inputs = data.inference_batch([i], role='train'); verified_plan(data.view, [i], inputs)
            def equal(a, b):
                return (set(a) == set(b) and all(equal(a[k], b[k]) for k in a)) if isinstance(a, dict) else torch.equal(a, b)
            if not equal(batch['inputs'], inputs): raise ValueError('training future changed past-only inputs')
            examples.append(dict(index=i, source=source, offset=offset, sample_id=data.view.rows[i]['sample_id']))
    print('STUDY_PREPARATION_TRAINING_WITNESSES', len(examples), flush=True)
    ids = data.view.indices('geometry_transfer'); transfer_rows = []
    for start in range(0, len(ids), 6):
        selected = ids[start:start+6]
        inputs = data.inference_batch(selected, role='geometry_transfer')
        verified_plan(data.view, selected, inputs)
        transfer_rows.extend(dict(index=i, original_index=data.view.transfer_index(i), sample_id=data.view.rows[i]['sample_id']) for i in selected)
        if len(transfer_rows)%84 == 0: print('STUDY_PREPARATION_TRANSFER', len(transfer_rows), flush=True)
    # Zero-valued synthetic predictions check score/index accounting only.
    # No model predictions or empirical predictive-performance claim result.
    values = np.zeros((len(ids), 8, 5), np.float32); values[:, :, 3] = 1.
    arrays = dict(indices=np.asarray(ids, np.int64), direct_outcomes=values,
        prediction_valid=np.asarray([[t['in_plan'] for t in data.view.rows[i]['targets']] for i in ids], bool),
        target_offsets_ns=np.asarray([[t['offset_ns'] for t in data.view.rows[i]['targets']] for i in ids], np.int64))
    actual = score(data.view, arrays, role='geometry_transfer', head='direct_outcomes')
    original_arrays = arrays | dict(indices=np.asarray([data.view.transfer_index(i) for i in ids], np.int64))
    expected = score(data.original.view, original_arrays, role='geometry_transfer', head='direct_outcomes')
    if actual != expected: raise ValueError('transfer scoring changed under new global indices')
    materialization_wall = time.perf_counter()-started
    print('STUDY_PREPARATION_FINAL_AUTHENTICATION', flush=True)
    authenticate(); verify_sources(sources); verify_sources(owner_sources)
    report = dict(status='ALL_PHASE_STUDY_STREAM_PREPARATION_COMPLETE',
        input_check_sha256=CHECK_SHA, scope_correction_sha256=CORRECTION_SHA,
        source_sha256=sources, original_owner_sources_unchanged=len(owner_sources),
        total_slots=5256, training_slots=4800, transfer_slots=456,
        available_training_contexts=4010, available_transfer_contexts=420,
        training_witness_examples=examples, transfer_witnesses=transfer_rows,
        all_transfer_inputs_match_original_admission=True,
        synthetic_transfer_score_accounting_exact=True,
        full_original_admission_before_and_after=True, training_sample_cache_bytes=data.training.cache_bytes,
        materialization_wall_s=materialization_wall, hardware_before=before, hardware_after=hardware(),
        model_constructed=False, optimizer_updates=0, fitted_checkpoints=0, native_scenes=0,
        training_schedule_fixed=False, matched_retraining_completed=False, navigation_qualified=False)
    print('STUDY_PREPARATION_COMPLETE', json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__': main()
