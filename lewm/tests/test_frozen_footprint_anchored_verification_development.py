"""Recorded evidence rejection checks on explicit synthetic prefix fixtures."""
from copy import deepcopy
import json
import sys
import pytest
from scripts import verify_go2_frozen_footprint_anchored_prefix_v1 as verifier


def population():
    replay = verifier.replay; rows = []; saved = []; profiled = []; copied = []; tape = []
    for frame in range(405):
        decision = dict(controller='residual_anchored_continuation_controller_v1', terminal=None,
            requested_command=[0., 0., .45], new_selection=None if frame < 3 else {
                'prediction': [[float(frame)]], 'complete_receipt': {'witness': [frame]}})
        saved.append(dict(tick=frame, decision=decision))
        sha = replay.profile.reference.saved.identity(decision)
        candidate = decision | {'controller': replay.CONTROLLER, replay.FLAG: True}
        rows.append(dict(frame=frame, execution_order=list(replay.preceding.execution_order(frame)),
            original_controller_s=.2+frame*.001, candidate_controller_s=.15+frame*.001,
            original_decision_sha256=sha, candidate_decision_sha256=replay.profile.reference.saved.identity(candidate),
            public_input_sha256='c'*64, complete_original_decision_reconstructed=True,
            candidate_normalized_decision_exact=True, public_input_arrays_unchanged=True))
        profiled.append(dict(frame=frame, original_decision_sha256=sha, public_input_sha256='c'*64))
        copied.append(deepcopy(profiled[-1]))
        tape.append(dict(tick=frame, completed=True, pre_sample_index=749+50*frame,
            post_sample_index=799+50*frame, requested_command=[0., 0., .45]))
    return rows, saved, profiled, copied, tape, replay.preceding.timing_summary(rows)


def test_complete_recorded_population_reconstructs_hashes_and_timings():
    result = verifier.check_rows(*population())
    assert result['original_saved_decisions_reconstructed'] == 405
    assert result['candidate_expected_decision_hashes_checked'] == 405
    assert result['forecast_count'] == 402 and result['original_command_endpoints_checked'] == 405
    assert result['timing_windows']['post_warmup_prefix']['observations'] == 402


@pytest.mark.parametrize('fault', ['partial', 'reordered', 'changed_saved_receipt', 'changed_candidate',
    'changed_profile', 'changed_reference', 'changed_public_hash', 'changed_command', 'unfinished_command',
    'wrong_start', 'wrong_end', 'wrong_timing', 'nan_timing', 'wrong_order', 'claimed_exact', 'missing_forecast'])
def test_corrupted_prefix_evidence_or_timing_rejects(fault):
    rows, saved, profiled, copied, tape, timings = population()
    if fault == 'partial': rows.pop()
    elif fault == 'reordered': rows[3], rows[4] = rows[4], rows[3]
    elif fault == 'changed_saved_receipt': saved[3]['decision']['new_selection']['complete_receipt']['witness'][0] = 99
    elif fault == 'changed_candidate': rows[3]['candidate_decision_sha256'] = 'd'*64
    elif fault == 'changed_profile': profiled[3]['original_decision_sha256'] = 'd'*64
    elif fault == 'changed_reference': copied[3]['original_decision_sha256'] = 'd'*64
    elif fault == 'changed_public_hash': rows[3]['public_input_sha256'] = 'd'*64
    elif fault == 'changed_command': tape[3]['requested_command'] = [.2, 0., 0.]
    elif fault == 'unfinished_command': tape[3]['completed'] = False
    elif fault == 'wrong_start': tape[3]['pre_sample_index'] += 1
    elif fault == 'wrong_end': tape[3]['post_sample_index'] += 1
    elif fault == 'wrong_timing': timings['post_warmup_prefix']['candidate_total_s'] += .01
    elif fault == 'nan_timing': rows[3]['candidate_controller_s'] = float('nan')
    elif fault == 'wrong_order': rows[3]['execution_order'] = [0, 1]
    elif fault == 'claimed_exact': rows[3]['candidate_normalized_decision_exact'] = 1
    else:
        saved[3]['decision']['new_selection'] = None
    with pytest.raises(ValueError): verifier.check_rows(rows, saved, profiled, copied, tape, timings)


def reports():
    replay = verifier.replay
    states = [dict(frame=f, state_sha256='e'*64, complete_retained_observed_state_equal=True) for f in replay.STATE_FRAMES]
    report = dict(frames=405, raw_model_forecast_comparisons=402,
        model_state_sha256=replay.profile.reference.MODEL_SHA, model_state_unchanged=True,
        complete_original_decisions_reconstructed=True, complete_normalized_candidate_decisions_exact=True,
        public_input_arrays_unchanged=True, alternating_execution_order=True, profiling_enabled=False,
        controller_observe_only_timed=True, sensor_acquisition_timed=False, isolated_benchmark=False,
        no_observation_405_consumed=True, native_execution=False, real_time_qualified=False,
        navigation_qualified=False, observed_state_checks=states)
    result = dict(status='FROZEN_FOOTPRINT_ANCHORED_PREFIX_V1_COMPLETE', native_execution=False,
        goal_achieved=False, preceding_result_sha256=replay.PRECEDING_SHA, source_sha256={}, wall_s=1., report=report)
    launch = dict(preceding_result_sha256=replay.PRECEDING_SHA, profile_result_sha256=replay.preceding.PROFILE_SHA,
        frames=405, state_frames=[3, 12, 395, 404], model_state_sha256=replay.profile.reference.MODEL_SHA,
        native_execution=False, model_training=False, profiling_enabled=False, imported_module_globals_mutated=False,
        receipt_copy_optimization_enabled=False, batched_patch_queries_enabled=False,
        invocation_local_surface_receipt_sharing=False, invocation_frozen_footprint_receipts=True, normalized_state_type_paths=[], source_sha256={})
    return result, launch, dict(report={'observed_state_checks': deepcopy(states)})


def test_exact_report_keeps_execution_scope_and_all_four_recorded_states():
    verifier.require_report(*reports())


@pytest.mark.parametrize('fault', ['missing', 'disabled', 'integer', 'combined_sharing', 'old_status'])
def test_exact_new_intervention_must_be_declared(fault):
    result, launch, reference = reports()
    if fault == 'missing': launch.pop('invocation_frozen_footprint_receipts')
    elif fault == 'disabled': launch['invocation_frozen_footprint_receipts'] = False
    elif fault == 'integer': launch['invocation_frozen_footprint_receipts'] = 1
    elif fault == 'combined_sharing': launch['invocation_local_surface_receipt_sharing'] = True
    else: result['status'] = 'SHARED_SURFACE_ANCHORED_PREFIX_V1_COMPLETE'
    with pytest.raises(ValueError): verifier.require_report(result, launch, reference)


@pytest.mark.parametrize('fault', ['incomplete', 'goal_claim', 'wall_nan', 'source_change', 'combined_optimization',
    'state_normalization', 'forecast_count', 'model_change', 'missing_state', 'wrong_state', 'real_time', 'boolean_count'])
def test_report_cannot_weaken_completeness_or_expand_claims(fault):
    result, launch, reference = reports()
    if fault == 'incomplete': result['status'] = 'RUNNING'
    elif fault == 'goal_claim': result['goal_achieved'] = True
    elif fault == 'wall_nan': result['wall_s'] = float('nan')
    elif fault == 'source_change': result['source_sha256']['changed'] = 'f'*64
    elif fault == 'combined_optimization': launch['batched_patch_queries_enabled'] = True
    elif fault == 'state_normalization': launch['normalized_state_type_paths'] = ['memory.type']
    elif fault == 'forecast_count': result['report']['raw_model_forecast_comparisons'] = 401
    elif fault == 'model_change': result['report']['model_state_sha256'] = 'f'*64
    elif fault == 'missing_state': result['report']['observed_state_checks'].pop()
    elif fault == 'wrong_state': result['report']['observed_state_checks'][0]['state_sha256'] = 'f'*64
    elif fault == 'real_time': result['report']['real_time_qualified'] = True
    else: result['report']['frames'] = True
    with pytest.raises(ValueError): verifier.require_report(result, launch, reference)


def test_missing_final_result_rejects_before_reading_any_packets(monkeypatch, tmp_path):
    def missing(root, ids): raise ValueError('missing final result')
    def forbidden(*args): raise AssertionError('read after rejected result')
    monkeypatch.setattr(verifier.replay, 'OUTPUT', tmp_path)
    monkeypatch.setattr(verifier, 'verify_artifacts', missing)
    monkeypatch.setattr(verifier, 'read_json', forbidden)
    monkeypatch.setattr(verifier, 'read_rows', forbidden)
    with pytest.raises(ValueError, match='missing final'): verifier.verified_result('f'*64, 'a'*64)


@pytest.mark.parametrize('changed_sources', [False, True])
def test_main_binds_its_sources_before_and_after_result_checks(monkeypatch, tmp_path, changed_sources):
    output = tmp_path/'verification.json'; calls = []
    original = {'source_sha256': {'original.py': 'a'*64}}
    sources = original['source_sha256'] | {'checker.py': 'b'*64}
    monkeypatch.setattr(verifier, 'OUTPUT', output)
    monkeypatch.setattr(sys, 'argv', [verifier.SOURCE, '--result-sha256', 'f'*64, '--launch-sha256', 'a'*64])
    monkeypatch.setattr(verifier, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(verifier, 'read_json', lambda *args: deepcopy(original))
    monkeypatch.setattr(verifier, 'discover_sources', lambda *args: deepcopy(sources))
    monkeypatch.setattr(verifier, 'verify', lambda value: calls.append(('source_check', value)))
    def checked(sha, launch_sha):
        assert launch_sha == 'a'*64
        calls.append(('result_check', sha))
        result = deepcopy(original)
        if changed_sources: result['source_sha256']['original.py'] = 'c'*64
        return result, {'result_sha256': sha}
    monkeypatch.setattr(verifier, 'verified_result', checked)
    if changed_sources:
        with pytest.raises(ValueError, match='source binding changed'): verifier.main()
        assert not output.exists()
    else:
        verifier.main()
        assert json.loads(output.read_text())['verification_source_sha256'] == sources
        assert calls == [('source_check', sources), ('result_check', 'f'*64), ('source_check', sources)]
