"""Temporary synthetic files only; no new-layout observations or raw audit."""
from copy import deepcopy
import json
import numpy as np
import pytest

from lewm.tests.test_independent_round_trip_population_readout_development import fixture as population_fixture
from lewm.independent_round_trip_comparison_study_development import CASES
from scripts import navigation_artifact_root_development as guard
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.independent_round_trip_paired_startup_development import reference_case
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'BASE', tmp_path)
    output = tmp_path/'go2_synthetic_population_case_attempt_001'; output.mkdir()
    write_json(output/'launch.json', {'synthetic_fixture': True})
    records, reports, _ = population_fixture()
    collections = []
    for case, record, report in zip(CASES, records, reports, strict=True):
        collection = record['collection']
        collection.update(physics_samples=750, decisions=0, completed_ticks=0, command_ticks=0,
            terminal_zero_ticks=0, acquisition_stop='SYNTHETIC_EARLY_PACKET_STOP',
            schedule_terminal=None, mission_receipt=None, setup_checked=False, rgbd_frames=0, auxiliary_frames=0)
        report['observation_and_control_wall_ms'] = []
        report['iteration_with_receipt_wall_ms'] = []
        collections.append(collection)
        for name in evidence.collection_names(case, collection):
            path = output/name; path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith('/result.json'): write_json(path, collection)
            elif name.endswith('/physics_trace.npz'): np.savez(path, physics_contact=np.zeros(750, np.uint8))
            else: path.write_bytes(b'SYNTHETIC PLACEHOLDER; NOT RAW SENSOR EVIDENCE\n')
        (output/(case.name+'_worker.log')).write_text('synthetic fixture; no worker executed\n')
    return output, collections, reports, digest(output/'launch.json')


def persist(saved, index, **changes):
    output, collections, reports, launch = saved; case = CASES[index]
    first = reference_case(case)
    kwargs = dict(launch_sha256=launch,
        collection_artifact_sha256=evidence.bind_collection(output, case, collections[index]),
        worker_log_sha256=digest(output/(case.name+'_worker.log')),
        reference_worker_sha256=None if first == case else digest(output/(first.name+'_worker_terminal.json')))
    kwargs.update(changes)
    record = evidence.persist_audited_case(output, case, collections[index], reports[index], **kwargs)
    return record, digest(output/(case.name+'_worker_terminal.json'))


def rewrite(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def test_all_negative_synthetic_population_retains_all_units_and_scope(saved):
    output, _, _, launch = saved
    ids = [[case.name, persist(saved, i)[1]] for i, case in enumerate(CASES)]
    result = evidence.complete_saved_population(output, ids, launch_sha256=launch)
    summary = result['summary']
    assert summary['completed_episodes'] == 32 and summary['independent_layout_units'] == 8
    assert summary['measured_round_trip_successes'] == 0 and not summary['all_startups_matched']
    assert all(arm['scientific_failures'] == 8 for arm in summary['arms'])
    assert all(row['readout']['observation_and_control']['median_ms'] is None
        for row in summary['ordered_case_readouts'])
    assert result['saved_case_evidence_authenticated']
    assert not result['original_raw_audit_reexecuted'] and not result['population_execution_permitted']
    assert not summary['goal_achieved'] and not summary['navigation_qualified']
    assert all(case.name+'_worker_terminal.json' in result['artifact_sha256'] for case in CASES)


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5, 8, 9, 12, 13])
def test_rotated_fixed_reference_and_exact_persisted_records(saved, index):
    output, collections, reports, launch = saved
    first = CASES.index(reference_case(CASES[index]))
    first_sha = None
    if index != first: _, first_sha = persist(saved, first)
    record, sha = persist(saved, index)
    found, report, contact, bindings = evidence.read_audited_case(output, CASES[index], sha, launch_sha256=launch)
    assert found == record and report == reports[index]
    assert record['collection'] == collections[index] and record['reference_worker_sha256'] == first_sha
    assert len(contact) == 750 and not contact.any()
    assert record['startup_comparison']['reference_case'] == CASES[first].name
    assert record['startup_comparison']['status'] == 'UNAVAILABLE_EARLY_STOP'
    assert bindings[CASES[index].name+'_worker_terminal.json'] == sha


@pytest.mark.parametrize('fault', ['missing_binding', 'extra_binding', 'changed_raw', 'wrong_launch',
    'wrong_log', 'failed_audit', 'wrong_model', 'false_success', 'wrong_collection', 'existing_output'])
def test_case_failure_does_not_publish_terminal_or_replace_evidence(saved, fault):
    output, collections, reports, launch = saved; case = CASES[0]
    original = evidence.bind_collection(output, case, collections[0]); kwargs = {'collection_artifact_sha256': original}
    raw = case.name+'/command_tape.json'; touched = None
    if fault == 'missing_binding': original.pop(raw)
    elif fault == 'extra_binding': original['launch.json'] = launch
    elif fault == 'changed_raw': (output/raw).write_text('changed after pre-audit binding')
    elif fault == 'wrong_launch': kwargs['launch_sha256'] = '0'*64
    elif fault == 'wrong_log': kwargs['worker_log_sha256'] = '0'*64
    elif fault == 'failed_audit': reports[0]['raw_sensor_reconstruction_pass'] = False
    elif fault == 'wrong_model': reports[0]['assignment']['model_state_sha256'] = '0'*64
    elif fault == 'false_success': reports[0]['verified_round_trip'] = True
    elif fault == 'wrong_collection': collections[0]['physical_stop'] = 'different'
    else:
        touched = output/(case.name+'_audit.json'); touched.write_text('retain existing failed output')
    with pytest.raises(ValueError): persist(saved, 0, **kwargs)
    assert not (output/(case.name+'_worker_terminal.json')).exists()
    assert (output/raw).exists()
    if touched is not None: assert touched.read_text() == 'retain existing failed output'


@pytest.mark.parametrize('suffix', ['/command_tape.json', '/result.json', '/physics_trace.npz',
    '_audit.json', '_readout.json', '_startup_comparison.json', '_worker.log', '_worker_terminal.json'])
def test_completed_artifact_tampering_rejected(saved, suffix):
    output, _, _, launch = saved
    _, sha = persist(saved, 0); path = output/(CASES[0].name+suffix)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError): evidence.read_audited_case(output, CASES[0], sha, launch_sha256=launch)


@pytest.mark.parametrize('fault', ['missing_raw', 'different_collection', 'different_readout', 'different_startup',
    'different_log_hash', 'different_launch', 'wrong_reference', 'failed_worker'])
def test_rehashed_terminal_cannot_hide_inconsistent_record(saved, fault):
    output, _, _, launch = saved; case = CASES[0]
    record, _ = persist(saved, 0); record = deepcopy(record)
    if fault == 'missing_raw': record['artifact_sha256'].pop(case.name+'/command_tape.json')
    elif fault == 'different_collection': record['collection']['physical_stop'] = 'changed'
    elif fault == 'different_readout': record['readout']['native_contact_samples'] = 1
    elif fault == 'different_startup': record['startup_comparison']['status'] = 'EXACT_PRECOMMAND_STARTUP'
    elif fault == 'different_log_hash': record['worker_log_sha256'] = '0'*64
    elif fault == 'different_launch': record['launch_sha256'] = '0'*64
    elif fault == 'wrong_reference': record['reference_worker_sha256'] = '0'*64
    else: record['failure'] = 'original audit failed'
    path = output/(case.name+'_worker_terminal.json'); rewrite(path, record)
    with pytest.raises(ValueError): evidence.read_audited_case(output, case, digest(path), launch_sha256=launch)


def test_changed_reference_rejected_before_later_case_published(saved):
    output, _, _, _ = saved; _, sha = persist(saved, 0)
    (output/(CASES[0].name+'/command_tape.json')).write_text('changed reference bytes')
    with pytest.raises(ValueError): persist(saved, 1, reference_worker_sha256=sha)
    assert not (output/(CASES[1].name+'_worker_terminal.json')).exists()


@pytest.mark.parametrize('reference_sha', [None, '0'*64])
def test_later_case_requires_exact_completed_reference(saved, reference_sha):
    persist(saved, 0)
    with pytest.raises(ValueError): persist(saved, 1, reference_worker_sha256=reference_sha)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'reorder', 'bad_hash'])
def test_population_never_drops_or_substitutes_workers(saved, fault):
    output, _, _, launch = saved
    ids = [[case.name, '0'*64] for case in CASES]
    if fault == 'missing': ids.pop()
    elif fault == 'duplicate': ids[1] = ids[0]
    elif fault == 'reorder': ids[0], ids[1] = ids[1], ids[0]
    with pytest.raises(ValueError): evidence.complete_saved_population(output, ids, launch_sha256=launch)


def test_changes_during_startup_reader_are_detected(saved, monkeypatch):
    output, _, _, _ = saved; original = evidence.compare_case_startup
    def altered(*args):
        receipt = original(*args)
        (output/(CASES[0].name+'/command_tape.json')).write_text('changed during startup')
        return receipt
    monkeypatch.setattr(evidence, 'compare_case_startup', altered)
    with pytest.raises(ValueError): persist(saved, 0)
    assert not (output/(CASES[0].name+'_worker_terminal.json')).exists()


@pytest.mark.parametrize('name', ['sealed_test.json', 'sealed/forbidden.json', 'sealed_legacy/forbidden.json'])
def test_extra_protected_binding_rejected_without_opening_it(saved, name):
    output, collections, _, _ = saved; ids = evidence.bind_collection(output, CASES[0], collections[0])
    ids[name] = '0'*64
    with pytest.raises(ValueError): persist(saved, 0, collection_artifact_sha256=ids)


def test_symlinked_artifact_is_rejected(saved):
    output, collections, _, _ = saved
    path = output/(CASES[0].name+'/command_tape.json'); path.unlink(); path.symlink_to(output/'launch.json')
    with pytest.raises(ValueError): evidence.bind_collection(output, CASES[0], collections[0])


@pytest.mark.parametrize('mismatch', [False, True])
def test_matched_startup_delegates_to_original_prefix_comparison(saved, monkeypatch, mismatch):
    from scripts import independent_round_trip_paired_startup_development as paired
    from scripts.all_phase_residual_maze02_startup_development import admit_startup
    from lewm.tests.test_all_phase_residual_maze02_startup_development import fixture as startup_fixture
    output, collections, reports, _ = saved
    for i in (0, 1):
        collections[i].update(physics_samples=950, decisions=4, completed_ticks=4, command_ticks=4,
            acquisition_stop=None, schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED')
        rewrite(output/(CASES[i].name+'/result.json'), collections[i])
        np.savez(output/(CASES[i].name+'/physics_trace.npz'), physics_contact=np.zeros(950, np.uint8))
        reports[i]['observation_and_control_wall_ms'] = [150.]*4
        reports[i]['iteration_with_receipt_wall_ms'] = [160.]*4
    seen = []
    def comparison(reference, candidate):
        seen.append((reference, candidate)); args = startup_fixture()
        if reference != candidate:
            args[2][1][3]['decision']['requested_command'] = [.2, 0., 0.]
            if mismatch: args[0][1]['position'][899, 0] = .001
        return admit_startup(*args)
    # Substitute only disk packet loading; exercise the original numerical
    # prefix comparison and the actual paired startup and persistence layers.
    monkeypatch.setattr(paired, 'compare_startup', comparison)
    persist(saved, 0)
    if mismatch:
        with pytest.raises(ValueError): persist(saved, 1)
        assert not (output/(CASES[1].name+'_worker_terminal.json')).exists()
    else:
        record, _ = persist(saved, 1); startup = record['startup_comparison']
        assert startup['status'] == 'EXACT_PRECOMMAND_STARTUP'
        assert startup['reference_first_command'] != startup['candidate_first_command']
        assert not startup['later_physical_outcomes_compared']
    assert seen == [(output/CASES[0].name, output/CASES[0].name),
        (output/CASES[0].name, output/CASES[1].name)]


@pytest.mark.parametrize('bad', [None, {}, [None]*32, [[]]*32, ['ab']*32])
def test_malformed_worker_roster_rejects_before_any_file_read(saved, bad):
    output, _, _, launch = saved
    with pytest.raises(ValueError): evidence.complete_saved_population(output, bad, launch_sha256=launch)
