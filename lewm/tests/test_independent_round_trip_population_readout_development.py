"""Synthetic population accounting; no simulated or physical outcome claims."""
from copy import deepcopy

import numpy as np
import pytest

from lewm.independent_round_trip_comparison_study_development import CASES, require_case
from lewm.independent_round_trip_multiarm_contract_development import COLLECTION_STATUS, treatment, replay_receipt
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS
from lewm import independent_round_trip_population_readout_development as study
from scripts import independent_round_trip_paired_startup_development as startup


def startup_receipt(case, reference, current):
    receipt = startup.definition(case, reference, current)
    if receipt['status'] == startup.MATCHED:
        receipt.update(common_prefix_frames=4, physical_prefix_samples=900, completed_zero_warmup_commands=3,
            raw_physics_prefix_sha256=f'{case.layout_index:064x}', public_startup_sha256=f'{case.layout_index+100:064x}',
            model_forecasts_required_equal=False, post_warmup_controller_state_required_equal=False,
            reference_first_command=[0., 0., 0.], candidate_first_command=[0., 0., 0.],
            reference_first_command_terminal=None, candidate_first_command_terminal=None)
    return receipt


def fixture():
    records = []; audits = []; contacts = []
    for case in CASES:
        collection = dict(status=COLLECTION_STATUS, **treatment(case), physics_samples=1450, decisions=15,
            completed_ticks=14, command_ticks=14, terminal_zero_ticks=10,
            physical_stop=None, acquisition_stop=None, schedule_terminal='SENSOR_OR_MODEL_FAILURE',
            mission_receipt=dict(arrivals=[], terminal='SENSOR_OR_MODEL_FAILURE'))
        report = dict(**treatment(case), **replay_receipt(case), raw_sensor_reconstruction_pass=True,
            raw_command_audit_pass=True, verified_round_trip=False, strict_physical_visibility_pass=True,
            hard_measurement_failed_frames=[], renderer_capture_audit={'synthetic_fixture': True},
            native_evaluation=dict(arrival_windows=[], outbound_traversal=dict(crossings=[]), return_traversal=None,
                physically_retraced_outbound_route=False, terminal_native_quiet_pass=False,
                native_round_trip_candidate_pass=False),
            observation_and_control_wall_ms=[150.]*15, iteration_with_receipt_wall_ms=[160.]*15)
        contact = np.zeros(1450, np.uint8)
        record = dict(status=study.WORKER_STATUS, **treatment(case), collection=collection,
            model_state_sha256=require_case(case).model_state_sha256,
            **{key:deepcopy(report[key]) for key in OUTCOME_KEYS}, readout=study.case_readout(report, collection, contact))
        records.append(record); audits.append(report); contacts.append(contact)
    refresh_startups(records)
    return records, audits, contacts


def refresh_startups(records):
    for case, record in zip(CASES, records, strict=True):
        reference = records[4*case.layout_index]['collection']
        record['startup_comparison'] = startup_receipt(case, reference, record['collection'])


def refresh(index, records, audits, contacts):
    for key in OUTCOME_KEYS: records[index][key] = deepcopy(audits[index][key])
    records[index]['readout'] = study.case_readout(audits[index], records[index]['collection'], contacts[index])


def test_all_32_negative_episodes_keep_eight_layout_denominators():
    report = study.complete_population(*fixture())
    assert report['completed_episodes'] == 32 and report['independent_layout_units'] == 8
    assert report['measured_round_trip_successes'] == 0 and report['all_startups_matched']
    assert all(row['planned_layouts'] == row['scientific_failures'] == 8 for row in report['arms'])
    assert all(row['layout_pairs'] == row['tied_outcomes'] == 8 for row in report['comparisons'])
    assert not any(report[key] for key in ('goal_achieved', 'navigation_qualified', 'real_time_qualified', 'hardware_qualified'))
    assert all(row['readout']['observation_and_control']['samples_above_command_interval_100ms'] == 15
        for row in report['ordered_case_readouts'])


@pytest.mark.parametrize('fault', ['missing', 'reorder', 'model', 'raw', 'worker_failure',
    'startup_reference', 'startup_hash', 'readout', 'contact_length', 'timing_length', 'reactive_replay'])
def test_incomplete_or_misbound_population_rejected(fault):
    records, audits, contacts = fixture()
    if fault == 'missing': records.pop(); audits.pop(); contacts.pop()
    elif fault == 'reorder': records[0], records[1] = records[1], records[0]
    elif fault == 'model': records[0]['model_state_sha256'] = 'different_seed'
    elif fault == 'raw': audits[0]['raw_sensor_reconstruction_pass'] = False
    elif fault == 'worker_failure': records[0]['failure'] = 'incomplete audit'
    elif fault == 'startup_reference': records[1]['startup_comparison']['reference_case'] = CASES[1].name
    elif fault == 'startup_hash': records[1]['startup_comparison']['public_startup_sha256'] = 'f'*64
    elif fault == 'readout': records[0]['readout']['native_contact_samples'] = 1
    elif fault == 'contact_length': contacts[0] = contacts[0][:-1]
    elif fault == 'timing_length': audits[0]['observation_and_control_wall_ms'].pop()
    else: audits[2]['raw_model_command_replay_pass'] = True
    with pytest.raises(ValueError): study.complete_population(records, audits, contacts)


def mark_success(index, records, audits, contacts):
    collection = records[index]['collection']; evaluation = audits[index]['native_evaluation']
    collection['schedule_terminal'] = 'OBSERVED_ROUND_TRIP_CANDIDATE'
    collection['mission_receipt'] = dict(terminal='OBSERVED_ROUND_TRIP_CANDIDATE', arrivals=[{'phase':'OUTBOUND'}, {'phase':'RETURN'}])
    evaluation.update(native_round_trip_candidate_pass=True, physically_retraced_outbound_route=True,
        terminal_native_quiet_pass=True, arrival_windows=[dict(phase=phase, native_one_second_arrival_and_quiet_pass=True)
            for phase in ('OUTBOUND', 'RETURN')])
    audits[index]['verified_round_trip'] = True
    refresh(index, records, audits, contacts)


def test_paired_descriptive_counts_do_not_claim_reliability_or_causation():
    records, audits, contacts = fixture()
    for index, case in enumerate(CASES):
        if ((case.layout_index == 0 and case.arm_name == 'persistent_jepa')
                or (case.layout_index == 1 and case.arm_name == 'reactive')):
            mark_success(index, records, audits, contacts)
    report = study.complete_population(records, audits, contacts)
    assert report['measured_round_trip_successes'] == 2
    training, method, memory = report['comparisons']
    assert training['paired_success_rate_difference'] == memory['paired_success_rate_difference'] == .125
    assert method['left_only_successes'] == method['right_only_successes'] == 1
    assert method['paired_success_rate_difference'] == 0 and method['tied_outcomes'] == 6
    assert all(row['descriptive_only'] and not row['advantage_established'] for row in report['comparisons'])
    assert not report['navigation_qualified'] and not report['goal_achieved']


@pytest.mark.parametrize('fault', ['visibility', 'hard_frame', 'contact', 'stop', 'arrival', 'return', 'quiet', 'drain'])
def test_agreeing_receipts_cannot_promote_failed_physical_or_visibility_gates(fault):
    records, audits, contacts = fixture(); mark_success(0, records, audits, contacts)
    if fault == 'visibility': audits[0]['strict_physical_visibility_pass'] = False
    elif fault == 'hard_frame': audits[0]['hard_measurement_failed_frames'] = [4]
    elif fault == 'contact': contacts[0][1000] = 1
    elif fault == 'stop': records[0]['collection']['physical_stop'] = 'CONTACT_STOP'
    elif fault == 'arrival': audits[0]['native_evaluation']['arrival_windows'].pop()
    elif fault == 'return': audits[0]['native_evaluation']['physically_retraced_outbound_route'] = False
    elif fault == 'quiet': audits[0]['native_evaluation']['terminal_native_quiet_pass'] = False
    else: records[0]['collection']['terminal_zero_ticks'] = 9
    refresh(0, records, audits, contacts)
    with pytest.raises(ValueError): study.complete_population(records, audits, contacts)


@pytest.mark.parametrize('index', [0, 1])
def test_short_audited_native_stop_stays_in_population_without_matching_claim(index):
    records, audits, contacts = fixture()
    collection = records[index]['collection']
    collection.update(physics_samples=800, decisions=1, completed_ticks=1, command_ticks=1,
        physical_stop='EARLY_CONTACT_STOP', terminal_zero_ticks=0, schedule_terminal=None)
    contacts[index] = np.zeros(800, np.uint8); contacts[index][-1] = 1
    audits[index]['observation_and_control_wall_ms'] = [150.]
    audits[index]['iteration_with_receipt_wall_ms'] = [160.]
    refresh(index, records, audits, contacts); refresh_startups(records)
    result = study.complete_population(records, audits, contacts)
    assert result['completed_episodes'] == 32 and not result['all_startups_matched']
    assert all(arm['planned_layouts'] == 8 for arm in result['arms'])
    assert not result['ordered_case_readouts'][index]['startup_comparison']['physical_and_public_startup_exact']
    if index == 0:
        assert all(row['startup_comparison']['status'] == startup.UNAVAILABLE for row in result['ordered_case_readouts'][:4])


def test_no_observation_stop_retains_zero_timing_sample_count():
    records, audits, contacts = fixture(); collection = records[0]['collection']
    collection.update(physics_samples=750, decisions=0, completed_ticks=0, command_ticks=0,
        acquisition_stop='PACKET_CONTRACT_STOP', terminal_zero_ticks=0, schedule_terminal=None, mission_receipt=None)
    contacts[0] = np.zeros(750, np.uint8)
    audits[0]['observation_and_control_wall_ms'] = []; audits[0]['iteration_with_receipt_wall_ms'] = []
    refresh(0, records, audits, contacts); refresh_startups(records)
    report = study.complete_population(records, audits, contacts)
    value = report['ordered_case_readouts'][0]['readout']['observation_and_control']
    assert value['samples'] == 0 and value['median_ms'] is None


def test_repeated_crossings_remain_one_distinct_edge():
    records, audits, contacts = fixture()
    forward = dict(from_cell=[-1, 0], to_cell=[0, 0], declared_open_edge=True)
    backward = dict(from_cell=[0, 0], to_cell=[-1, 0], declared_open_edge=True)
    audits[0]['native_evaluation']['outbound_traversal']['crossings'] = [forward, backward, forward]
    refresh(0, records, audits, contacts)
    report = study.complete_population(records, audits, contacts)
    readout = report['ordered_case_readouts'][0]['readout']['outbound']
    assert readout['total_crossings'] == 3 and readout['distinct_open_edges'] == 1


def test_actual_startup_reader_uses_only_declared_same_layout_directories(monkeypatch, tmp_path):
    records, _, _ = fixture(); case = CASES[5]
    calls = []
    monkeypatch.setattr(startup, 'validate_root', lambda root: root)
    details = startup_receipt(case, records[4]['collection'], records[5]['collection'])
    for owner, old in (('reference', 'original'), ('candidate', 'candidate')):
        details[old+'_first_model_command'] = details.pop(owner+'_first_command')
        details[old+'_first_model_terminal'] = details.pop(owner+'_first_command_terminal')
    # The reader's model forecasts/first commands may differ legitimately.
    details['candidate_first_model_command'] = [.16, 0., .45]
    def compare(first, second): calls.append((first, second)); return deepcopy(details)
    monkeypatch.setattr(startup, 'compare_startup', compare)
    receipt = startup.compare_case_startup(tmp_path, case, records[4]['collection'], records[5]['collection'])
    assert calls == [(tmp_path/CASES[4].name, tmp_path/CASES[5].name)]
    assert receipt['candidate_first_command'] == [.16, 0., .45]
    assert receipt['physical_and_public_startup_exact'] and not receipt['model_forecasts_required_equal']


def test_short_stop_cannot_fabricate_startup_hash_or_silently_ignore_missing_data(monkeypatch):
    records, _, _ = fixture(); case = CASES[0]; collection = records[0]['collection']
    collection['decisions'] = 0
    with pytest.raises(ValueError, match='recorded'): startup.definition(case, collection, collection)
    collection['acquisition_stop'] = 'PACKET_CONTRACT_STOP'
    def forbidden(*args): pytest.fail('incomplete startup accessed unavailable packets')
    monkeypatch.setattr(startup, 'compare_startup', forbidden)
    receipt = startup.compare_case_startup(None, case, collection, collection)
    receipt['public_startup_sha256'] = 'a'*64
    with pytest.raises(ValueError, match='additional'): startup.require_startup(case, collection, collection, receipt)
