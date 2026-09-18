"""Deterministic complete-population scheduling tests; no process or simulator."""
from copy import deepcopy
from hashlib import sha256
import pytest

from scripts import independent_round_trip_population_schedule_development as scheduling
from scripts import independent_round_trip_audit_process_development as parent
from scripts.independent_round_trip_paired_startup_development import reference_case
from lewm.independent_round_trip_comparison_study_development import CASES


def digest(label):
    return sha256(label.encode()).hexdigest()


def confirmation(case):
    return digest(case.name+' collection')


def completion(schedule, case, success=False):
    return dict(status=parent.STATUS, case=case.name, parent_verified_audit_zero_exit=True,
        owned_audit_exitcode=0, population_case_accepted=True,
        reference_worker_sha256=schedule.reference(case),
        collection_confirmation_sha256=confirmation(case), scientific_success_required=False,
        same_process_collection_and_raw_audit=False, automatic_retry=False,
        native_scene_ownership_released=False, real_time_qualified=False,
        hardware_qualified=False, goal_achieved=False, verified_round_trip=success,
        worker_terminal_sha256=digest(case.name+' terminal'),
        separate_audit_execution_sha256=digest(case.name+' audit'))


def simulate(collection_ticks, audit_ticks, failure=None):
    schedule = scheduling.PopulationSchedule(); active = {}; events = []; tick = 0
    while tick < 2000:
        kind, name = schedule.next_action()
        if kind in ('complete', 'failed'):
            assert not active
            return schedule, events
        if kind in ('collection', 'audit'):
            case = next(case for case in CASES if case.name == name)
            assert kind not in active
            if kind == 'collection':
                first = reference_case(case)
                if first != case:
                    assert first.name in [row['case'] for row in schedule.accepted]
                schedule.collection_started(case, schedule.reference(case))
                duration = collection_ticks + CASES.index(case) % 3
            else:
                schedule.audit_started(case, confirmation(case))
                duration = audit_ticks + CASES.index(case) % 2
            active[kind] = (case, tick+duration)
            events.append((kind, case.name, tick))
            assert len(active) <= 2
            assert schedule.next_collection_index-len(schedule.accepted) <= 2
            continue
        tick += 1
        for kind, (case, until) in tuple(active.items()):
            if tick < until:
                continue
            del active[kind]
            if failure == (kind, CASES.index(case)):
                schedule.failed(kind, case, 'original synthetic worker failed')
            elif kind == 'collection':
                schedule.collection_finished(case, confirmation(case))
            else:
                schedule.audit_finished(case, completion(schedule, case))
            events.append((kind+'_ended', case.name, tick))
    pytest.fail('bounded fixed population did not terminate')


@pytest.mark.parametrize('collection_ticks,audit_ticks', [(1, 9), (9, 1), (3, 3), (2, 5)])
def test_all_32_cases_complete_in_fixed_order_with_bounded_overlap(collection_ticks, audit_ticks):
    schedule, events = simulate(collection_ticks, audit_ticks)
    assert schedule.next_action() == ('complete', None)
    assert [row['case'] for row in schedule.accepted] == [case.name for case in CASES]
    assert len(schedule.collections) == 32 and not schedule.failures
    for stage in ('collection', 'audit'):
        assert [name for kind, name, _ in events if kind == stage] == [case.name for case in CASES]
    assert all(not row['verified_round_trip'] for row in schedule.accepted)
    # There is actual overlap in the virtual schedule after the reference arm.
    audit1_start = next(t for k, n, t in events if k == 'audit' and n == CASES[1].name)
    collect2_start = next(t for k, n, t in events if k == 'collection' and n == CASES[2].name)
    audit1_end = next(t for k, n, t in events if k == 'audit_ended' and n == CASES[1].name)
    assert audit1_start <= collect2_start < audit1_end


@pytest.mark.parametrize('stage', ['collection', 'audit'])
@pytest.mark.parametrize('index', range(32))
def test_every_worker_failure_position_stops_dispatch_and_drains_inflight_work(stage, index):
    schedule, events = simulate(2, 7, failure=(stage, index))
    assert schedule.next_action() == ('failed', None)
    assert len(schedule.failures) == 1 and schedule.failures[0]['case'] == CASES[index].name
    assert len(schedule.accepted) == index
    assert [row['case'] for row in schedule.accepted] == [case.name for case in CASES[:index]]
    assert schedule.collection is None and schedule.audit is None
    # No dispatch occurs after the failing completion event, even if the other
    # original worker finishes later. Its collection bytes remain recorded.
    failure_pos = next(i for i, event in enumerate(events)
        if event[:2] == (stage+'_ended', CASES[index].name))
    assert all(kind.endswith('_ended') for kind, _, _ in events[failure_pos+1:])


def first_auditing():
    schedule = scheduling.PopulationSchedule(); case = CASES[0]
    schedule.collection_started(case, None)
    schedule.collection_finished(case, confirmation(case))
    schedule.audit_started(case, confirmation(case))
    return schedule


def test_first_collection_completion_does_not_release_reference_barrier():
    schedule = first_auditing()
    assert schedule.next_action() == ('wait', None)
    with pytest.raises(ValueError, match='first-arm parent acceptance'): schedule.reference(CASES[1])
    with pytest.raises(ValueError): schedule.collection_started(CASES[1], digest('unaccepted'))
    schedule.audit_finished(CASES[0], completion(schedule, CASES[0]))
    assert schedule.reference(CASES[1]) == digest(CASES[0].name+' terminal')


def test_pending_collection_prevents_third_unaccepted_case():
    schedule = first_auditing(); schedule.audit_finished(CASES[0], completion(schedule, CASES[0]))
    for case in CASES[1:3]:
        schedule.collection_started(case, schedule.reference(case))
        schedule.collection_finished(case, confirmation(case))
        if case == CASES[1]: schedule.audit_started(case, confirmation(case))
    assert schedule.next_action() == ('wait', None)
    with pytest.raises(ValueError): schedule.collection_started(CASES[3], schedule.reference(CASES[3]))
    schedule.audit_finished(CASES[1], completion(schedule, CASES[1]))
    assert schedule.next_action() == ('audit', CASES[2].name)


def test_admission_stop_preserves_running_collection_then_retains_pending_data():
    schedule = scheduling.PopulationSchedule(); case = CASES[0]
    schedule.collection_started(case, None); schedule.stop_dispatch('resource admission failed')
    assert schedule.next_action() == ('wait', None)
    schedule.collection_finished(case, confirmation(case))
    assert schedule.next_action() == ('failed', None)
    assert schedule.pending['collection_confirmation_sha256'] == confirmation(case)
    with pytest.raises(ValueError): schedule.audit_started(case, confirmation(case))


@pytest.mark.parametrize('key,value', [('parent_verified_audit_zero_exit', False),
    ('owned_audit_exitcode', True), ('population_case_accepted', False),
    ('reference_worker_sha256', digest('changed')), ('collection_confirmation_sha256', digest('changed')),
    ('scientific_success_required', True), ('verified_round_trip', 1),
    ('worker_terminal_sha256', 'not a hash')])
def test_changed_parent_completion_cannot_advance_schedule(key, value):
    schedule = first_auditing(); record = completion(schedule, CASES[0]); record[key] = value
    with pytest.raises(ValueError): schedule.audit_finished(CASES[0], record)
    assert not schedule.accepted and schedule.audit['case'] == CASES[0].name


def test_completion_and_snapshot_cannot_alias_scheduler_state():
    schedule = first_auditing(); record = completion(schedule, CASES[0])
    schedule.audit_finished(CASES[0], record)
    record['worker_terminal_sha256'] = digest('mutated')
    assert schedule.reference(CASES[1]) == digest(CASES[0].name+' terminal')
    snapshot = schedule.snapshot(); snapshot['completed_cases'].clear()
    assert schedule.snapshot()['completed_cases'] == [CASES[0].name]


@pytest.mark.parametrize('key,value', [('memory_available_bytes', 63*1024**3),
    ('artifact_free_bytes', 391*1024**3), ('physical_cpus', 3), ('physical_cpus', True)])
def test_conservative_overlap_resource_thresholds(key, value):
    resources = dict(memory_available_bytes=64*1024**3, artifact_free_bytes=500*1024**3, physical_cpus=4)
    schedule = scheduling.PopulationSchedule()
    admitted = schedule.resources_before_collection(resources)
    assert admitted['maximum_unaccepted_cases'] == 2 and not admitted['execution_permitted']
    resources[key] = value
    with pytest.raises(ValueError): schedule.resources_before_collection(resources)


def test_failed_navigation_does_not_select_or_replace_layouts():
    schedule, _ = simulate(1, 1)
    assert all(row['verified_round_trip'] is False for row in schedule.accepted)
    assert schedule.next_collection_index == 32 and len(schedule.accepted) == 32
    assert not schedule.snapshot()['goal_achieved']
