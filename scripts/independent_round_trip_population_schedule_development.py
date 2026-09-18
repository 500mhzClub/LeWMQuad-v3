"""Fixed-order, bounded scheduling state for the prospective split population.

This module owns dispatch order, not process handles or artifact admission.
The future driver must call the frozen lifecycle checks before reporting a
completion here. No process, scene, model, or dataset is created by this module.
"""
from copy import deepcopy
import json

from scripts import independent_round_trip_audit_process_development as parent
from scripts.independent_round_trip_paired_startup_development import reference_case
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_round_trip_comparison_study_development import CASES, require_case, resources_for

SOURCE = 'scripts/independent_round_trip_population_schedule_development.py'
TEST = 'lewm/tests/test_independent_round_trip_population_schedule_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_population_schedule_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_round_trip_audit_process_preparation_2026-09-11.json'
PREPARATION_SHA = 'b03fd76db71e7eca675787dfd85a2a3b0c57dc357e9b027212bf26876254a06d'


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']
    verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited)
    verify(sources)
    return sources


def require_sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('exact lowercase SHA-256 identity required')


class PopulationSchedule:
    """One collection, one audit, at most two unaccepted cases; no retries."""

    def __init__(self):
        self.next_collection_index = 0
        self.collection = None
        self.audit = None
        self.pending = None
        self.accepted = []
        self.failures = []
        self.collections = []

    def reference(self, case):
        require_case(case)
        first = reference_case(case)
        if first == case:
            return None
        records = {row['case']:row for row in self.accepted}
        if first.name not in records:
            raise ValueError('first-arm parent acceptance required before collection')
        return records[first.name]['worker_terminal_sha256']

    def next_action(self):
        if self.failures:
            return ('wait', None) if self.collection is not None or self.audit is not None else ('failed', None)
        if self.pending is not None and self.audit is None:
            return 'audit', self.pending['case']
        if self.collection is None and self.pending is None and self.next_collection_index < len(CASES):
            case = CASES[self.next_collection_index]
            first = reference_case(case)
            if first == case or any(row['case'] == first.name for row in self.accepted):
                return 'collection', case.name
        if len(self.accepted) == len(CASES):
            if self.collection is not None or self.audit is not None or self.pending is not None:
                raise ValueError('completed population cannot retain work')
            return 'complete', None
        if self.collection is None and self.audit is None:
            raise ValueError('schedule cannot advance; original ordered evidence is inconsistent')
        return 'wait', None

    def collection_started(self, case, reference_sha):
        require_case(case)
        if self.next_action() != ('collection', case.name) or reference_sha != self.reference(case):
            raise ValueError('only the next fixed collection with original accepted reference may start')
        self.collection = dict(case=case.name, reference_worker_sha256=reference_sha)
        self.next_collection_index += 1

    def collection_finished(self, case, confirmation_sha):
        """Called only after lifecycle.confirm accepted the owned collector exit."""
        require_case(case); require_sha(confirmation_sha)
        if self.collection is None or self.collection['case'] != case.name or self.pending is not None:
            raise ValueError('one original active collector and one free pending slot required')
        finished = dict(self.collection, collection_confirmation_sha256=confirmation_sha)
        self.pending = finished
        self.collections.append(deepcopy(finished))
        self.collection = None

    def audit_started(self, case, confirmation_sha):
        require_case(case); require_sha(confirmation_sha)
        if (self.next_action() != ('audit', case.name)
                or confirmation_sha != self.pending['collection_confirmation_sha256']):
            raise ValueError('only the next pending original collection may enter audit')
        self.audit = self.pending
        self.pending = None

    def audit_finished(self, case, completion):
        """Consume parent.accept's authenticated result; this is not its verifier."""
        require_case(case)
        if (self.audit is None or self.audit['case'] != case.name
                or len(self.accepted) >= len(CASES) or CASES[len(self.accepted)] != case):
            raise ValueError('original active audit in fixed acceptance order required')
        expected = dict(status=parent.STATUS, case=case.name, parent_verified_audit_zero_exit=True,
            owned_audit_exitcode=0, population_case_accepted=True,
            reference_worker_sha256=self.audit['reference_worker_sha256'],
            collection_confirmation_sha256=self.audit['collection_confirmation_sha256'],
            scientific_success_required=False, same_process_collection_and_raw_audit=False,
            automatic_retry=False, native_scene_ownership_released=False,
            real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
        if any(type(completion.get(k)) is not type(v) or completion[k] != v for k, v in expected.items()):
            raise ValueError('original parent acceptance with unchanged case and collection required')
        for key in ('worker_terminal_sha256', 'separate_audit_execution_sha256'):
            require_sha(completion[key])
        if type(completion['verified_round_trip']) is not bool:
            raise ValueError('original boolean scientific outcome required')
        self.accepted.append(deepcopy(completion))
        self.audit = None

    def failed(self, stage, case, reason):
        """Record an ended failed worker; do not cancel the other active stage."""
        require_case(case)
        if type(reason) is not str or not reason:
            raise ValueError('retained failure reason required')
        if stage == 'collection':
            if self.collection is None or self.collection['case'] != case.name:
                raise ValueError('original active collection required for failure')
            self.collection = None
        elif stage == 'audit':
            if self.audit is None or self.audit['case'] != case.name:
                raise ValueError('original active audit required for failure')
            self.audit = None
        else:
            raise ValueError('collection or audit stage required')
        self.failures.append(dict(stage=stage, case=case.name, reason=reason))

    def stop_dispatch(self, reason):
        """Admission/resource failure stops dispatch while existing workers drain."""
        if type(reason) is not str or not reason:
            raise ValueError('retained admission failure reason required')
        self.failures.append(dict(stage='dispatch_admission', case=None, reason=reason))

    def resources_before_collection(self, resources):
        if self.next_action()[0] != 'collection':
            raise ValueError('resource admission only for the next eligible collection')
        admitted = resources_for(resources, [row['case'] for row in self.accepted])
        # Conservatively count every unaccepted case in remaining disk allowance,
        # including data already collected for the concurrent audit.
        if (resources['memory_available_bytes'] < 64*1024**3
                or type(resources['physical_cpus']) is not int or resources['physical_cpus'] < 4):
            raise ValueError('64 GiB available memory and four physical CPUs required for staged runtime')
        return dict(admitted, memory_admission_bytes=64*1024**3,
            maximum_active_collectors=1, maximum_active_auditors=1,
            maximum_unaccepted_cases=2, cpu_only_audit_qualified=False,
            native_idle_checked=False, execution_permitted=False)

    def snapshot(self):
        return deepcopy(dict(next_collection_index=self.next_collection_index,
            active_collection=self.collection, active_audit=self.audit, pending_collection=self.pending,
            completed_cases=[row['case'] for row in self.accepted],
            completed_collection_confirmations=self.collections, failures=self.failures,
            next_action=list(self.next_action()), automatic_retry=False,
            population_execution_permitted=False, goal_achieved=False))
