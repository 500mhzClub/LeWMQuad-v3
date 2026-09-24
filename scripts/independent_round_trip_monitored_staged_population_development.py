"""Original staged scheduling with mandatory monitored raw-audit acceptance."""
import json
from scripts import independent_round_trip_staged_population_development as staged
from scripts import independent_round_trip_monitored_audit_development as monitored
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/independent_round_trip_monitored_staged_population_development.py'
TEST = 'lewm/tests/test_independent_round_trip_monitored_audit_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_monitored_audit_v1_2026-09-11.md'


def prepared_sources():
    sources = monitored.prepared_sources()
    sources = discover_sources((SOURCE,), sources); verify(sources)
    return sources


def checked_launch(output, launch_sha, verifier, overlap_verifier, *, full=False):
    monitored.require_environment()
    launch = monitored.require_launch(output, launch_sha)
    if SOURCE not in launch['source_sha256']:
        raise ValueError('monitored staged driver must be frozen in launch')
    return staged.checked_launch(output, launch_sha, verifier, overlap_verifier, full=full)


def require_native_slot(audit_ticket=None):
    return monitored.isolated(staged.require_native_slot,
        audit_process=monitored.parent_interface())(audit_ticket)


def worker_entry(*args):
    # This top-level entry is pickleable by spawn. Its original registration
    # barrier remains before either native collection or the monitored auditor.
    return monitored.isolated(staged.worker_entry, separate=monitored)(*args)


class Driver(staged.Driver):
    def check(self, full=False):
        return checked_launch(self.output, self.launch_sha, self.verifier, self.overlap_verifier, full=full)

    def start(self, role, case):
        return monitored.isolated(staged.Driver.start, worker_entry=worker_entry,
            audit_process=monitored.parent_interface(), require_native_slot=require_native_slot)(self, role, case)

    def poll(self):
        # Only the returned audit binding gains a CPU receipt; retain original
        # exit handling, failure draining and ticket cleanup.
        for role, active in tuple(self.active.items()):
            if active.process.is_alive(): continue
            active.process.join()
            try:
                if active.ticket is None: raise ValueError('child ended without completed parent registration')
                if role == 'collection':
                    handoff_sha = digest(staged.artifact_path(self.output, active.case.name+staged.handoff.SUFFIX))
                    staged.collection_process.confirm(active.ticket, handoff_sha, self.verifier)
                    sha = digest(staged.artifact_path(self.output, active.case.name+staged.collection_process.CONFIRMATION))
                    self.schedule.collection_finished(active.case, sha)
                else:
                    name = active.case.name+monitored.EXECUTION
                    execution = staged.evidence.read_json(self.output, name)
                    returned = dict(status=monitored.WORKER_STATUS, case=active.case.name,
                        worker_terminal_sha256=execution['worker_terminal_sha256'],
                        separate_audit_execution_sha256=digest(staged.artifact_path(self.output, name)),
                        cpu_monitor_sha256=digest(staged.artifact_path(self.output, active.case.name+monitored.MONITOR)))
                    accepted = monitored.parent_accept(active.ticket, returned, self.verifier)
                    self.schedule.audit_finished(active.case, accepted)
                    self.bindings.update(accepted['artifact_sha256'])
                    name = active.case.name+staged.audit_process.COMPLETION
                    self.bindings[name] = digest(staged.artifact_path(self.output, name))
            except BaseException as error:
                self.schedule.failed(role, active.case, repr(error))
            finally:
                registry = staged.collection_process._tickets if role == 'collection' else staged.audit_process._tickets
                registry.pop(active.process, None)
                active.ready.close(); active.process.close(); del self.active[role]


def run_population(output, launch_sha, verifier, overlap_verifier):
    return monitored.isolated(staged.run_population, Driver=Driver, checked_launch=checked_launch)(
        output, launch_sha, verifier, overlap_verifier)
