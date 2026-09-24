"""Private fresh-output bindings and authenticated ended V1 failure evidence."""
import json
from types import FunctionType, SimpleNamespace

from scripts import external_body_projection_profile_replay_development as original_replay
from scripts import body_projection_external_profile_admission_development as original_admission
from scripts.navigation_artifact_root_development import BASE, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from lewm.independent_reactive_floor_transport_study_development import merge_sources

OUTPUT=BASE/'go2_body_projected_external_sampling_v2_attempt_001'
FAILED=BASE/'go2_body_projected_external_sampling_v1_attempt_001'
AUDIT='docs/go2_external_profile_v1_tracer_lock_failure_audit_2026-09-11.json'
AUDIT_SHA='c0ec1fd70fec1ea2e2848e5bb6eeab407c664e4e607e16234036af3d982c20cb'
WATCH='docs/go2_body_projection_external_profile_completion_watch_execution_2026-09-11.json'
DERIVATIVE='docs/go2_body_projection_external_profile_v2_source_derivatives_2026-09-11.json'


def fork(module, names, output):
    bindings=vars(module) | {'OUTPUT':output}
    for name in names:
        function=getattr(module,name)
        if function.__closure__ is not None:
            raise ValueError('closure-free original replay/admission function required')
        clone=FunctionType(function.__code__,bindings,function.__name__,function.__defaults__)
        clone.__kwdefaults__=function.__kwdefaults__
        bindings[name]=clone
    return SimpleNamespace(**bindings)


def fork_replay(output):
    return fork(original_replay, ('isolated_replay','replay'), output)


def fork_admission(output):
    return fork(original_admission, ('capture_verification','admit_completed'), output)


replay=fork_replay(OUTPUT)
admission=fork_admission(OUTPUT)


def failed_sources(inherited):
    verify({AUDIT:AUDIT_SHA})
    audit=json.loads((ROOT/AUDIT).read_text())
    if (audit['status']!='TERMINAL_EXTERNAL_PROFILE_V1_TRACER_LOCK_FAILURE_PRESERVED'
            or audit['full_replay_completed'] is not False or audit['complete_profile_written'] is not False
            or audit['completion_checker_invoked'] is not False):
        raise ValueError('preserved original terminal tracing failure required')
    verify(audit['document_sha256'])
    verify_artifacts(FAILED,audit['artifact_sha256'])
    watch=json.loads((ROOT/WATCH).read_text())
    records=[json.loads((FAILED/name).read_text()) for name in
        ('launch.json','child_execution.json','profiler_execution.json')]+[watch]
    if any(record['boot_id']!=BOOT or owner_live(record['owner']) for record in records):
        raise ValueError('all original failed profile owners and watcher must be ended')
    if any((FAILED/name).exists() or (FAILED/name).is_symlink() for name in ('result.json','profile.json')):
        raise ValueError('original missing profile and unsuccessful replay evidence required')
    sources=merge_sources(inherited,records[0]['source_sha256'],watch['source_sha256'])
    sources=discover_sources((AUDIT,DERIVATIVE,*audit['document_sha256']),sources)
    verify(sources)
    return sources
