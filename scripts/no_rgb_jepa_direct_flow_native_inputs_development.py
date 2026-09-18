"""Fixed tracking-recovery inputs and operational completion of the native queue."""
import json
from copy import deepcopy
from scripts import no_rgb_jepa_direct_flow_native_prefix_development as prefix
from scripts import independent_round_trip_queue_completion_development as queue
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from lewm.independent_reactive_floor_transport_study_development import merge_sources

replay = prefix.replay
PREPARATION = 'docs/go2_no_rgb_jepa_direct_flow_native_collection_audit_preparation_2026-09-10.json'
PREPARATION_SHA = '1f28f1d1a0829fa5f77b69988175551634b1758468d7ebdf166a5eb1aec3bab4'
REPLAY_OWNER = dict(pid=2749113,created=1789073975.55,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python','-B',replay.SOURCE])


def prepared_sources(seeds=()):
    verify({PREPARATION:PREPARATION_SHA})
    preparation=json.loads((ROOT/PREPARATION).read_text())
    if preparation['status'] != 'DIRECT_FLOW_ANCHORED_NATIVE_COLLECTION_AND_AUDIT_SOURCE_VERIFIED':
        raise ValueError('checked collector, audit and prefix sources required')
    inherited=merge_sources(preparation['source_sha256'],queue.prepared_sources())
    verify(inherited)
    sources=discover_sources((PREPARATION,*seeds),inherited);verify(sources)
    return sources


def owners_ended():
    if owner_live(REPLAY_OWNER): raise ValueError('original full-controller prefix owner remains live')
    queue.owners_ended()


def admit(prefix_sha,waiter_identities,batch_sha,sources):
    owners_ended();verify(sources)
    report=prefix.admit_prefix(prefix_sha,sources)
    result=read_json(replay.OUTPUT,'result.json');launch=read_json(replay.OUTPUT,'launch.json')
    replay.verify_inputs(sources,launch['observer_artifact_sha256'],launch['input_artifact_sha256'],full=True)
    # The queued policies are not adopted by this isolated tracker experiment.
    # Authenticate their completion and artifacts without treating them as new
    # learned-model inputs or claiming a final independent-population review.
    completed=queue.admit(waiter_identities,adapter_batch_result_sha256=batch_sha,sources=sources,full=False)
    old=read_json(replay.batch.OUTPUT,'launch.json')
    admission=dict(prefix_result_sha256=prefix_sha,prefix_artifact_sha256={'result.json':prefix_sha}|result['artifact_sha256'],
        prefix_report=report,observer_artifact_sha256=deepcopy(launch['observer_artifact_sha256']),
        original_episode_artifact_sha256=deepcopy(launch['input_artifact_sha256']),
        correction_admission=deepcopy(old['input_admission']['correction_admission']),
        full_original_model_and_episode_input_admission_reexecuted=True,queue_completion=completed,
        queue_scope='operational completion only; no queued policy adopted',
        final_independent_population_policy_review_performed=False)
    verify_bound(admission,sources)
    return admission


def verify_bound(admission,sources):
    owners_ended();verify(sources);prefix.boundary(admission['prefix_report'])
    if (admission['full_original_model_and_episode_input_admission_reexecuted'] is not True
            or admission['queue_scope'] != 'operational completion only; no queued policy adopted'
            or admission['final_independent_population_policy_review_performed'] is not False
            or admission['prefix_artifact_sha256'].get('result.json') != admission['prefix_result_sha256']):
        raise ValueError('exact tracker input admission and operational queue scope required')
    verify_artifacts(replay.OUTPUT,admission['prefix_artifact_sha256'])
    result=read_json(replay.OUTPUT,'result.json');launch=read_json(replay.OUTPUT,'launch.json')
    if (result['report'] != admission['prefix_report']
            or result['artifact_sha256']|{'result.json':admission['prefix_result_sha256']} != admission['prefix_artifact_sha256']
            or launch['observer_artifact_sha256'] != admission['observer_artifact_sha256']
            or launch['input_artifact_sha256'] != admission['original_episode_artifact_sha256']):
        raise ValueError('all exact original prefix and episode bindings required')
    verify_artifacts(replay.observer.OUTPUT,admission['observer_artifact_sha256'])
    verify_artifacts(replay.batch.OUTPUT,admission['original_episode_artifact_sha256'])
    original=read_json(replay.batch.OUTPUT,'launch.json');replay.batch.verify_inputs(original)
    if original['input_admission']['correction_admission'] != admission['correction_admission']:
        raise ValueError('unchanged original assigned correction and model input required')
    completed=admission['queue_completion']
    again=queue.admit(completed['ordered_waiter_result_sha256'],
        adapter_batch_result_sha256=completed['adapter_batch_result_sha256'],sources=sources,full=False)
    if again != completed: raise ValueError('same complete original native queue required')
