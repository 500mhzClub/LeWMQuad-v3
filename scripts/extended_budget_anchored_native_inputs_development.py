"""Original six-case results, assigned model and four-stage queue admission."""
from copy import deepcopy
import json

from scripts import independent_round_trip_extended_queue_completion_development as queue
from scripts import reached_frontier_native_inputs_development as cohort
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

batch = cohort.batch
CASE = batch.CASES[-1]
MODEL_SHA = '56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd'
COLLECTION_SHA = 'c323046813ca032fc2012a5f2774aef2bebd5c09937da7d0feb70c5ce577109b'
PREPARATION = 'docs/go2_extended_budget_anchored_prefix_preparation_2026-09-11.json'
PREPARATION_SHA = '13fea1e1ff7b30d562b0e48401e0613fc4f235b9808e88accb6a4aa13b7ba367'


def prepared_sources(seeds=()):
    verify({PREPARATION:PREPARATION_SHA})
    prior = json.loads((ROOT/PREPARATION).read_text())
    if prior['status'] != 'EXTENDED_BUDGET_ANCHORED_PREFIX_SOURCE_TESTED_NO_EXECUTION':
        raise ValueError('tested original budget pipeline and prefix preparation required')
    sources = merge_sources(prior['source_sha256'], queue.prepared_sources())
    sources = discover_sources((PREPARATION, *seeds), sources)
    verify(sources)
    return sources


def original_batch(batch_sha, sources, *, full=False):
    result, launch, ids = completed(batch.OUTPUT, batch_sha, cohort.BATCH_LAUNCH, sources)
    batch.verify_inputs(launch, full=full)
    groups = [[read_json(batch.OUTPUT, c[0]+suffix) for c in batch.CASES]
        for suffix in ('_worker_terminal.json', '_audit.json', '_startup_comparison.json', '_readout.json')]
    for c in batch.CASES:
        for suffix in ('_worker_terminal.json', '_audit.json', '_startup_comparison.json', '_readout.json', '_worker.log'):
            if c[0]+suffix not in ids: raise ValueError('all six original worker/audit/readout bindings required')
    review = cohort.require_batch(result, launch, *groups)
    target = groups[0][-1]; collection = read_json(batch.OUTPUT, CASE[0]+'/result.json')
    expected = {CASE[0]+'/'+n for n in batch.artifacts(2, collection)}
    if (launch['assigned_model_states'][CASE[4]] != MODEL_SHA
            or ids.get(CASE[0]+'/result.json') != COLLECTION_SHA
            or target['collection'] != collection or not expected <= ids.keys()):
        raise ValueError('exact original completed no-RGB direct case and all raw artifacts required')
    return dict(batch_result_sha256=batch_sha, batch_artifact_sha256=ids,
        cohort_review=review, original_case=list(CASE), original_model_state_sha256=MODEL_SHA,
        original_case_artifact_sha256={n:ids[n] for n in sorted(expected)},
        correction_admission=deepcopy(launch['input_admission']['correction_admission']))


def admit(batch_sha, waiter_ids, tracking_sha, sources):
    # Preserve original ownership before any expensive input verification.
    queue.owners_ended(); queue.original.owners_ended(); verify(sources)
    prior = original_batch(batch_sha, sources, full=True)
    original_queue = queue.original.admit(waiter_ids, adapter_batch_result_sha256=batch_sha,
        sources=sources, full=True)
    extended_queue = queue.admit(original_queue, tracking_sha, sources=sources, full=True)
    return prior | dict(extended_queue_completion=extended_queue,
        full_original_input_admission_performed=True, all_scientific_failures_retained=True,
        queued_controller_changes_adopted=False, budget_only_development_followup=True,
        final_independent_population_policy_review_performed=False)


def verify_bound(admission, sources):
    queue.owners_ended(); queue.original.owners_ended(); verify(sources)
    for key, expected in dict(full_original_input_admission_performed=True,
            all_scientific_failures_retained=True, queued_controller_changes_adopted=False,
            budget_only_development_followup=True, final_independent_population_policy_review_performed=False).items():
        if admission[key] is not expected: raise ValueError('exact development input scope required: '+key)
    prior = original_batch(admission['batch_result_sha256'], sources)
    if any(admission[k] != v for k, v in prior.items()):
        raise ValueError('original six-case review, raw case and model admission changed')
    queue.verify_bound(admission['extended_queue_completion'], sources)
    if admission['extended_queue_completion']['original_queue_admission']['adapter_batch_result_sha256'] != admission['batch_result_sha256']:
        raise ValueError('one original batch throughout the ordered completion chain required')
