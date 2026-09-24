"""Require the appended budget diagnostic before a final policy/budget review."""
from copy import deepcopy
import re

from scripts import independent_round_trip_extended_queue_completion_development as original
from scripts import await_go2_no_rgb_direct_extended_budget_maze02_native_v1 as budget
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE = 'scripts/independent_round_trip_budget_queue_completion_development.py'
TEST = 'lewm/tests/test_independent_round_trip_budget_queue_completion_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_budget_queue_completion_v1_2026-09-11.md'
LAUNCH_SHA = '0d767f341c8f0e28cd61107b198aa6eb76445f45cf13cc6e8602565c529fdf4f'
OWNER = dict(pid=2793505, created=1789096721.36, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', budget.SOURCE])


def prepared_sources():
    sources = original.prepared_sources()
    verify_artifacts(budget.OUTPUT, {'launch.json':LAUNCH_SHA})
    sources = merge_sources(sources, read_json(budget.OUTPUT, 'launch.json')['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL), sources)
    verify(sources)
    return sources


def owners_ended():
    if owner_live(OWNER): raise ValueError('original extended-budget waiter is still live')
    for key, owner, _, _, _ in budget.PREREQUISITES:
        if owner_live(owner): raise ValueError('original budget prerequisite is still live: '+key)


def expected_inputs(original_admission):
    prior = original_admission['original_queue_admission']
    ids = prior['ordered_waiter_result_sha256']; batch_sha = prior['adapter_batch_result_sha256']
    original.original.require_identities(ids, batch_sha)
    tracking_sha = original_admission['tracking_wait_result_sha256']
    if type(tracking_sha) is not str or re.fullmatch('[0-9a-f]{64}', tracking_sha) is None:
        raise ValueError('exact completed tracking identity required')
    return dict(batch=batch_sha, **ids, tracking=tracking_sha)


def require_waiter(result, launch, receipt, completion, inputs):
    if (result['status'] != 'NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or result['automatic_retry'] is not False
            or set(result['artifact_sha256']) != set(original.original.WAIT_FILES)
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA
            or launch['boot_id'] != BOOT or launch['waiter_pid'] != OWNER['pid']
            or launch['original_owners'] != {s[0]:s[1] for s in budget.PREREQUISITES}
            or launch['planned_case'] != list(budget.native.CASE)
            or launch['automatic_retry'] is not False or launch['source_changes_permitted'] is not False
            or type(launch['native_workers_while_waiting']) is not int
            or type(launch['native_workers_after_original_completion']) is not int
            or launch['native_workers_while_waiting'] != 0 or launch['native_workers_after_original_completion'] != 1):
        raise ValueError('exact original budget waiter, prerequisite roster and single native child required')
    if receipt != inputs or result['report'] != completion:
        raise ValueError('same five input identities and complete budget outcome required')


def admit(original_admission, budget_wait_result_sha256, *, sources, full=False):
    if type(full) is not bool: raise ValueError('explicit full-input-verification boolean required')
    if type(budget_wait_result_sha256) is not str or re.fullmatch('[0-9a-f]{64}', budget_wait_result_sha256) is None:
        raise ValueError('exact completed budget waiter SHA-256 required')
    inputs = expected_inputs(original_admission)
    owners_ended(); verify(sources)
    original.verify_bound(original_admission, sources)
    result, launch, wait_ids = completed(budget.OUTPUT, budget_wait_result_sha256, LAUNCH_SHA, sources)
    receipt = read_json(budget.OUTPUT, 'input_completion.json')
    completion = read_json(budget.OUTPUT, 'native_completion.json')
    require_waiter(result, launch, receipt, completion, inputs)
    reconstructed = budget.authenticate_completed(sources, inputs)
    if reconstructed != completion: raise ValueError('original budget completion and actual prefix finding must reconstruct')
    native = budget.native; sha = reconstructed['native_result_sha256']
    verify_artifacts(native.OUTPUT, {'result.json':sha})
    native_result = read_json(native.OUTPUT, 'result.json')
    _, native_launch, native_ids = completed(native.OUTPUT, sha, native_result['artifact_sha256']['launch.json'], sources)
    native.verify_inputs(native_launch)
    if full:
        again = native.inputs.admit(inputs['batch'], {k:inputs[k] for k in ('frontier', 'hold', 'contact')},
            inputs['tracking'], sources)
        if again != native_launch['input_admission']:
            raise ValueError('complete original budget input admission changed')
    verify_artifacts(budget.OUTPUT, wait_ids); verify_artifacts(native.OUTPUT, native_ids)
    owners_ended(); verify(sources)
    return dict(original_four_stage_admission=deepcopy(original_admission),
        budget_wait_result_sha256=budget_wait_result_sha256, budget_inputs=inputs,
        budget_completion=deepcopy(completion),
        artifact_bindings=[dict(root=str(budget.OUTPUT), artifact_sha256=wait_ids),
            dict(root=str(native.OUTPUT), artifact_sha256=native_ids)],
        complete_five_stage_queue_authenticated=True, original_queue_owners_ended=True,
        budget_native_input_admission_fully_reexecuted=full,
        original_four_stage_input_admissions_previously_fully_reexecuted=True,
        actual_budget_prefix_finding_reconstructed=True, native_case_raw_audits_reexecuted=False,
        all_scientific_failures_retained=True, final_policy_review_completed=False,
        independent_study_budget_selected=False, population_execution_permitted=False,
        new_layout_sensor_data_consumed=False)


def verify_bound(admission, sources):
    if admission['budget_native_input_admission_fully_reexecuted'] is not True:
        raise ValueError('original full budget input admission required')
    again = admit(admission['original_four_stage_admission'], admission['budget_wait_result_sha256'], sources=sources)
    again['budget_native_input_admission_fully_reexecuted'] = True
    if again != admission: raise ValueError('same complete five-stage evidence and unrevised study scope required')
