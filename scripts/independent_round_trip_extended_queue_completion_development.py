"""Require the original queue and appended tracking test before policy review.

This checks completion evidence only. It neither selects a policy nor launches
the independent population, and it never requires scientific success.
"""
from copy import deepcopy
import re

from scripts import independent_round_trip_queue_completion_development as original
from scripts import await_go2_no_rgb_jepa_direct_flow_maze02_native_v1 as tracking
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE = 'scripts/independent_round_trip_extended_queue_completion_development.py'
TEST = 'lewm/tests/test_independent_round_trip_extended_queue_completion_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_extended_queue_completion_v1_2026-09-10.md'
LAUNCH_SHA = 'eb6c5b8b5e2f26c1ff70c03c17b3aa6761fdca5693c907b034df938447fbb68e'
PREFIX_SHA = '152da30ba5c142d8150dacd5865278f81d5db46c7603bd637eb8ec2eee292fdf'
OWNER = dict(pid=2753911, created=1789076672.54, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', tracking.SOURCE])


def prepared_sources():
    sources = original.prepared_sources()
    verify_artifacts(tracking.OUTPUT, {'launch.json': LAUNCH_SHA})
    sources = merge_sources(sources, read_json(tracking.OUTPUT, 'launch.json')['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL), sources)
    verify(sources)
    return sources


def owners_ended():
    # Check the appended owner first, before any large predecessor admission.
    if owner_live(OWNER):
        raise ValueError('original tracking-recovery waiter is still live')
    for _, owner, _, _, _ in tracking.PREREQUISITES:
        if owner_live(owner):
            raise ValueError('original tracking-recovery prerequisite owner is still live')


def expected_inputs(original_admission):
    identities = original_admission['ordered_waiter_result_sha256']
    batch_sha = original_admission['adapter_batch_result_sha256']
    original.require_identities(identities, batch_sha)
    return dict(prefix=PREFIX_SHA, batch=batch_sha, **identities)


def require_waiter(result, launch, receipt, completion, inputs):
    if (result['status'] != 'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or result['automatic_retry'] is not False
            or set(result['artifact_sha256']) != set(original.WAIT_FILES)
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA
            or launch['boot_id'] != BOOT or launch['waiter_pid'] != OWNER['pid']
            or launch['original_owners'] != {s[0]: s[1] for s in tracking.PREREQUISITES}
            or launch['planned_case'] != list(tracking.native.CASE)
            or launch['reviewed_prefix_artifact_sha256'] != tracking.PREFIX_BINDINGS
            or launch['automatic_retry'] is not False or launch['source_changes_permitted'] is not False
            or type(launch['native_workers_while_waiting']) is not int
            or type(launch['native_workers_after_original_completion']) is not int
            or launch['native_workers_while_waiting'] != 0
            or launch['native_workers_after_original_completion'] != 1):
        raise ValueError('exact original tracking waiter and single native child required')
    if receipt != inputs or result['report'] != completion:
        raise ValueError('same completed replay, batch, ordered queue and native completion required')


def admit(original_admission, tracking_wait_result_sha256, *, sources, full=False):
    if type(full) is not bool:
        raise ValueError('explicit full-input-verification boolean required')
    if (type(tracking_wait_result_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', tracking_wait_result_sha256) is None):
        raise ValueError('exact completed tracking waiter SHA-256 required')
    inputs = expected_inputs(original_admission)
    owners_ended(); verify(sources)
    # Requires the original three-stage admission to have run with full=True.
    original.verify_bound(original_admission, sources)
    result, launch, wait_ids = completed(tracking.OUTPUT, tracking_wait_result_sha256, LAUNCH_SHA, sources)
    receipt = read_json(tracking.OUTPUT, 'input_completion.json')
    completion = read_json(tracking.OUTPUT, 'native_completion.json')
    require_waiter(result, launch, receipt, completion, inputs)
    reconstructed = tracking.authenticate_completed(sources, inputs)
    if reconstructed != completion:
        raise ValueError('original tracking native completion must reconstruct exactly')
    native = tracking.native
    native_sha = reconstructed['native_result_sha256']
    verify_artifacts(native.OUTPUT, {'result.json': native_sha})
    native_result = read_json(native.OUTPUT, 'result.json')
    _, native_launch, native_ids = completed(native.OUTPUT, native_sha,
        native_result['artifact_sha256']['launch.json'], sources)
    native.verify_inputs(native_launch, full=full)
    # The existing completion verifier reconstructs worker, raw audit, actual
    # physical prefix and physics-contact readout receipts; no raw audit rerun.
    verify_artifacts(tracking.OUTPUT, wait_ids); verify_artifacts(native.OUTPUT, native_ids)
    owners_ended(); verify(sources)
    return dict(original_queue_admission=deepcopy(original_admission),
        tracking_wait_result_sha256=tracking_wait_result_sha256, tracking_inputs=inputs,
        tracking_completion=deepcopy(completion),
        artifact_bindings=[dict(root=str(tracking.OUTPUT), artifact_sha256=wait_ids),
            dict(root=str(native.OUTPUT), artifact_sha256=native_ids)],
        complete_extended_queue_authenticated=True, original_queue_owners_ended=True,
        tracking_native_input_admission_fully_reexecuted=full,
        original_native_input_admissions_previously_fully_reexecuted=True,
        native_case_raw_audits_reexecuted=False, all_scientific_failures_retained=True,
        final_policy_review_completed=False, population_execution_permitted=False,
        new_layout_sensor_data_consumed=False)


def verify_bound(admission, sources):
    if admission['tracking_native_input_admission_fully_reexecuted'] is not True:
        raise ValueError('original full tracking input admission required')
    again = admit(admission['original_queue_admission'], admission['tracking_wait_result_sha256'],
        sources=sources)
    again['tracking_native_input_admission_fully_reexecuted'] = True
    if again != admission:
        raise ValueError('same complete extended queue evidence required')
