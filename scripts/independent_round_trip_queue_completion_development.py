"""Authenticate the original ordered native queue; never select a final policy.

No launcher, waiting loop, model construction or native scene. The final
population verifier must also authenticate its separate complete input admission
and the subsequent policy review before permitting independent-layout execution.
"""
from copy import deepcopy
import json
import re
import numpy as np

from scripts import await_go2_reached_frontier_maze03_native_v1 as frontier
from scripts import await_go2_hold_reorientation_maze02_native_v1 as hold
from scripts import await_go2_commitment_contact_anchored_maze02_native_v1 as contact
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.navigation_artifact_root_development import artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE = 'scripts/independent_round_trip_queue_completion_development.py'
TEST = 'lewm/tests/test_independent_round_trip_queue_completion_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_queue_completion_v1_2026-09-10.md'
RUNTIME_VERIFICATION = 'docs/go2_independent_round_trip_population_runtime_verification_2026-09-10.json'
RUNTIME_VERIFICATION_SHA = 'f5b25eb2c15f86e2e0948a62b39f4e859002f3c3d896ab773b14837b225ce8e7'
JOBS = (
    ('frontier', frontier, '68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74',
     'REACHED_FRONTIER_MAZE03_NATIVE_WAIT_COMPLETE', 2663938, 1789032169.81),
    ('hold', hold, 'e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6',
     'HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE', 2671835, 1789036731.41),
    ('contact', contact, '0c57eabf201e1eaba5f19e4f4fba55ed5948d490c8e5dc20d3bf4098f92e4cb2',
     'COMMITMENT_CONTACT_ANCHORED_MAZE02_NATIVE_WAIT_V1_COMPLETE', 2703476, 1789048977.88),
)
WAIT_FILES = ('launch.json', 'events.jsonl', 'input_completion.json', 'native_stdout.log', 'native_completion.json')
RAW_RESULTS = (
    ('prefix_result_sha256', '00579b00e70179bd1687a44ade0ed7282d8f6d3a742bbb7727a1a18733d14d25'),
    ('raw_prefix_wait_result_sha256', '62980125c12a4455f355f6c311ccf56d74d4f94bbc6ee0507b7908d53d6a135d'),
    ('raw_prefix_wait_result_sha256', 'cbcd31ee0ae1b1f598984cef16fa94524029b4e0e611bfde3dd684b0b029ad5a'),
)


def read(root, name):
    return json.loads(artifact_path(root, name).read_text())


def prepared_sources(seeds=()):
    verify({RUNTIME_VERIFICATION: RUNTIME_VERIFICATION_SHA})
    inherited = json.loads((ROOT/RUNTIME_VERIFICATION).read_text())['source_sha256']
    verify(inherited)
    for _, waiter, sha, _, _, _ in JOBS:
        verify_artifacts(waiter.OUTPUT, {'launch.json': sha})
        inherited = merge_sources(inherited, read(waiter.OUTPUT, 'launch.json')['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, RUNTIME_VERIFICATION,
        'docs/go2_independent_round_trip_population_runtime_result_2026-09-10.md', *seeds), inherited)
    verify(sources)
    return sources


def owners_ended():
    for key, waiter, _, _, pid, created in JOBS:
        owner = dict(pid=pid, created=created, command=[
            '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', waiter.SOURCE])
        if owner_live(owner):
            raise ValueError('original '+key+' native waiter is still live; preserve its queue ownership')


def require_identities(identities, batch_sha):
    if type(identities) is not dict or set(identities) != {job[0] for job in JOBS}:
        raise ValueError('all three exact original waiter result identities required')
    if any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
            for value in (*identities.values(), batch_sha)):
        raise ValueError('exact completed SHA-256 identities required')


def expected_receipt(index, identities, batch_sha):
    key, value = RAW_RESULTS[index]
    link, predecessor = (
        ('adapter_batch_result_sha256', batch_sha),
        ('frontier_wait_result_sha256', identities['frontier']),
        ('prior_native_wait_result_sha256', identities['hold']))[index]
    return {key:value, link:predecessor}


def require_waiter(index, result, launch, receipt, completion, identities, batch_sha):
    key, waiter, launch_sha, status, pid, _ = JOBS[index]
    if (result['status'] != status or result['automatic_retry'] is not False
            or set(result['artifact_sha256']) != set(WAIT_FILES)
            or result['artifact_sha256']['launch.json'] != launch_sha
            or launch['waiter_pid'] != pid or launch['planned_case'] != list(waiter.native.CASE)
            or launch['automatic_retry'] is not False or launch['source_changes_permitted'] is not False
            or type(launch['native_workers_while_waiting']) is not int
            or type(launch['native_workers_after_original_completion']) is not int
            or launch['native_workers_while_waiting'] != 0 or launch['native_workers_after_original_completion'] != 1):
        raise ValueError('same complete original '+key+' waiter and single child required')
    if receipt != expected_receipt(index, identities, batch_sha) or result['report'] != completion:
        raise ValueError('exact original batch-frontier-hold-contact chain and saved completion required')


def authenticate_job(index, identities, batch_sha, sources, *, full):
    key, waiter, launch_sha, _, _, _ = JOBS[index]
    result, launch, wait_ids = completed(waiter.OUTPUT, identities[key], launch_sha, sources)
    receipt = read(waiter.OUTPUT, 'input_completion.json')
    saved_completion = read(waiter.OUTPUT, 'native_completion.json')
    require_waiter(index, result, launch, receipt, saved_completion, identities, batch_sha)
    # This invokes the original native input, worker, raw-audit and prospective
    # physical-prefix receipt checks. It does not repeat the native raw audit.
    reconstructed = waiter.authenticate_completed(sources, receipt)
    if reconstructed != saved_completion:
        raise ValueError('original native completion verifier must reproduce the saved report')
    native = waiter.native; root = native.OUTPUT
    native_sha = reconstructed['native_result_sha256']
    verify_artifacts(root, {'result.json': native_sha})
    native_result = read(root, 'result.json')
    native_launch_sha = native_result['artifact_sha256']['launch.json']
    native_result, native_launch, native_ids = completed(root, native_sha, native_launch_sha, sources)
    native.verify_inputs(native_launch, full=full)
    record = native_result['conditions'][0]; name, layout = native.CASE[:2]
    collection = read(root, name+'/result.json')
    required = [name+'/'+n for n in native.artifacts(layout, collection)]
    required += [name+s for s in ('_worker_terminal.json', '_audit.json', '_prefix_comparison.json',
        '_readout.json', '_worker.log')]
    if (any(n not in native_ids for n in required)
            or record['collection'] != collection
            or record['worker_log_sha256'] != native_ids[name+'_worker.log']):
        raise ValueError('all original native collection and worker artifacts required')
    report = read(root, name+'_audit.json')
    with np.load(artifact_path(root, name+'/physics_trace.npz'), allow_pickle=False) as saved:
        readout = native.case_readout(report, collection, saved['physics_contact'])
    if readout != record['readout'] or readout != read(root, name+'_readout.json'):
        raise ValueError('complete native contact/timing readout must reconstruct')
    verify_artifacts(waiter.OUTPUT, wait_ids); verify_artifacts(root, native_ids)
    return dict(stage=key, waiter_result_sha256=identities[key], native_result_sha256=native_sha,
        case=name, collection=deepcopy(collection), readout=readout,
        measured_round_trip_successes=reconstructed['measured_round_trip_successes'],
        original_completion_verifier_reexecuted=True, scientific_success_required=False,
        artifact_bindings=[dict(root=str(waiter.OUTPUT), artifact_sha256=wait_ids),
            dict(root=str(root), artifact_sha256=native_ids)])


def admit(identities, *, adapter_batch_result_sha256, sources, full=False):
    if type(full) is not bool:
        raise ValueError('explicit full-input-verification boolean required')
    require_identities(identities, adapter_batch_result_sha256)
    owners_ended(); verify(sources)
    rows = [authenticate_job(i, identities, adapter_batch_result_sha256, sources, full=full)
        for i in range(len(JOBS))]
    owners_ended(); verify(sources)
    return dict(ordered_waiter_result_sha256={job[0]:identities[job[0]] for job in JOBS},
        adapter_batch_result_sha256=adapter_batch_result_sha256, completed=rows,
        complete_original_native_queue_authenticated=True, original_queue_owners_ended=True,
        original_native_input_admissions_fully_reexecuted=bool(full),
        native_case_raw_audits_reexecuted=False, all_scientific_failures_retained=True,
        final_policy_review_completed=False, population_execution_permitted=False,
        new_layout_sensor_data_consumed=False)


def verify_bound(admission, sources):
    if admission['original_native_input_admissions_fully_reexecuted'] is not True:
        raise ValueError('original full queue input admission required before bound verification')
    again = admit(admission['ordered_waiter_result_sha256'],
        adapter_batch_result_sha256=admission['adapter_batch_result_sha256'], sources=sources)
    again['original_native_input_admissions_fully_reexecuted'] = True
    if again != admission:
        raise ValueError('same complete original ordered queue evidence required')
