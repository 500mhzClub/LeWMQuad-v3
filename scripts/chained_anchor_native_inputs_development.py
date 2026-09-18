"""Admit the original tracking case and the completed diagnostic queue."""
from contextlib import closing
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path

from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import chained_anchor_native_prefix_development as prefix
from scripts import await_go2_direct_flow_commitment_contact_maze02_native_v1 as flow
from scripts import await_go2_no_rgb_jepa_direct_flow_maze02_native_v1 as tracking
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT

SOURCE = 'scripts/chained_anchor_native_inputs_development.py'
TEST = 'lewm/tests/test_chained_anchor_native_inputs_development.py'
PROTOCOL = 'docs/go2_chained_anchor_maze02_pilot_v1_2026-09-11.md'
PREPARATION = 'docs/go2_chained_anchor_native_prefix_preparation_2026-09-11.json'
PREPARATION_SHA = '6acce51f98106224761b7987ccd3a58a2624a1975a501a6c01211b2ab182bc5e'
COMPLETION_SHA = '6b0c3298bb4df47aa59affc80d5dcb9236c1e4d9ae5cabd07550800c94d26425'
FLOW_LAUNCH = '3fc8e765b6edc16120b418e6dc8cedf1da47eb134d959f98450b2adeaa2c6c72'
FLOW_OWNER = dict(pid=2827789, created=1789121410.15, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', flow.SOURCE])
TRACKING_SHA = 'ce80ef3dffb6a249acfb3ea11dcbab459206ae6a088046b4a6a325d60e0c1bc1'
TRACKING_WAIT_SHA = '361ad8f517a8cef47384931156adb2341c749fb10d92fa06732ac9c67539fcee'
TRACKING_WAIT_LAUNCH = 'eb6c5b8b5e2f26c1ff70c03c17b3aa6761fdca5693c907b034df938447fbb68e'
TRACKING_OWNER = dict(pid=2753911, created=1789076672.54, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', tracking.SOURCE])
replay = prefix.replay


def prepared_sources(seeds=()):
    verify({PREPARATION: PREPARATION_SHA})
    prior = json.loads((ROOT/PREPARATION).read_text())
    if prior['status'] != 'CHAINED_ANCHOR_NATIVE_PREFIX_SOURCE_AND_ACTUAL_REPLAY_CHECKED':
        raise ValueError('actual authenticated chained-anchor prefix preparation required')
    verify_artifacts(flow.OUTPUT, {'launch.json': FLOW_LAUNCH})
    sources = merge_sources(prior['source_sha256'], read_json(flow.OUTPUT, 'launch.json')['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION, *seeds), sources)
    verify(sources)
    return sources


def owners_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('original replay and diagnostic queue boot required')
    for owner in (prefix.completed.OWNER, TRACKING_OWNER, FLOW_OWNER):
        if owner_live(owner): raise ValueError('original controller or diagnostic waiter is still live')
    flow.native.inputs.owners_ended()


def raw_inputs(completion_sha, sources):
    owners_ended()
    if completion_sha != COMPLETION_SHA: raise ValueError('exact completed controller verification required')
    verify({str(prefix.completed.OUTPUT): completion_sha})
    proof = json.loads(prefix.completed.OUTPUT.read_text())
    if (proof['status'] != 'CHAINED_ANCHOR_CONTROLLER_COMPLETION_VERIFIED'
            or proof['owner'] != prefix.completed.OWNER or proof['owner_ended'] is not True
            or proof['public_packets_reconstructed'] != prefix.FRAMES
            or proof['complete_saved_comparisons_reconstructed'] != prefix.FRAMES
            or proof['result_sha256'] != prefix.completed.RESULT_SHA
            or any(sources.get(n) != h for n,h in proof['source_sha256'].items())):
        raise ValueError('complete original controller and public packet verification required')
    result, launch, ids = completed(replay.OUTPUT, prefix.completed.RESULT_SHA, prefix.completed.LAUNCH_SHA, sources)
    if ids != proof['controller_artifact_sha256'] or not prefix.completed.equal(result['report'], proof['report']):
        raise ValueError('same exact controller report and artifacts required')
    native_result, native_launch, native_ids = completed(replay.native.OUTPUT, TRACKING_SHA,
        replay.observer.NATIVE_LAUNCH_SHA, sources)
    waited, _, wait_ids = completed(tracking.OUTPUT, TRACKING_WAIT_SHA, TRACKING_WAIT_LAUNCH, sources)
    if (waited['status'] != 'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or waited['report']['native_result_sha256'] != TRACKING_SHA
            or waited['report']['complete_native_worker_and_artifact_roster_verified'] is not True
            or native_result['status'] != 'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_PILOT_V1_COMPLETE'
            or native_launch['planned_case'] != list(replay.native.CASE)
            or native_launch['model_state_sha256'] != replay.MODEL_SHA
            or native_launch['input_admission']['full_original_model_and_episode_input_admission_reexecuted'] is not True):
        raise ValueError('same fully admitted and audited original tracking case and assigned model required')
    actual = replay.observer.admit_worker(sources)
    if actual != proof['worker_artifact_sha256'] or any(native_ids.get(n) != h for n,h in actual.items()):
        raise ValueError('original worker must be bound by completed tracking and controller evidence')
    prefix.boundary(proof['report'])
    verify_artifacts(replay.observer.OUTPUT, launch['observer_artifact_sha256'])
    prior = replay.native.OUTPUT/replay.native.CASE[0]
    with closing(prefix.read_rows(prior)) as old, closing(prefix.read_rows(replay.OUTPUT)) as saved, \
            closing(prefix.read_rows(replay.observer.OUTPUT)) as visual:
        prefix.reconstruct(proof['report'], islice(old, prefix.FRAMES), saved,
            read_json(prior, 'command_tape.json'), visual)
    return dict(controller_completion_sha256=completion_sha, controller_result_sha256=prefix.completed.RESULT_SHA,
        controller_artifact_sha256=ids, prefix_report=deepcopy(proof['report']),
        original_worker_artifact_sha256=actual, source_tracking_result_sha256=TRACKING_SHA,
        source_tracking_wait_result_sha256=TRACKING_WAIT_SHA, source_tracking_artifact_sha256=native_ids,
        source_tracking_wait_artifact_sha256=wait_ids,
        correction_admission=deepcopy(native_launch['input_admission']['correction_admission']),
        completed_original_training_admission_reused=True, training_data_replayed=False)


def queue_completion(wait_sha, sources, *, full=True):
    if type(full) is not bool: raise ValueError('explicit full completion check boolean required')
    owners_ended()
    waited, launch, wait_ids = completed(flow.OUTPUT, wait_sha, FLOW_LAUNCH, sources)
    ids = read_json(flow.OUTPUT, 'input_completion.json')
    receipt = read_json(flow.OUTPUT, 'native_completion.json')
    required = {'launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json'}
    if (waited['status'] != 'DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or waited['automatic_retry'] is not False or set(waited['artifact_sha256']) != required
            or launch['waiter_pid'] != FLOW_OWNER['pid'] or launch['boot_id'] != BOOT
            or launch['original_owners'] != {s[0]:s[1] for s in flow.PREREQUISITES}
            or set(ids) != {'raw','sustained'} or ids['raw'] != flow.RAW_RESULT_SHA
            or not prefix.completed.equal(waited['report'], receipt)):
        raise ValueError('complete original contact-plus-flow waiter and fixed prerequisite identities required')
    required_receipt = dict(complete_native_worker_and_artifact_roster_verified=True,
        actual_physical_prefix_reconstructed=True, scientific_success_required=False,
        final_independent_population_policy_review_performed=False)
    if any(receipt.get(k) is not v for k,v in required_receipt.items()):
        raise ValueError('complete queued physical-prefix and raw-artifact verification receipt required')
    actual = flow.authenticate_completed(sources, ids) if full else receipt
    if not prefix.completed.equal(actual, receipt): raise ValueError('original queued native outcome must reconstruct')
    native_sha = actual['native_result_sha256']
    verify_artifacts(flow.native.OUTPUT, {'result.json': native_sha})
    native_result = read_json(flow.native.OUTPUT, 'result.json')
    _, native_launch, native_ids = completed(flow.native.OUTPUT, native_sha,
        native_result['artifact_sha256']['launch.json'], sources)
    five = native_launch['input_admission']['completed_native_queue']['original_sustained_input_admission']['five_stage_queue_completion']
    if (five['budget_inputs']['tracking'] != TRACKING_WAIT_SHA
            or five['all_scientific_failures_retained'] is not True
            or five['complete_five_stage_queue_authenticated'] is not True):
        raise ValueError('same original tracking case throughout the completed diagnostic queue required')
    return dict(flow_wait_result_sha256=wait_sha, flow_native_result_sha256=native_sha,
        flow_wait_artifact_sha256=wait_ids, flow_native_artifact_sha256=native_ids,
        completion=actual, queued_completion_reexecuted=full,
        completed_through_contact_plus_flow=True, all_original_queue_failures_retained=True,
        independent_policy_review_performed=False)


def admit(completion_sha, flow_wait_sha, sources):
    owners_ended()
    raw = raw_inputs(completion_sha, sources)
    queue = queue_completion(flow_wait_sha, sources)
    return raw | dict(flow_wait_result_sha256=flow_wait_sha, completed_native_queue=queue,
        all_scientific_failures_retained=True, anchor_only_change_to_original_tracking_controller=True,
        other_queued_policy_changes_adopted=False, independent_study_policy_selected=False)


def verify_bound(admission, sources):
    owners_ended()
    raw = raw_inputs(admission['controller_completion_sha256'], sources)
    saved = admission['completed_native_queue']
    if saved['queued_completion_reexecuted'] is not True:
        raise ValueError('full original queued completion must have been executed at admission')
    queue = queue_completion(admission['flow_wait_result_sha256'], sources, full=False)
    # The original full admission is immutable; bound checks authenticate its
    # exact source, result, worker and artifact receipts without executing it again.
    queue['queued_completion_reexecuted'] = True
    expected = raw | dict(flow_wait_result_sha256=admission['flow_wait_result_sha256'], completed_native_queue=queue,
        all_scientific_failures_retained=True, anchor_only_change_to_original_tracking_controller=True,
        other_queued_policy_changes_adopted=False, independent_study_policy_selected=False)
    if not prefix.completed.equal(expected, admission):
        raise ValueError('same original tracking model, prefix and completed diagnostic queue required')
    owners_ended()
    verify(sources)
