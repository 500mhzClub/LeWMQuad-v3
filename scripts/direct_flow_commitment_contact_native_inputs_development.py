"""Admit contact tracking recovery only after the complete existing native queue."""
from copy import deepcopy
import json
from pathlib import Path

from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import direct_flow_commitment_contact_native_prefix_development as prefix
from scripts import await_go2_sustained_hold_reorientation_maze02_native_v1 as sustained
from scripts import await_go2_commitment_contact_anchored_maze02_native_v1 as contact
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT

SOURCE = 'scripts/direct_flow_commitment_contact_native_inputs_development.py'
TEST = 'lewm/tests/test_direct_flow_commitment_contact_native_inputs_development.py'
PROTOCOL = 'docs/go2_direct_flow_commitment_contact_native_inputs_v1_2026-09-11.md'
PREPARATION = 'docs/go2_direct_flow_commitment_contact_native_prefix_preparation_2026-09-11.json'
PREPARATION_SHA = '8cee02c567bb7602f1cee31846e40bf9449d83c2b829a517b82c47cb4b3fa0be'
SUSTAINED_LAUNCH = '679d0519e3eb5ddacf5cc3708254de12a117551725c43c0136141e43ce64263e'
SUSTAINED_RAW_SHA = '977d00f6774d68c1b86d32d4e903650e13b89ced6cad34b43e0bd935b1a33e2d'
SUSTAINED_OWNER = dict(pid=2817601, created=1789117418.98, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', sustained.SOURCE])
CONTACT_SHA = '786107574dd5a4affc22d9a97ce16e9247214914ecf021df5079b7a1f2113865'
CONTACT_LAUNCH = '8d6a736c2f1f6de1f461fa6952341c1a7f3d321ccdd37881ac27a6da35191b8c'
CONTACT_WAIT_SHA = '8ce84ca1568df9848ca3f95ea02c400bde911162e50441cca70f5c913a4ea4f7'
CONTACT_WAIT_LAUNCH = '0c57eabf201e1eaba5f19e4f4fba55ed5948d490c8e5dc20d3bf4098f92e4cb2'
replay = prefix.replay


def prepared_sources(seeds=()):
    verify({PREPARATION:PREPARATION_SHA})
    prior = json.loads((ROOT/PREPARATION).read_text())
    if prior['status'] != 'DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_PREFIX_HELPER_PREPARED':
        raise ValueError('tested exact contact-plus-flow native prefix helper required')
    verify_artifacts(sustained.OUTPUT, {'launch.json':SUSTAINED_LAUNCH})
    sources = merge_sources(prior['source_sha256'], read_json(sustained.OUTPUT, 'launch.json')['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION, *seeds), sources)
    verify(sources); return sources


def owners_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('original contact replay and native queue boot required')
    if owner_live(prefix.OWNER): raise ValueError('original contact controller replay is still live')
    if owner_live(SUSTAINED_OWNER): raise ValueError('original sustained-turn native waiter is still live')
    sustained.native.inputs.owners_ended()


def raw_inputs(raw_sha, sources):
    owners_ended(); verify(sources)
    result, launch, ids = completed(replay.OUTPUT, raw_sha, prefix.LAUNCH_SHA, sources)
    report = prefix.admit_prefix(raw_sha, sources)
    actual = replay.verify_inputs(sources, launch['observer_artifact_sha256'])
    if actual != launch['input_artifact_sha256']:
        raise ValueError('same original contact worker inputs required')
    native_result, native_launch, native_ids = completed(contact.native.OUTPUT, CONTACT_SHA, CONTACT_LAUNCH, sources)
    waited, _, wait_ids = completed(contact.OUTPUT, CONTACT_WAIT_SHA, CONTACT_WAIT_LAUNCH, sources)
    name = contact.native.CASE[0]
    if (waited['report']['native_result_sha256'] != CONTACT_SHA
            or native_launch['model_state_sha256'] != replay.MODEL_SHA
            or report['model_state_sha256'] != replay.MODEL_SHA
            or native_ids[name+'_worker_terminal.json'] != replay.observer.probe.diagnosis.WORKER_SHA
            or native_result['prospective_prefix_result_sha256'] != native_launch['input_admission']['prefix_result_sha256']):
        raise ValueError('same completed original contact case, model and raw worker required')
    return dict(raw_prefix_result_sha256=raw_sha, raw_prefix_artifact_sha256=ids,
        prefix_report=deepcopy(report), original_raw_worker_artifact_sha256=actual,
        source_contact_result_sha256=CONTACT_SHA, source_contact_wait_result_sha256=CONTACT_WAIT_SHA,
        source_contact_artifact_sha256=native_ids, source_contact_wait_artifact_sha256=wait_ids,
        original_batch_result_sha256=native_launch['input_admission']['adapter_batch_result_sha256'],
        correction_admission=deepcopy(native_launch['input_admission']['correction_admission']))


def queue_completion(wait_sha, sources, *, full):
    if type(full) is not bool: raise ValueError('explicit full-input-admission boolean required')
    owners_ended()
    waited, launch, wait_ids = completed(sustained.OUTPUT, wait_sha, SUSTAINED_LAUNCH, sources)
    ids = read_json(sustained.OUTPUT, 'input_completion.json')
    receipt = read_json(sustained.OUTPUT, 'native_completion.json')
    required = {'launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json'}
    if (waited['status'] != 'SUSTAINED_HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or waited['automatic_retry'] is not False or set(waited['artifact_sha256']) != required
            or launch['waiter_pid'] != SUSTAINED_OWNER['pid'] or launch['boot_id'] != BOOT
            or launch['original_owners'] != {s[0]:s[1] for s in sustained.PREREQUISITES}
            or set(ids) != {'raw','budget'} or ids['raw'] != SUSTAINED_RAW_SHA
            or waited['report'] != receipt):
        raise ValueError('complete original sustained waiter and its fixed prerequisite identities required')
    actual = sustained.authenticate_completed(sources, ids)
    if actual != receipt: raise ValueError('complete original sustained native outcome must reconstruct')
    native_sha = actual['native_result_sha256']; root = sustained.native.OUTPUT
    verify_artifacts(root, {'result.json':native_sha})
    native_result = read_json(root, 'result.json')
    _, native_launch, native_ids = completed(root, native_sha, native_result['artifact_sha256']['launch.json'], sources)
    admission = native_launch['input_admission']
    if full and sustained.native.inputs.admit(ids['raw'], ids['budget'], sources) != admission:
        raise ValueError('complete original sustained and five-stage native input admission must reconstruct')
    owners_ended(); verify(sources)
    return dict(sustained_wait_result_sha256=wait_sha, sustained_native_result_sha256=native_sha,
        sustained_wait_artifact_sha256=wait_ids, sustained_native_artifact_sha256=native_ids,
        original_sustained_input_admission=deepcopy(admission), completion=actual,
        sustained_input_admission_fully_reexecuted=full, all_original_queue_failures_retained=True,
        original_queue_and_sustained_native_complete=True, independent_policy_review_performed=False)


def require_links(raw, queue):
    five = queue['original_sustained_input_admission']['five_stage_queue_completion']
    if (five['budget_inputs']['batch'] != raw['original_batch_result_sha256']
            or five['budget_inputs']['contact'] != CONTACT_WAIT_SHA
            or queue['sustained_input_admission_fully_reexecuted'] is not True
            or queue['all_original_queue_failures_retained'] is not True
            or queue['original_queue_and_sustained_native_complete'] is not True
            or five['all_scientific_failures_retained'] is not True
            or five['complete_five_stage_queue_authenticated'] is not True):
        raise ValueError('same original batch and contact completion throughout full native queue required')


def admit(raw_sha, sustained_wait_sha, sources):
    owners_ended(); raw = raw_inputs(raw_sha, sources)
    queue = queue_completion(sustained_wait_sha, sources, full=True); require_links(raw, queue)
    return raw | dict(sustained_wait_result_sha256=sustained_wait_sha, completed_native_queue=queue,
        complete_original_inputs_admitted=True, all_scientific_failures_retained=True,
        original_queue_and_sustained_complete_before_new_native=True,
        tracking_only_change_to_original_contact_controller=True, other_queued_policy_changes_adopted=False,
        independent_study_policy_selected=False)


def verify_bound(admission, sources):
    owners_ended()
    flags = dict(complete_original_inputs_admitted=True, all_scientific_failures_retained=True,
        original_queue_and_sustained_complete_before_new_native=True,
        tracking_only_change_to_original_contact_controller=True, other_queued_policy_changes_adopted=False,
        independent_study_policy_selected=False)
    if any(admission[k] is not v for k,v in flags.items()): raise ValueError('exact contact tracking-only scope required')
    raw = raw_inputs(admission['raw_prefix_result_sha256'], sources)
    if any(admission[k] != v for k,v in raw.items()): raise ValueError('same original raw, model and contact evidence required')
    saved = admission['completed_native_queue']; require_links(raw, saved)
    actual = queue_completion(admission['sustained_wait_result_sha256'], sources, full=False)
    actual['sustained_input_admission_fully_reexecuted'] = True
    if actual != saved: raise ValueError('same full original queue and completed sustained native outcome required')
    owners_ended(); verify(sources)
