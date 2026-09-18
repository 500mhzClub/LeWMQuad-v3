"""Admit the completed sustained raw prefix after the original five-stage queue."""
from copy import deepcopy
import json
from pathlib import Path
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import independent_round_trip_budget_queue_completion_development as queue
from scripts import replay_go2_sustained_hold_reorientation_maze02_prefix_v1 as replay
from scripts import sustained_hold_reorientation_native_prefix_development as prefix
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live,BOOT

SOURCE='scripts/sustained_hold_reorientation_native_inputs_development.py'
TEST='lewm/tests/test_sustained_hold_reorientation_native_inputs_development.py'
PROTOCOL='docs/go2_sustained_hold_reorientation_native_inputs_v1_2026-09-11.md'
PREPARATION='docs/go2_sustained_hold_reorientation_native_prefix_preparation_2026-09-11.json'
PREPARATION_SHA='a50a56836c91d882408a3ddb8f64b17bae718ba7052438e46dea9eac02b91d84'
RAW_OWNER=dict(pid=2813368,created=1789115323.91,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python','-B',replay.SOURCE])


def prepared_sources(seeds=()):
    verify({PREPARATION:PREPARATION_SHA})
    prior=json.loads((ROOT/PREPARATION).read_text())
    if prior['status']!='SUSTAINED_RAW_COMPLETION_AND_NATIVE_PREFIX_HELPER_PREPARED':
        raise ValueError('tested fixed native-prefix helper required')
    sources=merge_sources(prior['source_sha256'],queue.prepared_sources())
    sources=discover_sources((SOURCE,TEST,PROTOCOL,PREPARATION,*seeds),sources)
    verify(sources);return sources


def owners_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=BOOT:
        raise ValueError('original raw and queued execution boot required')
    if owner_live(RAW_OWNER):raise ValueError('original sustained raw replay is still live')
    queue.owners_ended()


def raw_inputs(raw_sha,sources):
    owners_ended();verify(sources)
    result,launch,ids=completed(replay.OUTPUT,raw_sha,prefix.LAUNCH_SHA,sources)
    actual=replay.admit(sources)
    if actual!=launch['input_admission']:
        raise ValueError('complete original raw input admission must reconstruct')
    report=prefix.admit_prefix(replay.OUTPUT,result)
    old=replay.saved.prior
    native_result,native_launch,_=completed(replay.original.OUTPUT,old.NATIVE_SHA,old.NATIVE_LAUNCH,sources)
    if (native_launch['model_state_sha256']!=replay.MODEL_SHA
            or report['model_state_sha256']!=replay.MODEL_SHA
            or native_result['prospective_prefix_result_sha256']!=native_launch['input_admission']['prefix_result_sha256']):
        raise ValueError('same original full-JEPA model and native predecessor required')
    return dict(raw_prefix_result_sha256=raw_sha,raw_prefix_artifact_sha256=ids,
        prefix_report=deepcopy(report),original_raw_input_admission=actual,
        source_native_result_sha256=old.NATIVE_SHA,source_native_launch_sha256=old.NATIVE_LAUNCH,
        original_batch_result_sha256=native_launch['input_admission']['adapter_batch_result_sha256'],
        correction_admission=deepcopy(native_launch['input_admission']['correction_admission']))


def require_links(raw,five):
    if (five['budget_inputs']['batch']!=raw['original_batch_result_sha256']
            or five['budget_inputs']['hold']!=replay.saved.prior.WAIT_SHA
            or five['all_scientific_failures_retained'] is not True
            or five['complete_five_stage_queue_authenticated'] is not True
            or five['budget_native_input_admission_fully_reexecuted'] is not True):
        raise ValueError('same original batch and hold run throughout complete five-stage queue required')


def admit(raw_sha,budget_wait_sha,sources):
    owners_ended()
    raw=raw_inputs(raw_sha,sources)
    waited,_,_=completed(queue.budget.OUTPUT,budget_wait_sha,queue.LAUNCH_SHA,sources)
    native_sha=waited['report']['native_result_sha256'];native=queue.budget.native
    verify_artifacts(native.OUTPUT,{'result.json':native_sha})
    native_result=read_json(native.OUTPUT,'result.json')
    _,native_launch,_=completed(native.OUTPUT,native_sha,native_result['artifact_sha256']['launch.json'],sources)
    four=native_launch['input_admission']['extended_queue_completion']
    five=queue.admit(four,budget_wait_sha,sources=sources,full=True)
    require_links(raw,five);owners_ended();verify(sources)
    return raw | dict(budget_wait_result_sha256=budget_wait_sha,five_stage_queue_completion=five,
        complete_original_inputs_admitted=True,all_scientific_failures_retained=True,
        original_queue_complete_before_new_native=True,queued_controller_changes_adopted=False,
        sustained_turn_only_development_followup=True,independent_study_policy_selected=False)


def verify_bound(admission,sources):
    owners_ended()
    for k,v in dict(complete_original_inputs_admitted=True,all_scientific_failures_retained=True,
            original_queue_complete_before_new_native=True,queued_controller_changes_adopted=False,
            sustained_turn_only_development_followup=True,independent_study_policy_selected=False).items():
        if admission[k] is not v:raise ValueError('exact sustained development scope required: '+k)
    raw=raw_inputs(admission['raw_prefix_result_sha256'],sources)
    if any(admission[k]!=v for k,v in raw.items()):
        raise ValueError('same original raw prefix, native case and assigned model required')
    five=admission['five_stage_queue_completion']
    if five['budget_wait_result_sha256']!=admission['budget_wait_result_sha256']:
        raise ValueError('same final original waiter identity required')
    queue.verify_bound(five,sources);require_links(raw,five)
    owners_ended();verify(sources)
