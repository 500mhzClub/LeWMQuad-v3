"""Join the three later native outcomes to the original five-diagnostic evidence.

This read-only admission does not select a policy or launch a population.
Original waiter/native authentication retains failures and reconstructs raw
physical prefixes. All original owners must end before outcome reads begin.
"""
from copy import deepcopy
from dataclasses import dataclass
import re

from scripts import independent_round_trip_budget_queue_completion_development as previous
from scripts import independent_round_trip_final_admission_development as final
from scripts import await_go2_sustained_hold_reorientation_maze02_native_v1 as sustained
from scripts import await_go2_direct_flow_commitment_contact_maze02_native_v1 as flow
from scripts import await_go2_chained_anchor_maze02_native_v1 as anchor
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE='scripts/independent_round_trip_later_diagnostics_evidence_development.py'
TEST='lewm/tests/test_independent_round_trip_later_diagnostics_evidence_development.py'
PROTOCOL='docs/go2_independent_round_trip_later_diagnostics_evidence_v1_2026-09-11.md'
WAIT_FILES={'launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json'}


@dataclass(frozen=True)
class Stage:
    name: str
    waiter: object
    owner: dict
    launch_sha256: str
    status: str
    predecessor_key: str
    raw_sha256: str


STAGES=(
    Stage('sustained_turn',sustained,flow.native.inputs.SUSTAINED_OWNER,
        flow.native.inputs.SUSTAINED_LAUNCH,'SUSTAINED_HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE',
        'budget',flow.native.inputs.SUSTAINED_RAW_SHA),
    Stage('contact_flow',flow,anchor.native.inputs.FLOW_OWNER,
        anchor.native.inputs.FLOW_LAUNCH,'DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_NATIVE_WAIT_V1_COMPLETE',
        'sustained',flow.RAW_RESULT_SHA),
    Stage('chained_anchor',anchor,dict(pid=2845479,created=1789129072.88,command=[
        '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python','-B',anchor.SOURCE]),
        'cf6703e83197d6c75df35b2b53834a47a53a293a06c64737ce94ade2ac0b87c1',
        'CHAINED_ANCHOR_MAZE02_NATIVE_WAIT_V1_COMPLETE','flow',anchor.RAW_RESULT_SHA),
)


def sha256(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('exact completed SHA-256 identity required')
    return value


def require_launch(stage,launch):
    waiter=stage.waiter
    if (launch['boot_id']!=BOOT or launch['waiter_pid']!=stage.owner['pid']
            or launch['original_owners']!={s[0]:s[1] for s in waiter.PREREQUISITES}
            or launch['prerequisite_launch_sha256']!={s[0]:s[3] for s in waiter.PREREQUISITES}
            or launch['planned_case']!=list(waiter.native.CASE)
            or launch['maximum_wait_s']!=waiter.WAIT_SECONDS
            or launch['automatic_retry'] is not False or launch['source_changes_permitted'] is not False
            or type(launch['native_workers_while_waiting']) is not int
            or launch['native_workers_while_waiting']!=0
            or type(launch['native_workers_after_original_completion']) is not int
            or launch['native_workers_after_original_completion']!=1):
        raise ValueError('exact original later-diagnostic waiter and single-child definition required')


def prepared_sources():
    sources=merge_sources(previous.prepared_sources(),final.prepared_sources())
    for stage in STAGES:
        verify_artifacts(stage.waiter.OUTPUT,{'launch.json':stage.launch_sha256})
        launch=read_json(stage.waiter.OUTPUT,'launch.json');require_launch(stage,launch)
        sources=merge_sources(sources,launch['source_sha256'])
    sources=discover_sources((SOURCE,TEST,PROTOCOL),sources);verify(sources)
    return sources


def owners_ended():
    # Reject all live later waiters before the older gate or artifact reads.
    for stage in STAGES:
        if owner_live(stage.owner):raise ValueError('later diagnostic still live: '+stage.name)
        for key,owner,_,_,_ in stage.waiter.PREREQUISITES:
            if owner_live(owner):raise ValueError('later diagnostic prerequisite still live: '+stage.name+'/'+key)
    previous.owners_ended()


def expected_inputs(stage,predecessor_sha):
    return {'raw':sha256(stage.raw_sha256),stage.predecessor_key:sha256(predecessor_sha)}


def require_waiter(stage,result,launch,receipt,completion,inputs):
    require_launch(stage,launch)
    if (result['status']!=stage.status or result['automatic_retry'] is not False
            or set(result['artifact_sha256'])!=WAIT_FILES
            or result['artifact_sha256']['launch.json']!=stage.launch_sha256
            or receipt!=inputs or result['report']!=completion):
        raise ValueError('complete original waiter, exact predecessor link and outcome required')
    for key in ('navigation_qualified','real_time_qualified','hardware_qualified','goal_achieved'):
        if result[key] is not False:raise ValueError('unqualified development outcome scope required')
    if (set(completion)!={'native_result_sha256','measured_round_trip_successes',
            'complete_native_worker_and_artifact_roster_verified','actual_physical_prefix_reconstructed',
            'scientific_success_required','final_independent_population_policy_review_performed'}
            or completion['complete_native_worker_and_artifact_roster_verified'] is not True
            or completion['actual_physical_prefix_reconstructed'] is not True
            or completion['scientific_success_required'] is not False
            or completion['final_independent_population_policy_review_performed'] is not False
            or type(completion['measured_round_trip_successes']) is not int
            or completion['measured_round_trip_successes'] not in (0,1)):
        raise ValueError('complete actual native verification with negative outcomes retained required')
    sha256(completion['native_result_sha256'])


def admit_stage(stage,wait_sha,predecessor_sha,sources):
    inputs=expected_inputs(stage,predecessor_sha);waiter=stage.waiter
    result,launch,wait_ids=completed(waiter.OUTPUT,sha256(wait_sha),stage.launch_sha256,sources)
    receipt=read_json(waiter.OUTPUT,'input_completion.json')
    completion=read_json(waiter.OUTPUT,'native_completion.json')
    require_waiter(stage,result,launch,receipt,completion,inputs)
    # Reexecute each original native authenticator, not just its saved flags.
    actual=waiter.authenticate_completed(sources,inputs)
    if actual!=completion:raise ValueError('whole original native completion must reconstruct')
    native=waiter.native;native_sha=actual['native_result_sha256']
    verify_artifacts(native.OUTPUT,{'result.json':native_sha})
    native_result=read_json(native.OUTPUT,'result.json')
    _,_,native_ids=completed(native.OUTPUT,native_sha,native_result['artifact_sha256']['launch.json'],sources)
    name=native.CASE[0]
    record=read_json(native.OUTPUT,name+'_worker_terminal.json')
    readout=read_json(native.OUTPUT,name+'_readout.json')
    prefix=read_json(native.OUTPUT,name+'_prefix_comparison.json')
    if (native_result['conditions']!=[record] or readout!=record['readout']
            or prefix!=record['prefix_comparison']
            or type(record['verified_round_trip']) is not bool
            or actual['measured_round_trip_successes']!=int(record['verified_round_trip'])):
        raise ValueError('same actual single worker, readout, prefix and outcome accounting required')
    verify_artifacts(waiter.OUTPUT,wait_ids);verify_artifacts(native.OUTPUT,native_ids)
    return dict(stage=stage.name,waiter_result_sha256=wait_sha,native_result_sha256=native_sha,
        prerequisite_result_sha256=inputs,case=name,completion=deepcopy(actual),
        collection=deepcopy(record['collection']),readout=deepcopy(readout),prefix_comparison=deepcopy(prefix),
        measured_round_trip_successes=actual['measured_round_trip_successes'],
        artifact_bindings=[dict(root=str(waiter.OUTPUT),artifact_sha256=wait_ids),
            dict(root=str(native.OUTPUT),artifact_sha256=native_ids)])


def admit(five_stage_admission,waiter_ids,*,sources):
    if type(waiter_ids) is not dict or set(waiter_ids)!={s.name for s in STAGES}:
        raise ValueError('all three later diagnostic result identities required')
    for value in waiter_ids.values():sha256(value)
    owners_ended();verify(sources)
    previous.verify_bound(five_stage_admission,sources)
    preceding=sha256(five_stage_admission['budget_wait_result_sha256']);rows=[]
    for stage in STAGES:
        rows.append(admit_stage(stage,waiter_ids[stage.name],preceding,sources))
        preceding=waiter_ids[stage.name]
    owners_ended();verify(sources)
    return dict(schema='independent_round_trip_eight_diagnostic_evidence.v1',
        original_five_stage_admission=deepcopy(five_stage_admission),
        later_waiter_result_sha256=deepcopy(waiter_ids),later_diagnostics=rows,
        all_eight_diagnostics_authenticated=True,all_original_owners_ended=True,
        all_scientific_failures_retained=True,later_native_authenticators_reexecuted=True,
        later_native_case_raw_audits_reexecuted=False,final_policy_review_completed=False,
        population_definition_selected=False,population_execution_permitted=False,
        independent_layout_sensor_data_consumed=False,goal_achieved=False)


def verify_bound(admission,sources):
    again=admit(admission['original_five_stage_admission'],admission['later_waiter_result_sha256'],sources=sources)
    if again!=admission:raise ValueError('whole eight-diagnostic evidence and scope must reconstruct')


def review_evidence(input_admission,admission,*,sources):
    verify_bound(admission,sources)
    five=admission['original_five_stage_admission']
    original=final.review_evidence(input_admission,five)
    return dict(original_five_diagnostic_evidence=original,
        later_diagnostics=deepcopy(admission['later_diagnostics']),
        diagnostic_count=8,final_policy_review_completed=False,
        population_definition_selected=False,population_execution_permitted=False)
