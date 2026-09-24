"""Authenticate study inputs; final policy review and execution remain separate.

There is no launcher or scene construction here. Full admission requires the
original six-case batch to complete, including its negative outcomes.
"""
from dataclasses import asdict
import json
from lewm import independent_round_trip_comparison_study_development as study
from lewm.independent_round_trip_layouts_development import validate_inventory
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import build_go2_independent_round_trip_layout_inventory_v1 as inventory
from scripts import replay_go2_independent_adapter_factory_startup_v1 as factory
from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as batch
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.maze_decision_stream_development import read_rows, NAME
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/independent_round_trip_population_inputs_development.py'
TEST = 'lewm/tests/test_independent_round_trip_population_inputs_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_population_inputs_v1_2026-09-10.md'
INVENTORY_LAUNCH = '222b66212b7beb7405a175c7c156b94d188aafe58beb01aa02323c9b8f76cda3'
FACTORY_RESULT = 'cd0af02bdbe978e714e55c1c0253f1fa01df88c4554eeea887deae84e40ae6db'
FACTORY_LAUNCH = '2cbd89c0be82ddb26957393c9c299c62b1442c59dbfb8284fdabf1ae62aac9d6'
BATCH_LAUNCH = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
ASSIGNMENT = 'docs/go2_independent_round_trip_comparison_assignments_v1_2026-09-10.json'
DOCUMENTS = {
    ASSIGNMENT: '39754702e7c709e9e2e9e47d2b01ea748d8f94c40a3041f53e2cead6b1678bda',
    'docs/go2_independent_round_trip_comparison_assignments_verification_2026-09-10.json':
        '066774eacc0c9587a888c58248eba0b0c42f28e402bff510ae998cc202c67b85',
    'docs/go2_independent_round_trip_multiarm_integration_verification_2026-09-10.json':
        '3e825e87a43b7958be94a57db64a02cf8a354185d3c369288905eb35304bc76a',
    'docs/go2_independent_round_trip_population_readout_verification_2026-09-10.json':
        '7d264f828f5be0a6f40b3e7cf2d500b17a48e5fad3e23cb1162fa51297b0d08a',
}
STATIC = ((inventory.OUTPUT, study.INVENTORY_RESULT_SHA256, INVENTORY_LAUNCH),
    (factory.OUTPUT, FACTORY_RESULT, FACTORY_LAUNCH))


def prepared_sources(seeds=()):
    inherited = {}
    for root, result_sha, launch_sha in STATIC:
        verify_artifacts(root, {'result.json': result_sha, 'launch.json': launch_sha})
        inherited = merge_sources(inherited, read_json(root, 'result.json')['source_sha256'])
    verify_artifacts(batch.OUTPUT, {'launch.json': BATCH_LAUNCH})
    inherited = merge_sources(inherited, read_json(batch.OUTPUT, 'launch.json')['source_sha256'])
    verify(DOCUMENTS)
    for name in DOCUMENTS:
        if name != ASSIGNMENT:
            inherited = merge_sources(inherited, json.loads((ROOT/name).read_text())['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, *DOCUMENTS, *seeds), inherited)
    verify(sources)
    return sources


def require_factory_reports(result, launch, streams):
    if (result['status'] != 'INDEPENDENT_ADAPTER_FACTORY_STARTUP_V1_COMPLETE'
            or result['native_execution'] is not False or result['model_training'] is not False
            or result['new_layout_sensor_data_consumed'] is not False
            or result['new_layout_navigation_execution'] is not False
            or launch['planned_cases'] != [asdict(c) for c in study.CASES[:4]]
            or len(result['reports']) != 4 or len(streams) != 4):
        raise ValueError('complete four-arm old-packet adapter check required')
    for case, report, rows in zip(study.CASES[:4], result['reports'], streams, strict=True):
        arm = study.require_case(case); learned = arm.name != 'reactive'
        expected = dict(arm=arm.name, assigned_case=case.name,
            original_public_packet_source=factory.INPUT_CASE, controller_class=arm.implementation,
            frames=4, model_state_sha256=arm.model_state_sha256, model_state_unchanged=True,
            all_original_expanded_forward_outputs_exact=learned, actual_planner_forecast_exact=learned,
            reactive_path_unchanged=not learned, no_packet_after_new_planning_command_consumed=True,
            command_executed=False, study_public_mission_instantiated=True,
            independent_layout_sensor_data_consumed=False, independent_layout_navigation_execution=False,
            navigation_verified=False)
        if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
            raise ValueError('exact preassigned factory treatment and evidence scope required')
        if [r['tick'] for r in rows] != list(range(4)):
            raise ValueError('exact four saved observations per factory arm required')
        for frame, row in enumerate(rows):
            old, new = row['old_factory_decision'], row['decision']
            if (row['public_input_arrays_unchanged'] is not True
                    or row['complete_retained_contact_state_equal'] is not True
                    or row['same_assigned_study_public_mission'] is not True
                    or row['original_native_decision_reconstruction_claimed'] is not False
                    or new['terminal'] is not None):
                raise ValueError('complete successful factory packet/state receipts required')
            if (frame < 3 and (old != new or new['requested_command'] != [0., 0., 0.])) or (not learned and old != new):
                raise ValueError('three unchanged zero warmups and complete reactive equality required')
        old, new = rows[-1]['old_factory_decision'], rows[-1]['decision']
        boundary = dict(original_factory_terminal=old['terminal'], adapter_factory_terminal=new['terminal'],
            original_factory_command=old['requested_command'], adapter_factory_command=new['requested_command'],
            selected_action=None if new['new_selection'] is None else new['new_selection']['action'])
        if report['boundary'] != boundary or (learned and old['terminal'] != 'SENSOR_OR_MODEL_FAILURE'):
            raise ValueError('exact reported original-wrapper boundary required')


def admit_static(sources):
    verify(sources); verify(DOCUMENTS)
    groups = []; results = []; launches = []
    for root, sha, launch_sha in STATIC:
        result, launch, ids = completed(root, sha, launch_sha, sources)
        groups.append(dict(root=str(root), artifact_sha256=ids))
        results.append(result); launches.append(launch)
    inv, startup = results
    if (inv['status'] != 'INDEPENDENT_ROUND_TRIP_LAYOUT_INVENTORY_COMPLETE'
            or type(inv['layouts']) is not int or inv['layouts'] != 8
            or inv['native_execution'] is not False or inv['model_loaded'] is not False
            or inv['exact_abstract_topology_and_grid_disjointness'] is not True):
        raise ValueError('complete source-only eight-layout inventory required')
    validate_inventory(read_json(inventory.OUTPUT, 'inventory.json'))
    roster = json.loads((ROOT/ASSIGNMENT).read_text())
    if roster != study.manifest():
        raise ValueError('complete exact 32-case assignment manifest required')
    streams = []
    for case in study.CASES[:4]:
        if case.arm_name+'/'+NAME not in startup['artifact_sha256']:
            raise ValueError('each full factory decision stream must be bound')
        streams.append(list(read_rows(factory.OUTPUT/case.arm_name)))
    require_factory_reports(startup, launches[1], streams)
    old_root = factory.original.INPUT
    old, old_launch, old_ids = completed(old_root, factory.evidence.ORIGINAL_SHA,
        factory.evidence.ORIGINAL_LAUNCH, sources)
    if (launches[1]['original_result_sha256'] != factory.evidence.ORIGINAL_SHA
            or launches[1]['original_artifact_sha256'] != old_ids):
        raise ValueError('complete original failed-cohort factory inputs required')
    groups.append(dict(root=str(old_root), artifact_sha256=old_ids))
    correction = old_launch['input_admission']['correction_admission']
    if correction['correction_result_sha256'] != study.CORRECTION_RESULT_SHA256:
        raise ValueError('same original assigned corrected-model cohort required')
    verify(sources)
    return dict(inventory_result_sha256=study.INVENTORY_RESULT_SHA256, factory_result_sha256=FACTORY_RESULT,
        assignment_manifest_sha256=DOCUMENTS[ASSIGNMENT], document_sha256=dict(DOCUMENTS),
        ordered_cases=roster['ordered_cases'], planned_episodes=32, independent_layout_units=8,
        factory_correction_admission=correction, completed_artifact_bindings=groups,
        static_inputs_verified=True, full_training_and_coefficient_verifiers_rerun=False,
        completed_six_case_admission_performed=False, final_policy_review_completed=False,
        population_execution_permitted=False, new_layout_sensor_data_consumed=False)


def require_batch(result, launch):
    if (result['status'] != 'ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE'
            or len(result['conditions']) != len(batch.CASES)
            or result['all_fixed_cases_executed'] is not True
            or result['planner_interface_adapter_enabled'] is not True
            or result['frontier_transition_enabled'] is not False):
        raise ValueError('complete exact six-case adapter cohort required')
    ids = result['artifact_sha256']; reports = []
    for case, record in zip(batch.CASES, result['conditions'], strict=True):
        name = case[0]
        required = [name+s for s in ('_worker_terminal.json', '_worker.log', '_audit.json',
            '_startup_comparison.json', '_readout.json', '_parent_completion.json')]
        required += [name+'/'+n for n in batch.artifacts(case[1], record['collection'])]
        if any(n not in ids for n in required) or any(ids.get(n) != h for n, h in record['artifact_sha256'].items()):
            raise ValueError('every original case artifact and completion receipt required')
        report = read_json(batch.OUTPUT, name+'_audit.json')
        batch.require_case(case, record, report)
        if (record != read_json(batch.OUTPUT, name+'_worker_terminal.json')
                or record['collection'] != read_json(batch.OUTPUT, name+'/result.json')
                or record['startup_comparison'] != read_json(batch.OUTPUT, name+'_startup_comparison.json')
                or record['readout'] != read_json(batch.OUTPUT, name+'_readout.json')
                or record['worker_log_sha256'] != ids[name+'_worker.log']
                or record['model_state_sha256'] != launch['assigned_model_states'][case[4]]):
            raise ValueError('same worker, collection, audit, startup, readout and assigned state required')
        parent = dict(case=name, worker_terminal_sha256=ids[name+'_worker_terminal.json'],
            verified_round_trip=record['verified_round_trip'], scientific_success_required=False)
        if read_json(batch.OUTPUT, name+'_parent_completion.json') != parent:
            raise ValueError('exact original parent completion receipt required')
        reports.append(report)
    summary = batch.complete_cohort(result['conditions'], reports)
    if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in summary.items()):
        raise ValueError('complete original six-case summary must reconstruct')
    for arm in study.ARMS:
        if arm.model_name is not None and launch['assigned_model_states'][arm.model_name] != arm.model_state_sha256:
            raise ValueError('preassigned independent-study model states must match the completed batch')
    return summary


def admit(batch_result_sha, sources):
    # Refuse incomplete native work before doing static reconstruction or full
    # original training/input verification. This function never launches work.
    result, launch, ids = completed(batch.OUTPUT, batch_result_sha, BATCH_LAUNCH, sources)
    summary = require_batch(result, launch)
    static = admit_static(sources)
    if static['factory_correction_admission'] != launch['input_admission']['correction_admission']:
        raise ValueError('byte-identical original corrected-model admission required')
    batch.verify_inputs(launch, full=True)
    verify(sources)
    return dict(static, adapter_batch_result_sha256=batch_result_sha,
        completed_artifact_bindings=static['completed_artifact_bindings']+[
            dict(root=str(batch.OUTPUT), artifact_sha256=ids)],
        completed_six_case_admission_performed=True, complete_input_admission_performed=True,
        full_training_and_coefficient_verifiers_rerun=True, development_summary=summary,
        all_scientific_failures_retained=True)


def verify_bound(admission, sources):
    verify(sources); verify(DOCUMENTS)
    if (admission['static_inputs_verified'] is not True
            or admission['completed_six_case_admission_performed'] is not True
            or admission['complete_input_admission_performed'] is not True
            or admission['full_training_and_coefficient_verifiers_rerun'] is not True
            or admission['final_policy_review_completed'] is not False
            or admission['population_execution_permitted'] is not False
            or admission['new_layout_sensor_data_consumed'] is not False
            or admission['all_scientific_failures_retained'] is not True
            or admission['inventory_result_sha256'] != study.INVENTORY_RESULT_SHA256
            or admission['factory_result_sha256'] != FACTORY_RESULT
            or admission['document_sha256'] != DOCUMENTS
            or admission['assignment_manifest_sha256'] != DOCUMENTS[ASSIGNMENT]
            or admission['ordered_cases'] != study.manifest()['ordered_cases']
            or (admission['planned_episodes'], admission['independent_layout_units']) != (32, 8)):
        raise ValueError('complete input evidence with final policy review still outstanding required')
    expected = (*STATIC, (factory.original.INPUT, factory.evidence.ORIGINAL_SHA, factory.evidence.ORIGINAL_LAUNCH),
        (batch.OUTPUT, admission['adapter_batch_result_sha256'], BATCH_LAUNCH))
    if len(admission['completed_artifact_bindings']) != len(expected):
        raise ValueError('all four ordered original artifact groups required')
    for binding, (root, sha, launch_sha) in zip(admission['completed_artifact_bindings'], expected, strict=True):
        _, _, ids = completed(root, sha, launch_sha, sources)
        if binding != dict(root=str(root), artifact_sha256=ids):
            raise ValueError('exact complete original artifact bindings required')
    launch = read_json(batch.OUTPUT, 'launch.json'); result = read_json(batch.OUTPUT, 'result.json')
    if (admission['factory_correction_admission'] != launch['input_admission']['correction_admission']
            or admission['development_summary'] != require_batch(result, launch)):
        raise ValueError('unchanged completed development cohort and model inputs required')
    batch.verify_inputs(launch)
