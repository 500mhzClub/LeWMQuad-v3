"""Separate revised-perception assignments; no independent scene execution."""
from dataclasses import asdict, replace

from lewm import independent_round_trip_comparison_study_development as previous
from lewm.independent_round_trip_layouts_development import specification
from lewm.novel_maze_round_trip_contract_development import WARMUP_TICKS, DRAIN_TICKS, RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from lewm.pulse_timed_training_runner_development import state_digest

Arm, Case = previous.Arm, previous.Case
INVENTORY_RESULT_SHA256 = previous.INVENTORY_RESULT_SHA256
CORRECTION_RESULT_SHA256 = previous.CORRECTION_RESULT_SHA256
NAVIGATION_TICKS = 4000
MAX_COMMAND_TICKS = WARMUP_TICKS+NAVIGATION_TICKS+DRAIN_TICKS
MAX_OBSERVATIONS = MAX_COMMAND_TICKS+1
COLLECTION_ALLOWANCE_BYTES = 14*1024**3
COLLECTION_STATUS = 'MEASURED_PLANE_INDEPENDENT_ROUND_TRIP_TERMINAL_AUDIT_REQUIRED'
IMPLEMENTATIONS = dict(persistent_jepa='MeasuredPlaneResidualController',
    persistent_supervised='MeasuredPlaneResidualController', reactive='MeasuredPlaneReactiveController',
    current_pair_jepa='MeasuredPlaneCurrentObservationPlanningController')
ARMS = tuple(replace(arm, implementation=IMPLEMENTATIONS[arm.name]) for arm in previous.ARMS)
CASES = tuple(Case(f'measured_plane_independent_round_trip_{index:02d}_{arm.name}', index, arm.name)
    for index in range(8) for arm in ARMS[index % 4:]+ARMS[:index % 4])


def require_case(case):
    if type(case) is not Case or type(case.layout_index) is not int or case not in CASES:
        raise ValueError('exact case from the new measured-plane independent roster required')
    return next(arm for arm in ARMS if arm.name == case.arm_name)


def treatment(case):
    arm = require_case(case); learned = arm.model_name is not None
    return dict(case=case.name, layout_index=case.layout_index, assignment=asdict(arm),
        high_level_world_model_configured=learned,
        residual_first_interval_feasibility_fallback_enabled=learned,
        residual_hold_feasibility_enabled=learned, residual_anchored_continuation_enabled=learned,
        measured_plane_constrained_estimator=True, planning_map_variant=arm.planning_map,
        persistent_contact_history_retained=True, independent_layout_development_execution=True,
        reused_development_layout=False)


def require_collection(case, result):
    if result.get('status') != COLLECTION_STATUS or result.get('navigation_ticks') != NAVIGATION_TICKS:
        raise ValueError('complete new measured-plane collection and 4000-tick budget required')
    for key, expected in treatment(case).items():
        if type(result.get(key)) is not type(expected) or result[key] != expected:
            raise ValueError('assigned measured-plane treatment changed: '+key)


def verify_model(case, model):
    arm = require_case(case)
    if arm.model_name is None:
        if model is not None: raise ValueError('reactive arm must have no high-level world model')
        return None
    if (model is None or model.training or state_digest(model.state_dict()) != arm.model_state_sha256
            or any(parameter.grad is not None for parameter in model.parameters())):
        raise ValueError('same assigned corrected evaluation model without gradients required')
    return arm.model_state_sha256


def command_role(case):
    return 'online_reactive_round_trip_command' if require_case(case).model_name is None else 'online_learned_round_trip_command'


def replay_receipt(case):
    learned = require_case(case).model_name is not None
    return dict(raw_controller_command_replay_pass=True, raw_model_command_replay_pass=learned,
        model_state_unchanged=True if learned else None, high_level_world_model_used=learned)


def manifest():
    rows = []
    for case in CASES:
        spec = specification(case.layout_index)
        rows.append(asdict(case) | dict(assignment=asdict(require_case(case)), scene_id=spec['scene_id'],
            physics_seed=spec['procedural_seed'], appearance_seed=spec['appearance_seed']))
    return dict(schema='measured_plane_independent_round_trip_assignments_development.v1',
        inventory_result_sha256=INVENTORY_RESULT_SHA256, correction_result_sha256=CORRECTION_RESULT_SHA256,
        ordered_cases=rows, planned_episodes=32, independent_layout_units=8, training_seed_replications=1,
        navigation_ticks=NAVIGATION_TICKS, max_observations=MAX_OBSERVATIONS,
        case_order='layout_major_cyclic_arm_rotation', fresh_process_controller_memory_and_model_per_case=True,
        retain_scientific_failures=True, outcome_based_layout_replacement=False, checkpoint_selection_performed=False,
        measured_plane_constrained_estimator=True, reactive_is_whole_method_comparison=True,
        isolated_prediction_ranking_ablation=False, current_pair_arm_is_fully_memoryless=False,
        repeated_appearance_condition_included=False, nominal_predictive_arm_included=False,
        no_rgb_direct_development_reference_included=False,
        completed_measured_plane_comparison_review_required_before_launch=True,
        previous_32_case_definition_modified=False, execution_protocol_frozen=False,
        population_execution_permitted=False, native_execution=False, navigation_qualified=False, goal_achieved=False)


def resources_for(resources, completed_case_names=()):
    completed = tuple(completed_case_names); expected = tuple(case.name for case in CASES)
    if len(completed) >= len(CASES) or completed != expected[:len(completed)]:
        raise ValueError('ordered completed new-roster prefix with cases remaining required')
    remaining = len(CASES)-len(completed)
    required = RESERVE_BYTES+remaining*(COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES)
    if any(type(resources[key]) is not int or resources[key] < 0 for key in ('memory_available_bytes', 'artifact_free_bytes')):
        raise ValueError('nonnegative integer measured resource bytes required')
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < required:
        raise ValueError('complete remaining measured-plane population allowance unavailable')
    return dict(remaining_cases=remaining, required_free_bytes=required, memory_admission_bytes=32*1024**3,
        native_scene_workers=1, os_resource_limits_enforced=False, completion_evidence_verified=False)
