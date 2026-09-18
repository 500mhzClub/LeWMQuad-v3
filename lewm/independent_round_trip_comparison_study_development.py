"""Prospective fixed assignments; no scene execution or outcome selection."""
from dataclasses import asdict, dataclass

from lewm.independent_round_trip_layouts_development import specification
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES

INVENTORY_RESULT_SHA256 = 'c8021010cd7d22ef7f5a0b057c2c1144dfdfb152dc95895a3066631de2221474'
CORRECTION_RESULT_SHA256 = '1b36dc77ca51d342e45d73da027ebdbdbd1263ab5be4142766948fd8258dd460'


@dataclass(frozen=True)
class Arm:
    name: str
    implementation: str
    model_name: str | None
    model_state_sha256: str | None
    condition: str | None
    variant: str | None
    planning_map: str


ARMS = (
    Arm('persistent_jepa', 'ResidualAnchoredContinuationController',
        'seed_2026091001_full_jepa',
        '35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a',
        'jepa', 'full', 'persistent'),
    Arm('persistent_supervised', 'ResidualAnchoredContinuationController',
        'seed_2026091001_full_supervised_rollout',
        '755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5',
        'supervised_rollout', 'full', 'persistent'),
    Arm('reactive', 'ReactiveFloorTransportController', None, None, None, None, 'persistent'),
    Arm('current_pair_jepa', 'ResidualCurrentObservationPlanningController',
        'seed_2026091001_full_jepa',
        '35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a',
        'jepa', 'full', 'current_paired_observation'),
)


@dataclass(frozen=True)
class Case:
    name: str
    layout_index: int
    arm_name: str


# Layout-major cyclic rotation. Each arm occupies each within-layout position
# twice over eight layouts. This is fixed before any new-layout execution.
CASES = tuple(Case(f'independent_round_trip_{index:02d}_{arm.name}', index, arm.name)
    for index in range(8) for arm in ARMS[index % 4:] + ARMS[:index % 4])


def require_case(case):
    if (type(case) is not Case or type(case.layout_index) is not int or case not in CASES):
        raise ValueError('exact case from the fixed independent-layout roster required')
    return next(arm for arm in ARMS if arm.name == case.arm_name)


def manifest():
    rows = []
    for case in CASES:
        spec = specification(case.layout_index)
        rows.append(asdict(case) | dict(assignment=asdict(require_case(case)),
            scene_id=spec['scene_id'], physics_seed=spec['procedural_seed'],
            appearance_seed=spec['appearance_seed']))
    return dict(schema='independent_round_trip_comparison_assignments_development.v1',
        inventory_result_sha256=INVENTORY_RESULT_SHA256,
        correction_result_sha256=CORRECTION_RESULT_SHA256,
        ordered_cases=rows, planned_episodes=32, independent_layout_units=8,
        case_order='layout_major_cyclic_arm_rotation',
        fresh_process_controller_memory_and_model_per_case=True,
        retain_scientific_failures=True, outcome_based_layout_replacement=False,
        checkpoint_selection_performed=False, reactive_is_whole_method_comparison=True,
        current_pair_arm_is_fully_memoryless=False,
        completed_six_model_development_review_required_before_launch=True,
        execution_protocol_frozen=False, native_execution=False,
        navigation_qualified=False, goal_achieved=False)


def resources_for(resources, completed_case_names=()):
    completed = tuple(completed_case_names)
    expected = tuple(case.name for case in CASES)
    if len(completed) >= len(CASES) or completed != expected[:len(completed)]:
        raise ValueError('ordered completed roster prefix with cases remaining required')
    remaining = len(CASES) - len(completed)
    required = RESERVE_BYTES + remaining * (COLLECTION_ALLOWANCE_BYTES + PERSISTENCE_HEADROOM_BYTES)
    if any(type(resources[key]) is not int or resources[key] < 0
            for key in ('memory_available_bytes', 'artifact_free_bytes')):
        raise ValueError('nonnegative integer measured resource bytes required')
    if resources['memory_available_bytes'] < 32 * 1024**3 or resources['artifact_free_bytes'] < required:
        raise ValueError('complete remaining population resource allowance unavailable')
    return dict(remaining_cases=remaining, required_free_bytes=required,
        memory_admission_bytes=32 * 1024**3, native_scene_workers=1,
        os_resource_limits_enforced=False, completion_evidence_verified=False)
