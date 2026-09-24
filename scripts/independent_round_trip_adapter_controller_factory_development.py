"""Fresh, explicitly assigned controllers; full input admission stays upstream."""
from lewm.independent_round_trip_comparison_study_development import (
    CORRECTION_RESULT_SHA256, require_case)
from lewm.independent_round_trip_layouts_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.reactive_floor_transport_controller_development import ReactiveFloorTransportController
from lewm.residual_current_observation_planning_controller_development import ResidualCurrentObservationPlanningController
from scripts.all_phase_planner_model_admission_development import load_assigned
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel


def create(case, geometry, *, correction_admission=None):
    """Load afresh on each call; never accept a caller-selected model/controller.

    The future launcher must first authenticate the full inventory, correction,
    model and completed development-study admissions. This factory checks the
    assigned treatment and actual corrected tensor state; it is not admission.
    """
    arm = require_case(case)
    options = dict(public_mission=public_mission(case.layout_index), navigation_ticks=NAVIGATION_TICKS)
    if arm.name == 'reactive':
        controller = ReactiveFloorTransportController(geometry, **options)
        if hasattr(controller, 'model') or hasattr(controller, 'residual'):
            raise ValueError('reactive controller must have no high-level model or learned residual')
        return controller, None
    if (correction_admission is None or
            correction_admission['correction_result_sha256'] != CORRECTION_RESULT_SHA256):
        raise ValueError('exact completed expanded correction admission required')
    model, condition, variant = load_assigned(correction_admission, arm.model_name)
    if (type(model) is not AllPhasePlannerModel or model.training or
            (condition, variant) != (arm.condition, arm.variant) or
            state_digest(model.state_dict()) != arm.model_state_sha256):
        raise ValueError('exact preassigned corrected model state and treatment required')
    implementation = (ResidualCurrentObservationPlanningController if arm.name == 'current_pair_jepa'
        else ResidualAnchoredContinuationController)
    controller = implementation(model, geometry, **options, condition=condition, variant=variant, persistent=True)
    if state_digest(model.state_dict()) != arm.model_state_sha256:
        raise ValueError('controller construction changed the assigned model state')
    return controller, model
