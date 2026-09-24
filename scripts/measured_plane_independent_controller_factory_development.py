"""Fresh preassigned revised-perception controllers; no population launcher."""
from lewm.measured_plane_independent_round_trip_study_development import (
    CORRECTION_RESULT_SHA256, NAVIGATION_TICKS, require_case, verify_model)
from lewm.independent_round_trip_layouts_development import public_mission
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.measured_plane_comparator_controllers_development import MeasuredPlaneReactiveController
from lewm.measured_plane_current_observation_planning_controller_development import MeasuredPlaneCurrentObservationPlanningController
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from scripts.all_phase_planner_model_admission_development import load_assigned


def create(case, geometry, *, correction_admission=None):
    """The future launcher must authenticate complete inputs and final review."""
    arm = require_case(case)
    options = dict(public_mission=public_mission(case.layout_index), navigation_ticks=NAVIGATION_TICKS)
    if arm.name == 'reactive':
        controller = MeasuredPlaneReactiveController(geometry, **options)
        if hasattr(controller, 'model') or hasattr(controller, 'residual'):
            raise ValueError('reactive controller must not instantiate a model or learned residual')
        return controller, None
    if correction_admission is None or correction_admission['correction_result_sha256'] != CORRECTION_RESULT_SHA256:
        raise ValueError('exact original completed correction admission required')
    model, condition, variant = load_assigned(correction_admission, arm.model_name)
    if type(model) is not AllPhasePlannerModel or (condition, variant) != (arm.condition, arm.variant):
        raise ValueError('exact preassigned planner model adapter and treatment required')
    verify_model(case, model)
    implementation = MeasuredPlaneCurrentObservationPlanningController if arm.name == 'current_pair_jepa' else MeasuredPlaneResidualController
    controller = implementation(model, geometry, **options, condition=condition, variant=variant, persistent=True)
    if type(controller).__name__ != arm.implementation:
        raise ValueError('explicit new preassigned controller implementation required')
    verify_model(case, model)
    return controller, model
