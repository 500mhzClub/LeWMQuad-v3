"""Fresh longer comparator instances from explicit, caller-bound assignments.

This is a construction helper, not a population roster or input admission.
The caller must authenticate the full correction admission and freeze every
case's mode, model name, corrected tensor SHA and public mission before use.
"""
from dataclasses import dataclass

from lewm.independent_round_trip_comparison_study_development import CORRECTION_RESULT_SHA256
from lewm.extended_return_budget_mission_development import NAVIGATION_TICKS
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from scripts.all_phase_fit_execution_development import ROSTER
from scripts.all_phase_planner_model_admission_development import load_assigned
from scripts.extended_return_budget_comparator_pipeline_development import (
    CONTROLLERS, require_mode, forbid_model_forward)


@dataclass(frozen=True)
class ModelAssignment:
    name: str
    model_state_sha256: str


def require_assignment(assignment):
    if (type(assignment) is not ModelAssignment or type(assignment.name) is not str
            or type(assignment.model_state_sha256) is not str
            or len(assignment.model_state_sha256) != 64
            or any(c not in '0123456789abcdef' for c in assignment.model_state_sha256)):
        raise ValueError('explicit trained-model name and corrected tensor SHA required')
    rows = [row for row in ROSTER if row['name'] == assignment.name]
    if len(rows) != 1:
        raise ValueError('assignment must belong to the original eighteen-model training roster')
    return dict(rows[0])


def _check_model(model, expected_sha):
    if type(model) is not AllPhasePlannerModel:
        raise ValueError('exact planner-compatible expanded-model adapter required')
    if (any(module.training for module in model.modules())
            or any(p.grad is not None for p in model.parameters())
            or any(t.device.type != 'cpu' for t in model.state_dict().values())
            or state_digest(model.state_dict()) != expected_sha):
        raise ValueError('unchanged assigned CPU model in evaluation mode without gradients required')


def create(mode, geometry, *, public_mission, assignment=None, correction_admission=None):
    """Return controller, separately loaded model (or None), and identity receipt.

    No tensors, controller, navigation budget or private layout can be supplied
    as a substitute for the assigned model or the fixed public constructor.
    Nominal inference exclusion during later execution remains the pipeline's
    responsibility; every mode forbids inference during construction here.
    """
    require_mode(mode)
    options = dict(public_mission=public_mission, navigation_ticks=NAVIGATION_TICKS)
    receipt = dict(mode=mode, navigation_ticks=NAVIGATION_TICKS,
        population_assignment_authenticated=False, native_execution=False,
        model_forward_during_construction=False, model_training=False)
    if mode == 'reactive':
        if assignment is not None or correction_admission is not None:
            raise ValueError('reactive construction accepts no model assignment or correction admission')
        controller = CONTROLLERS[mode](geometry, **options)
        if hasattr(controller, 'model') or hasattr(controller, 'residual'):
            raise ValueError('reactive controller must have no world model or learned residual')
        return controller, None, receipt | dict(model_name=None, model_state_sha256=None,
            learned_model_loaded=False, reactive_is_whole_method_comparison=True)
    row = require_assignment(assignment)
    if (not isinstance(correction_admission, dict)
            or correction_admission.get('correction_result_sha256') != CORRECTION_RESULT_SHA256
            or correction_admission.get('all_coefficients_reconstructed') is not True
            or type(correction_admission.get('all_models')) is not int
            or correction_admission['all_models'] != 18
            or type(correction_admission.get('all_trained_heads')) is not int
            or correction_admission['all_trained_heads'] != 30):
        raise ValueError('complete original eighteen-model correction admission required')
    model, condition, variant = load_assigned(correction_admission, assignment.name)
    if (condition, variant) != (row['condition'], row['variant']):
        raise ValueError('loaded model treatment differs from the original training assignment')
    _check_model(model, assignment.model_state_sha256)
    with forbid_model_forward(model):
        controller = CONTROLLERS[mode](model, geometry, **options,
            condition=condition, variant=variant, persistent=True)
    _check_model(model, assignment.model_state_sha256)
    if controller.model is not model:
        raise ValueError('controller must retain exactly the freshly loaded assigned model')
    return controller, model, receipt | dict(model_name=assignment.name,
        model_state_sha256=assignment.model_state_sha256, condition=condition, variant=variant,
        training_seed=row['seed'], correction_result_sha256=CORRECTION_RESULT_SHA256,
        learned_model_loaded=True, model_unchanged_after_construction=True)
