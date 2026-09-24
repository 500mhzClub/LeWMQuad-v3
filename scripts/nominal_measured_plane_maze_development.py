"""Same native loop and full audit with explicitly nominal predictive control."""
from contextlib import contextmanager

from lewm.measured_plane_comparator_controllers_development import MeasuredPlaneForecastSourceController
from scripts import measured_plane_extended_maze_development as original


class NominalMeasuredPlaneController(MeasuredPlaneForecastSourceController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, forecast_source='nominal_requested_twist', **kwargs)


@contextmanager
def forbid_model_forward(model):
    """Fail even if a controller catches an attempted model-call exception."""
    attempted = []
    def reject(module, args):
        attempted.append(type(module).__name__)
        raise RuntimeError('nominal navigation must not execute a learned model forward')
    handles = [module.register_forward_pre_hook(reject) for module in model.modules()]
    try:
        yield
        if attempted:
            raise ValueError('nominal navigation attempted model inference: '+str(attempted))
    finally:
        for handle in handles: handle.remove()


_collect = original.original._bind(original.collect,
    ResidualAnchoredContinuationController=NominalMeasuredPlaneController)
_audit = original.original._bind(original.audit,
    ResidualAnchoredContinuationController=NominalMeasuredPlaneController)
artifacts = original.artifacts
read_rows = original.read_rows
rgb_packet = original.rgb_packet
ExtendedBudgetRGBDReplay = original.ExtendedBudgetRGBDReplay


def collect(*args, model, **kwargs):
    with forbid_model_forward(model):
        result = _collect(*args, model=model, **kwargs)
    # The native collector persists its exact result internally. Return it
    # unchanged; source assignment is explicit in every controller decision
    # and in the parent launch/worker receipt.
    return result


def audit(*args, model, **kwargs):
    with forbid_model_forward(model):
        result = _audit(*args, model=model, **kwargs)
    if result['raw_model_command_replay_pass'] is not True:
        raise ValueError('original complete controller replay must have passed')
    return result | dict(raw_model_command_replay_pass=False,
        raw_controller_command_replay_pass=True, raw_nominal_forecast_command_replay_pass=True,
        high_level_world_model_used=False, actual_learned_model_forward_calls=0,
        nominal_predictive_controller=True, fully_nonpredictive_controller=False)


def definition():
    return original.definition() | dict(implementation_class='NominalMeasuredPlaneController',
        assigned_forecast_source='nominal_requested_twist', learned_model_forward_permitted=False,
        nominal_predictive_controller=True, fully_nonpredictive_controller=False,
        original_scoring_and_predictive_feasibility_retained=True,
        observed_forecast_residual_update_retained=True,
        isolated_planning_on_off_comparison=False)
