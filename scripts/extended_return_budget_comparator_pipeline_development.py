"""Longer comparator collection/raw audit with common sampled resource guards.

These reusable development pipelines have no launcher or population roster.
Model admission, closed-artifact authentication and whole-worker accounting
remain the future runner's responsibilities.
"""
from contextlib import nullcontext
from functools import partial

from lewm.extended_return_budget_comparator_controllers_development import (
    ExtendedReturnBudgetForecastSourceController, ExtendedReturnBudgetReactiveController)
from lewm.extended_return_budget_current_planning_development import ExtendedReturnBudgetCurrentPlanningController
from scripts import extended_return_budget_maze_pipeline_development as extended
from scripts import resource_guarded_extended_return_maze_development as guarded
from scripts import reactive_floor_transport_maze_episode_development as reactive_episode
from scripts import reactive_floor_transport_maze_audit_development as reactive_audit
from scripts import reactive_nominal_maze_command_audit_development as reactive_commands
from scripts.nominal_measured_plane_maze_development import forbid_model_forward

resources = guarded.resources
MODES = ('frozen_reference', 'nominal', 'reactive', 'current_planning')
CONTROLLERS = dict(
    frozen_reference=partial(ExtendedReturnBudgetForecastSourceController, forecast_source='frozen_world_model'),
    nominal=partial(ExtendedReturnBudgetForecastSourceController, forecast_source='nominal_requested_twist'),
    reactive=ExtendedReturnBudgetReactiveController,
    current_planning=ExtendedReturnBudgetCurrentPlanningController)
audit_reactive_commands = extended.bind(reactive_commands.audit_commands, NAVIGATION_TICKS=8000)
_reactive_collect = extended.bind(reactive_episode.collect,
    NAVIGATION_TICKS=8000, MAX_OBSERVATIONS=extended.MAX_OBSERVATIONS,
    COLLECTION_ALLOWANCE_BYTES=extended.COLLECTION_ALLOWANCE_BYTES,
    writer=extended.writer, RendererWitnessDualCameraMazeSession=extended.ExtendedReturnBudgetRendererSession,
    ReactiveFloorTransportController=ExtendedReturnBudgetReactiveController)
_reactive_audit = extended.bind(reactive_audit.audit,
    NAVIGATION_TICKS=8000, MAX_COMMAND_TICKS=extended.MAX_COMMAND_TICKS,
    IntentReturnRGBDReplay=extended.ExtendedReturnBudgetRGBDReplay,
    audit_sensors=extended.audit_sensors, audit_commands=audit_reactive_commands,
    read_rows=extended.read_rows, packet=extended.rgb_packet,
    renderer_audit=extended.renderer_audit, evaluate=extended.evaluate,
    ReactiveFloorTransportController=ExtendedReturnBudgetReactiveController)


def require_mode(mode):
    if type(mode) is not str or mode not in MODES:
        raise ValueError('explicit prepared longer comparator mode required')
    return mode


def functions(mode):
    require_mode(mode)
    if mode == 'reactive': return _reactive_collect, _reactive_audit
    return tuple(extended.bind(function, ResidualAnchoredContinuationController=CONTROLLERS[mode])
        for function in (extended.collect, extended.audit))


def _execute(mode, phase, args, kwargs):
    require_mode(mode)
    if mode == 'reactive' and any(key in kwargs for key in ('model', 'condition', 'variant')):
        raise ValueError('reactive pipeline has no model or predictive treatment arguments')
    if mode != 'reactive' and kwargs.get('model') is None:
        raise ValueError('caller-admitted model assignment required for predictive comparator')
    root = kwargs['output'] if phase == 'collection' else kwargs['input_root']
    episode = kwargs['episode_name']
    guard = resources.ResourceGuard(root, episode, phase); error = None
    try:
        first = guard.check('begin')
        if phase == 'collection': resources.admission(first)
        with forbid_model_forward(kwargs['model']) if mode == 'nominal' else nullcontext():
            function = functions(mode)[0 if phase == 'collection' else 1]
            def create(*a, **k):
                return guarded.CheckedController(CONTROLLERS[mode](*a, **k), guard)
            key = 'ReactiveFloorTransportController' if mode == 'reactive' else 'ResidualAnchoredContinuationController'
            bindings = {key:create}
            if phase == 'collection':
                bindings['RendererWitnessDualCameraMazeSession'] = guarded.session_type(guard)
            else:
                def sensors(*a, **k):
                    guard.check('before_sensor_audit')
                    result = extended.audit_sensors(*a, **k)
                    guard.check('after_sensor_audit')
                    return result
                bindings['audit_sensors'] = sensors
            result = extended.bind(function, **bindings)(*args, **kwargs)
        if phase == 'audit':
            flag = 'raw_controller_command_replay_pass' if mode == 'reactive' else 'raw_model_command_replay_pass'
            if result.get(flag) is not True: raise ValueError('complete original raw controller replay required')
            if mode == 'nominal':
                result = result | dict(raw_model_command_replay_pass=False, raw_controller_command_replay_pass=True,
                    raw_nominal_forecast_command_replay_pass=True, high_level_world_model_used=False,
                    actual_learned_model_forward_calls=0, nominal_predictive_controller=True,
                    fully_nonpredictive_controller=False)
        guard.check('completed')
        return result
    except BaseException as failure:
        error = failure
        raise
    finally:
        guard.finish(error)


def collect(*args, mode, **kwargs):
    return _execute(mode, 'collection', args, kwargs)


def audit(*args, mode, **kwargs):
    return _execute(mode, 'audit', args, kwargs)


def artifacts(layout_index, result, *, mode):
    require_mode(mode)
    return (reactive_episode.artifacts if mode == 'reactive' else extended.artifacts)(layout_index, result)


resource_artifacts = guarded.resource_artifacts


def definition(mode):
    require_mode(mode)
    result = guarded.definition() | dict(comparator_mode=mode, population_execution_permitted=False,
        controller_assignment_check_required_from_caller=True, whole_worker_accounting_required=True)
    if mode == 'reactive':
        return result | dict(implementation_class='ExtendedReturnBudgetReactiveController',
            high_level_world_model_loaded=False, candidate_future_outcomes_evaluated=False,
            predictive_surface_or_path_gates_applied=False, learned_residual_used=False, fully_nonpredictive_controller=True,
            reactive_is_whole_method_comparison=True, isolated_prediction_ranking_ablation=False,
            future_constraint_gates_matched=False, original_reactive_physics_loop_retained=True,
            original_reactive_command_audit_retained=True)
    if mode == 'current_planning':
        from lewm.residual_current_observation_planning_controller_development import METADATA
        return result | dict(implementation_class='ExtendedReturnBudgetCurrentPlanningController', **METADATA)
    return result | dict(implementation_class='ExtendedReturnBudgetForecastSourceController',
        assigned_forecast_source='nominal_requested_twist' if mode == 'nominal' else 'frozen_world_model',
        learned_model_forward_permitted=mode != 'nominal', nominal_predictive_controller=mode == 'nominal',
        fully_nonpredictive_controller=False, observed_forecast_residual_update_retained=True,
        original_scoring_and_predictive_feasibility_retained=True)
