"""Matched observed state until a training-objective change alters actual control."""
import numpy as np
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.observation_horizon_predictive_selection_development import require_short_forecast

MAX_FRAMES={1:215,2:504,3:265}
SHARED=('original_visual_evidence','evidence','memory_receipt','mission_receipt',
    'floor_partition_receipt','auxiliary_floor_partition_receipt','goal_initial_body_xy_m',
    'observed_goal_distance_m','quiet_intervals')


def compare_step(original,jepa,supervised,actual_command,*,frame,layout,corrections):
    if (type(frame) is not int or layout not in MAX_FRAMES or not 0<=frame<MAX_FRAMES[layout]
            or jepa!=original or original['requested_command']!=actual_command):
        raise ValueError('complete exact original JEPA replay on the fixed actual prefix required')
    for condition,d in [('jepa',jepa),('supervised_rollout',supervised)]:
        if (d['controller']!='measured_floor_transport_round_trip_controller_v1'
                or d['model_condition']!=condition or d['input_variant']!='full'
                or d['memory_variant']!='persistent'):
            raise ValueError('same full observed controller, memory and declared objective arm required')
        selection=d['new_selection']
        if selection and 'prediction' in selection:
            require_short_forecast(selection)
            prediction=np.asarray(selection['prediction'])
            if (selection['head']!='rollout_outcomes' or selection['input_variant']!='full'
                    or selection['model_prediction_corrected'] is not True
                    or selection['translation_bias_training_only'] is not True
                    or selection['translation_bias_xy_m']!=corrections[condition]
                    or prediction.shape!=(6,8,5) or not np.isfinite(prediction).all()):
                raise ValueError('same complete rollout head and each own training-only correction required')
        if d['terminal'] is None:
            if d['tick']!=frame or d['failure'] is not None:
                raise ValueError('current nonfailed controller state required')
            action=selection['action'] if selection else None
            request=[0.,0.,0.] if action is None else candidate_commands(action)[0]
            if d['requested_command']!=request: raise ValueError('request must match current selected action')
        elif d['requested_command']!=[0.,0.,0.]:
            raise ValueError('terminal controller must request zero')
    if any(jepa[k]!=supervised[k] for k in SHARED):
        raise ValueError('same current raw/registered observations, map and mission state required')
    if frame<3:
        normalized=dict(supervised,model_condition='jepa')
        if normalized!=jepa: raise ValueError('complete matched warmup before model use required')
    a=jepa['new_selection'] or {}; b=supervised['new_selection'] or {}
    banks='prediction' in a and 'prediction' in b
    command_changed=supervised['requested_command']!=actual_command
    terminal_changed=supervised['terminal']!=jepa['terminal']
    return dict(original_jepa_decision_exact=True,shared_observed_state_exact=True,
        both_full_forecast_banks_present=banks,
        raw_prediction_changed=banks and a['prediction']!=b['prediction'],
        requested_command_changed=command_changed,terminal_changed=terminal_changed,
        stop=command_changed or terminal_changed or jepa['terminal'] is not None or supervised['terminal'] is not None,
        online_residual_values_required_equal=False,
        model_objective_is_only_assigned_pipeline_change=True)
