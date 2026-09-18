"""Exact treatment receipts and model-integrity checks for all four arms."""
from dataclasses import asdict

from lewm.independent_round_trip_comparison_study_development import require_case
from lewm.pulse_timed_training_runner_development import state_digest

COLLECTION_STATUS = 'INDEPENDENT_ROUND_TRIP_MULTIARM_TERMINAL_AUDIT_REQUIRED'


def treatment(case):
    arm = require_case(case)
    learned = arm.model_name is not None
    return dict(case=case.name, layout_index=case.layout_index, assignment=asdict(arm),
        high_level_world_model_configured=learned,
        residual_first_interval_feasibility_fallback_enabled=learned,
        residual_hold_feasibility_enabled=learned,
        residual_anchored_continuation_enabled=learned,
        planning_map_variant=arm.planning_map,
        persistent_contact_history_retained=True,
        independent_layout_development_execution=True, reused_development_layout=False)


def require_collection(case, result):
    if result.get('status') != COLLECTION_STATUS:
        raise ValueError('complete multiarm collection receipt required')
    for key, expected in treatment(case).items():
        if type(result.get(key)) is not type(expected) or result[key] != expected:
            raise ValueError('fixed collected treatment changed: ' + key)


def verify_model(case, model):
    arm = require_case(case)
    if arm.model_name is None:
        if model is not None:
            raise ValueError('reactive arm must have no high-level world model')
        return None
    if (model is None or state_digest(model.state_dict()) != arm.model_state_sha256
            or any(parameter.grad is not None for parameter in model.parameters())):
        raise ValueError('unchanged assigned corrected model without gradients required')
    return arm.model_state_sha256


def command_role(case):
    return ('online_reactive_round_trip_command' if require_case(case).model_name is None
        else 'online_learned_round_trip_command')


def replay_receipt(case):
    learned = require_case(case).model_name is not None
    return dict(raw_controller_command_replay_pass=True,
        raw_model_command_replay_pass=learned,
        model_state_unchanged=True if learned else None,
        high_level_world_model_used=learned)
