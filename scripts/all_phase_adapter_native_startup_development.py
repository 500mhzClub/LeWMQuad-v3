"""Require actual startup and executed boundary command to match adapter replay."""
from itertools import islice
from scripts import replay_go2_all_phase_planner_adapter_startup_v2 as prefix
from scripts.all_phase_residual_maze02_startup_development import compare_startup
from scripts.maze_decision_stream_development import read_rows
from scripts.startup_raw_sensor_audit_development import read_json
from lewm.all_phase_residual_maze02_study_development import CASES


def require_boundary(replayed, actual, tape):
    if len(replayed) != 4 or len(actual) != 4 or len(tape) < 4:
        raise ValueError('four actual and replayed observations plus completed boundary command required')
    for i, (expected, observed) in enumerate(zip(replayed, actual, strict=True)):
        if (expected['tick'] != i or observed['tick'] != i
                or expected['decision'] != observed['decision']):
            raise ValueError('complete candidate decisions must match prospective adapter replay')
    command = tape[3]; decision = actual[3]['decision']
    if (decision['terminal'] is not None or decision['new_selection'] is None
            or 'prediction' not in decision['new_selection']
            or command['tick'] != 3 or command['completed'] is not True
            or command['pre_sample_index'] != 899 or command['post_sample_index'] != 949
            or command['requested_command'] != decision['requested_command']):
        raise ValueError('actual unchanged adapter forecast and complete first planned command required')
    return dict(complete_candidate_decisions_match_prospective_prefix=True,
        candidate_intervention_command_completed=True, first_intervention_frame=3,
        all_selected_forecasts_match_prospective_prefix=True,
        no_later_counterfactual_observations_used=True)


def compare(case, current):
    if tuple(case) not in CASES: raise ValueError('fixed assigned adapter case required')
    startup = compare_startup(prefix.INPUT/case[0], current)
    receipt = require_boundary(list(islice(read_rows(prefix.OUTPUT/case[0]), 4)),
        list(islice(read_rows(current), 4)), read_json(current, 'command_tape.json'))
    return startup | receipt
