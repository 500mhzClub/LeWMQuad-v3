import ast
from pathlib import Path
from lewm.reactive_nominal_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES

ROOT = Path(__file__).resolve().parents[2]


def test_collector_changes_only_high_level_controller_interface_role_and_declared_identity():
    s = (ROOT/'scripts/executed_waypoint_maze_episode_development.py').read_text()
    s = s.replace('from lewm.executed_waypoint_round_trip_controller_development import ExecutedWaypointRoundTripController',
        'from lewm.reactive_nominal_round_trip_controller_development import ReactiveNominalRoundTripController')
    s = s.replace('from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES',
        'from lewm.reactive_nominal_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES')
    s = s.replace('def collect(layout_index, definition, *, output, model, geometry, episode_name, condition, variant):',
        'def collect(layout_index, definition, *, output, geometry, episode_name):')
    s = s.replace('controller = ExecutedWaypointRoundTripController(model, geometry, public_mission=mission,\n            navigation_ticks=NAVIGATION_TICKS, persistent=True, condition=condition, variant=variant)',
        'controller = ReactiveNominalRoundTripController(geometry, public_mission=mission,\n            navigation_ticks=NAVIGATION_TICKS)')
    s = s.replace('online_learned_round_trip_command', 'online_reactive_round_trip_command').replace('EXECUTED_WAYPOINT_MAZE', 'REACTIVE_NOMINAL_MAZE')
    assert ast.dump(ast.parse(s)) == ast.dump(ast.parse((ROOT/'scripts/reactive_nominal_maze_episode_development.py').read_text()))
    assert COLLECTION_ALLOWANCE_BYTES == 10*1024**3


def test_actual_float64_command_slew_phase_and_drain_audit_only_changes_role_label():
    s = (ROOT/'scripts/novel_maze_round_trip_command_audit_development.py').read_text()
    expected = s.replace('online_learned_round_trip_command', 'online_reactive_round_trip_command')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((ROOT/'scripts/reactive_nominal_maze_command_audit_development.py').read_text()))


def test_independent_raw_audit_retains_all_sensor_visibility_and_mission_checks_without_model():
    s = (ROOT/'scripts/executed_waypoint_maze_audit_development.py').read_text()
    s = s.replace('Full raw sensor/model/command verification and independent maze evaluation.',
        'Full raw sensor/reactive-controller/command audit and independent maze evaluation.')
    s = s.replace('from lewm.executed_waypoint_round_trip_controller_development import ExecutedWaypointRoundTripController',
        'from lewm.reactive_nominal_round_trip_controller_development import ReactiveNominalRoundTripController')
    s = s.replace('from lewm.pulse_timed_training_runner_development import state_digest\n', '')
    s = s.replace('from scripts.novel_maze_round_trip_command_audit_development import audit_commands',
        'from scripts.reactive_nominal_maze_command_audit_development import audit_commands')
    s = s.replace('def audit(layout_index, result, definition, *, input_root, model, robot_geometry, episode_name, condition, variant):',
        'def audit(layout_index, result, definition, *, input_root, robot_geometry, episode_name):')
    s = s.replace('    before = state_digest(model.state_dict())\n', '')
    s = s.replace('controller = ExecutedWaypointRoundTripController(model, robot_geometry, public_mission=mission,\n        navigation_ticks=NAVIGATION_TICKS, condition=condition, variant=variant, persistent=True)',
        "controller = ReactiveNominalRoundTripController(robot_geometry, public_mission=mission,\n        navigation_ticks=NAVIGATION_TICKS)\n    assert not hasattr(controller, 'model') and not hasattr(controller, 'residual')")
    s = s.replace('    assert state_digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())\n', '')
    s = s.replace('raw_model_command_replay_pass=True', 'raw_controller_command_replay_pass=True').replace('model_state_unchanged=True', 'high_level_world_model_used=False')
    assert ast.dump(ast.parse(s)) == ast.dump(ast.parse((ROOT/'scripts/reactive_nominal_maze_audit_development.py').read_text()))
