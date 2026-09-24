import ast
from pathlib import Path


def test_collector_and_full_raw_audit_preserve_native_execution_and_strict_gates():
    for component in ('episode', 'audit'):
        old = Path(f'scripts/confirmed_floor_maze_{component}_development.py').read_text()
        expected = old.replace('confirmed_floor_round_trip_controller_development', 'joint_floor_registered_controller_development')
        expected = expected.replace('ConfirmedFloorRoundTripController', 'JointFloorRegisteredRoundTripController')
        expected = expected.replace('CONFIRMED_FLOOR_MAZE', 'JOINT_FLOOR_REGISTERED_MAZE')
        actual = Path(f'scripts/joint_floor_registered_maze_{component}_development.py').read_text()
        assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
