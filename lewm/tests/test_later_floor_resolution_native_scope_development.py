import ast
from pathlib import Path


def test_collector_and_raw_audit_preserve_physics_sensors_and_strict_gates():
    for component in ('episode', 'audit'):
        source = Path(f'scripts/joint_floor_registered_maze_{component}_development.py').read_text()
        expected = source.replace('joint_floor_registered_controller_development', 'later_floor_resolution_controller_development')
        expected = expected.replace('JointFloorRegisteredRoundTripController', 'LaterFloorResolutionRoundTripController')
        expected = expected.replace('JOINT_FLOOR_REGISTERED_MAZE', 'LATER_FLOOR_RESOLUTION_MAZE')
        actual = Path(f'scripts/later_floor_resolution_maze_{component}_development.py').read_text()
        assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
