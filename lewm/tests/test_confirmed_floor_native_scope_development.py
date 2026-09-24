import ast
from pathlib import Path
from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES

ROOT = Path(__file__).resolve().parents[2]


def test_floor_intervention_preserves_native_execution_and_strict_evaluation():
    for component in ('episode', 'audit'):
        old = (ROOT/f'scripts/view_reentry_maze_{component}_development.py').read_text()
        expected = old.replace('view_reentry_round_trip_controller_development',
            'confirmed_floor_round_trip_controller_development')
        expected = expected.replace('ViewReentryRoundTripController', 'ConfirmedFloorRoundTripController')
        expected = expected.replace('VIEW_REENTRY_MAZE', 'CONFIRMED_FLOOR_MAZE')
        actual = (ROOT/f'scripts/confirmed_floor_maze_{component}_development.py').read_text()
        assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    assert COLLECTION_ALLOWANCE_BYTES == 10*1024**3
