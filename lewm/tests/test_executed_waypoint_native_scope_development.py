import ast
from pathlib import Path
from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES

ROOT = Path(__file__).resolve().parents[2]


def test_native_collector_and_raw_audit_change_only_controller_and_explicit_storage_identity():
    for component in ('episode', 'audit'):
        old = (ROOT/f'scripts/nominal_reentry_maze_{component}_development.py').read_text()
        expected = old.replace('nominal_reentry_round_trip_controller_development',
            'executed_waypoint_round_trip_controller_development')
        expected = expected.replace('NominalReentryRoundTripController', 'ExecutedWaypointRoundTripController')
        expected = expected.replace('nominal_reentry_resource_envelope_development',
            'executed_waypoint_resource_envelope_development')
        expected = expected.replace('NOMINAL_REENTRY_MAZE', 'EXECUTED_WAYPOINT_MAZE')
        actual = (ROOT/f'scripts/executed_waypoint_maze_{component}_development.py').read_text()
        assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    assert COLLECTION_ALLOWANCE_BYTES == 10*1024**3
