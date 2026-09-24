import ast
from pathlib import Path
from lewm.nominal_reentry_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES

ROOT = Path(__file__).resolve().parents[2]


def test_collector_changes_only_controller_identity_and_declared_storage_allowance():
    old = (ROOT/'scripts/novel_maze_round_trip_episode_development.py').read_text()
    expected = old.replace('from lewm.observed_round_trip_controller_development import ObservedRoundTripController',
        'from lewm.nominal_reentry_round_trip_controller_development import NominalReentryRoundTripController')
    expected = expected.replace('controller = ObservedRoundTripController(', 'controller = NominalReentryRoundTripController(')
    expected = expected.replace('RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES, COLLECTION_ALLOWANCE_BYTES)',
        'RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES)\nfrom lewm.nominal_reentry_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES')
    expected = expected.replace('NOVEL_MAZE_ROUND_TRIP_TERMINAL_AUDIT_REQUIRED', 'NOMINAL_REENTRY_MAZE_TERMINAL_AUDIT_REQUIRED')
    expected = expected.replace('NOVEL_MAZE_ROUND_TRIP_COLLECTED', 'NOMINAL_REENTRY_MAZE_COLLECTED')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((ROOT/'scripts/nominal_reentry_maze_episode_development.py').read_text()))
    assert COLLECTION_ALLOWANCE_BYTES == 11*1024**3


def test_raw_audit_and_goal_verification_unchanged_and_reused_layout_identified():
    old = (ROOT/'scripts/novel_maze_round_trip_audit_development.py').read_text()
    expected = old.replace('from lewm.observed_round_trip_controller_development import ObservedRoundTripController',
        'from lewm.nominal_reentry_round_trip_controller_development import NominalReentryRoundTripController')
    expected = expected.replace('controller = ObservedRoundTripController(', 'controller = NominalReentryRoundTripController(')
    expected = expected.replace('independent_layout_development_execution=True',
        'independent_layout_development_execution=False, reused_development_layout=True')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((ROOT/'scripts/nominal_reentry_maze_audit_development.py').read_text()))
