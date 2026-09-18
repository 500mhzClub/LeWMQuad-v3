"""Preserve native execution and independent audit when changing route targets."""
import ast
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('part', ['episode', 'audit'])
def test_native_collection_and_raw_audit_only_replace_controller(part):
    original = (ROOT/f'scripts/reactive_nominal_maze_{part}_development.py').read_text()
    expected = original.replace(
        'from lewm.reactive_nominal_round_trip_controller_development import ReactiveNominalRoundTripController',
        'from lewm.reactive_connector_round_trip_controller_development import ReactiveConnectorRoundTripController')
    expected = expected.replace('ReactiveNominalRoundTripController(', 'ReactiveConnectorRoundTripController(')
    expected = expected.replace('REACTIVE_NOMINAL_MAZE', 'REACTIVE_CONNECTOR_MAZE')
    actual = (ROOT/f'scripts/reactive_connector_maze_{part}_development.py').read_text()
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
