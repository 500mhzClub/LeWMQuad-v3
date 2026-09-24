import ast
from pathlib import Path


def test_collection_and_audit_change_only_the_declared_controller_and_status():
    for component in ('episode','audit'):
        old=Path(f'scripts/later_floor_resolution_maze_{component}_development.py').read_text()
        expected=old.replace('lewm.later_floor_resolution_controller_development',
            'lewm.settled_boundary_round_trip_development').replace('LaterFloorResolutionRoundTripController',
            'SettledBoundaryRoundTripController').replace('LATER_FLOOR_RESOLUTION_MAZE','SETTLED_BOUNDARY_MAZE')
        new=Path(f'scripts/settled_boundary_maze_{component}_development.py').read_text()
        assert ast.dump(ast.parse(new))==ast.dump(ast.parse(expected))
