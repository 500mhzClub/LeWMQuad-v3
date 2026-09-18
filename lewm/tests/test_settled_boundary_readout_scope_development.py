import ast
from pathlib import Path


def test_readout_retains_original_execution_and_failure_accounting():
    old=Path('scripts/read_go2_later_floor_resolution_maze_pilot_v1.py').read_text()
    expected=old.replace('scripts.run_go2_later_floor_resolution_maze_pilot_v1',
        'scripts.run_go2_settled_boundary_maze_pilot_v1').replace('go2_later_floor_resolution_maze_readout_v1',
        'go2_settled_boundary_maze_readout_v1').replace('scripts/read_go2_later_floor_resolution_maze_pilot_v1.py',
        'scripts/read_go2_settled_boundary_maze_pilot_v1.py').replace('LATER_FLOOR_RESOLUTION_MAZE','SETTLED_BOUNDARY_MAZE')
    actual=Path('scripts/read_go2_settled_boundary_maze_pilot_v1.py').read_text()
    assert ast.dump(ast.parse(actual))==ast.dump(ast.parse(expected))
