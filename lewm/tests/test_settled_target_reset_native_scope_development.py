import ast
from pathlib import Path
from lewm.tests.test_settled_boundary_native_scope_development import (
    test_collection_and_audit_change_only_the_declared_controller_and_status)


def test_worker_unchanged_and_only_reviewed_collector_audit_are_imported():
    old=ast.parse(Path('scripts/run_go2_settled_boundary_maze_pilot_v1.py').read_text())
    new=ast.parse(Path('scripts/run_go2_settled_boundary_maze_pilot_v2.py').read_text())
    worker=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    assert ast.dump(worker(old))==ast.dump(worker(new))
    modules={n.module for n in new.body if isinstance(n,ast.ImportFrom)}
    assert {'scripts.settled_boundary_maze_episode_development',
        'scripts.settled_boundary_maze_audit_development',
        'scripts.settled_target_reset_native_prefix_comparison_development',
        'scripts.replay_go2_settled_boundary_controller_prefix_v2'} <= modules


def test_readout_changes_only_input_output_and_status_identities():
    old=Path('scripts/read_go2_settled_boundary_maze_pilot_v1.py').read_text()
    expected=old.replace('settled_boundary_maze_pilot_v1','settled_boundary_maze_pilot_v2')
    expected=expected.replace('settled_boundary_maze_readout_v1','settled_boundary_maze_readout_v2')
    expected=expected.replace('SETTLED_BOUNDARY_MAZE_PILOT_COMPLETE','SETTLED_BOUNDARY_MAZE_PILOT_V2_COMPLETE')
    expected=expected.replace('SETTLED_BOUNDARY_MAZE_READOUT_COMPLETE','SETTLED_BOUNDARY_MAZE_READOUT_V2_COMPLETE')
    actual=Path('scripts/read_go2_settled_boundary_maze_pilot_v2.py').read_text()
    assert ast.dump(ast.parse(actual))==ast.dump(ast.parse(expected))
