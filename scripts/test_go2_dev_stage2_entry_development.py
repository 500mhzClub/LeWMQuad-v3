"""Tests for the stage-2 mission wrapper's loader (scripts/run_go2_dev_mission_stage2_development.py), on a temporary
registry. They cover hash verification, the role and maze-range checks, the spec built as the owner's loader builds it,
and deferral to the unchanged loader for every other set.
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_stage2_entry_development
"""
import hashlib
import json
from pathlib import Path
import tempfile

from scripts import run_go2_dev_mission_stage2_development as stage2


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def registry(root, tamper=False):
    entries = []
    for maze, role in ((3, 'stage2_fit'), (31, 'stage2_eval')):
        maze_record = write(root/'sets'/role/f'maze_{maze:02d}.json',
                            dict(data_role=role, layout_index=maze, geometry=dict(spawn_se2_world=[0, 0, 0])))
        episode_record = write(root/'sets'/role/f'episode_{maze:02d}_1.json',
                               dict(role=role, simulation_seed=7, home_se2_world=[1.3, 0, 0], beacon_xy_world=[2.6, 0]))
        entries.append(dict(maze_id=maze, role=role, episode=1, maze=maze_record, episodes=[episode_record]))
    write(root/stage2.REGISTRY, dict(entries=entries))
    if tamper:
        (root/'sets/stage2_fit/maze_03.json').write_text('{}')


def expect_error(f, *args):
    try:
        f(*args)
    except (ValueError, AssertionError):
        return
    raise AssertionError('expected a refusal')


def test_loader():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        registry(root)
        spec, packet = stage2.loader_for('stage2_fit', 3, False)(root, 3, 1)
        assert spec['procedural_seed'] == 7 and spec['geometry']['spawn_se2_world'] == [1.3, 0, 0] and packet['role'] == 'stage2_fit'
        spec, _ = stage2.loader_for('stage2_eval', 31, False)(root, 31, 1)
        assert spec['data_role'] == 'stage2_eval'
        expect_error(stage2.loader_for('stage2_eval', 3, False), root, 3, 1)    # maze outside the eval range
        expect_error(stage2.loader_for('stage2_fit', 3, False), root, 3, 0)     # unregistered episode
        expect_error(stage2.loader_for('stage2_heldout', 31, False), root, 31, 1)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        registry(root, tamper=True)
        expect_error(stage2.loader_for('stage2_fit', 3, False), root, 3, 1)     # hash mismatch


def test_other_sets_defer():
    from scripts import run_go2_dev_mission_development as dev
    assert dev.loader_for is stage2.loader_for
    assert stage2.loader_for('prelim_test', 33, False) is dev.prelim_inputs
    expect_error(stage2.loader_for, 'sealed_test', 0, False)


if __name__ == '__main__':
    for test in (test_loader, test_other_sets_defer):
        test()
        print('ok', test.__name__)
    print('2 passed')
