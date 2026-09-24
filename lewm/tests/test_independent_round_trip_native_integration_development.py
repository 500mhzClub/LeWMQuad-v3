"""Source-boundary checks; these tests never create a native scene."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from lewm.independent_round_trip_layouts_development import specification, pack
from lewm.novel_maze_round_trip_scene_development import specification as old_specification
from scripts.independent_round_trip_session_development import (
    IndependentRoundTripPhysicalInit, IndependentRoundTripSession)
from scripts.novel_maze_round_trip_physical_session_development import NovelMazeRoundTripPhysicalInit
from scripts.renderer_witness_dual_camera_maze_session_development import RendererWitnessDualCameraMazeSession
from scripts.run_physical_graph_edge_handoff_qualification_v1 import ExperimentError


class SceneBoundary(RuntimeError):
    pass


@pytest.mark.parametrize('index', range(8))
def test_real_constructor_chain_reaches_only_the_exact_new_pack(monkeypatch, tmp_path, index):
    from lewm_genesis import visible_robot_union_rgbd_scene_development as scene
    calls = []

    def boundary(definition, **kwargs):
        calls.append((definition, kwargs))
        raise SceneBoundary('no native scene constructed')

    monkeypatch.setattr(scene, 'build_scene_from_pack', boundary)
    spec = specification(index)
    obj = IndependentRoundTripSession.__new__(IndependentRoundTripSession)
    with pytest.raises(SceneBoundary):
        obj.__init__(spec, tmp_path)
    assert calls == [(pack(spec), dict(output=tmp_path/'visual_meshes',
        appearance_arm=spec['appearance_arm'], appearance_seed=spec['appearance_seed'],
        n_envs=1, backend='cpu', show_viewer=False, render_robot=True))]
    assert obj.spec == spec and obj.spec is not spec
    assert obj.geometry == spec['geometry'] and obj.geometry is not spec['geometry']
    assert obj.renderer_witnesses == dict(primary=[], paired=[], failures=[])
    assert obj.auxiliary_audit == obj.depth_manifest == obj.fast_rows == obj.sensor_rows == []
    assert obj.guard is None and obj.guard_rows == [] and obj.samples == []
    assert obj.backend == 'cpu' and not hasattr(obj, 'ctx')


@pytest.mark.parametrize('change', ['predecessor', 'wall', 'seed', 'route', 'backend'])
def test_wrong_scene_rejected_before_scene_builder_or_visual_output(monkeypatch, tmp_path, change):
    from lewm_genesis import visible_robot_union_rgbd_scene_development as scene
    def forbidden(*args, **kwargs):
        pytest.fail('rejected input reached the scene builder')
    monkeypatch.setattr(scene, 'build_scene_from_pack', forbidden)
    spec = deepcopy(specification(0)); backend = 'cpu'
    if change == 'predecessor': spec = old_specification(0)
    elif change == 'wall': spec['geometry']['wall_boxes'][0]['size_xyz'][0] += .01
    elif change == 'seed': spec['procedural_seed'] += 1
    elif change == 'route': spec['evaluation_layout']['shortest_outbound_route'].reverse()
    else: backend = 'gpu'
    obj = IndependentRoundTripPhysicalInit.__new__(IndependentRoundTripPhysicalInit)
    obj.output = tmp_path
    with pytest.raises(ExperimentError): obj.__init__(spec, backend=backend)
    assert not (tmp_path/'visual_meshes').exists()


def test_all_acquisition_contact_and_persistence_wrappers_retain_original_order():
    old = RendererWitnessDualCameraMazeSession.__mro__
    new = IndependentRoundTripSession.__mro__
    assert tuple(c for c in new if c not in (IndependentRoundTripSession, IndependentRoundTripPhysicalInit)) == old
    assert new.index(IndependentRoundTripPhysicalInit)+1 == new.index(NovelMazeRoundTripPhysicalInit)
    for name in ('_sample', 'command_tick', 'capture_fixed_rgb', 'capture_current',
            'sensor_packets', 'persist', 'persist_observations', 'install_contact_identity'):
        assert getattr(IndependentRoundTripSession, name) is getattr(RendererWitnessDualCameraMazeSession, name)


def body(function):
    import textwrap
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))), include_attributes=False)


def test_physical_initialization_changes_no_gait_physics_or_guard_operations():
    assert body(IndependentRoundTripPhysicalInit.__init__) == body(NovelMazeRoundTripPhysicalInit.__init__)


def test_collector_and_auditor_keep_full_raw_contract_and_controller_boundary():
    from scripts import independent_residual_round_trip_episode_development as new_collect
    from scripts import residual_anchored_continuation_maze_episode_development as old_collect
    from scripts import independent_residual_round_trip_audit_development as new_audit
    from scripts import residual_anchored_continuation_maze_audit_development as old_audit
    source = inspect.getsource(new_collect.collect)
    source = source.replace('IndependentRoundTripSession', 'RendererWitnessDualCameraMazeSession')
    source = source.replace('INDEPENDENT_RESIDUAL_ROUND_TRIP_TERMINAL_AUDIT_REQUIRED',
        'RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED')
    source = source.replace('INDEPENDENT_RESIDUAL_ROUND_TRIP_COLLECTED',
        'RESIDUAL_ANCHORED_CONTINUATION_MAZE_COLLECTED')
    assert ast.dump(ast.parse(source)) == body(old_collect.collect)
    assert body(new_collect.artifacts) == body(old_collect.artifacts)
    source = inspect.getsource(new_audit.audit).replace(
        'independent_layout_development_execution=True, reused_development_layout=False',
        'independent_layout_development_execution=False, reused_development_layout=True')
    assert ast.dump(ast.parse(source)) == body(old_audit.audit)
    assert new_collect.specification is new_audit.specification is specification
    assert new_collect.public_mission is new_audit.public_mission
    assert new_collect.ResidualAnchoredContinuationController is new_audit.ResidualAnchoredContinuationController
    assert new_audit.evaluate.__module__ == 'lewm.independent_round_trip_evaluation_development'
    for module in (new_collect, new_audit):
        tree = ast.parse(Path(module.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == 'ResidualAnchoredContinuationController']
        assert len(calls) == 1
        assert {kw.arg for kw in calls[0].keywords} == {
            'public_mission', 'navigation_ticks', 'persistent', 'condition', 'variant'}
