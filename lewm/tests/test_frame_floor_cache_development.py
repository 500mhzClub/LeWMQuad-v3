import ast
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.frame_floor_index_cache_development import FrameFloorIndexCache
from lewm.frame_cached_floor_geometry_development import FloorFrameGeometry
from lewm.frame_cached_floor_map_development import FrameCachedJointFloorRoundTripController
from lewm.joint_floor_registered_controller_development import JointFloorRegisteredRoundTripController
from lewm.joint_visual_floor_map_development import floor_coverage
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.current_primary_floor_plane_development import primary_floor_plane, confirm_auxiliary_floor, ROWS, COLUMNS
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.tests.test_current_primary_floor_plane_development import depth_plane
from lewm.tests.test_joint_floor_registered_controller_development import packets
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual


def equal(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict) or hasattr(a, 'keys'):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def test_cache_exact_reuse_and_output_cannot_be_made_writable():
    d, v = depth_plane(); cache = FrameFloorIndexCache()
    a = cache.index(d, v, [0., 0., 1.])
    equal(a, observed_floor_cell_index(d, v, [0., 0., 1.]))
    assert cache.index(d.copy(), v.copy(), np.array([0., 0., 1.])) is a
    assert cache.counts() == dict(hits=1, misses=1, uncached=0)
    for value in a.values():
        with pytest.raises(ValueError): value.flags.writeable = True
    with pytest.raises(TypeError): a['up'] = np.ones(3)


def test_input_mutation_changes_key_without_corrupting_past_result():
    d, v = depth_plane(); cache = FrameFloorIndexCache()
    old = cache.index(d, v, [0, 0, 1]); snapshot = old['ground_cells'].copy()
    r, c = np.argwhere(v)[len(np.argwhere(v))//2]
    d[r, c] = 0.; v[r, c] = False
    new = cache.index(d, v, [0, 0, 1])
    assert new is not old
    equal(new, observed_floor_cell_index(d, v, [0, 0, 1]))
    np.testing.assert_array_equal(old['ground_cells'], snapshot)
    assert cache.misses == 2


def test_depth_dtype_up_and_camera_values_do_not_alias():
    d, v = depth_plane(); cache = FrameFloorIndexCache()
    old = cache.index(d, v, [0, 0, 1])
    assert cache.index(d.astype(np.float64), v, [0, 0, 1]) is not old
    up = np.array([.005, 0., 1.]); up /= np.linalg.norm(up)
    assert cache.index(d, v, up) is not old
    other, valid = depth_plane(height=-.31)
    assert cache.index(other, valid, [0, 0, 1]) is not old
    assert cache.misses == 4


@pytest.mark.parametrize('fault', ['invalid_nonzero', 'depth_nan', 'up', 'valid_dtype', 'shape'])
def test_bad_input_after_hit_still_reaches_original_rejection(fault):
    d, v = depth_plane(); cache = FrameFloorIndexCache(); cache.index(d, v, [0, 0, 1])
    u = [0., 0., 1.]
    if fault == 'invalid_nonzero': d[0, 0] = 1.; v[0, 0] = False
    elif fault == 'depth_nan': d[0, 0] = np.nan
    elif fault == 'up': u = [0., 0., 2.]
    elif fault == 'valid_dtype': v = v.astype(int)
    else: d = d[:10]
    with pytest.raises(SensorContractError): cache.index(d, v, u)
    assert cache.hits == 0


def test_bounded_entries_fallback_and_observation_lifetime():
    d, v = depth_plane(); cache = FrameFloorIndexCache()
    for i in range(10):
        u = np.array([i*.001, 0., 1.]); u /= np.linalg.norm(u)
        equal(cache.index(d, v, u), observed_floor_cell_index(d, v, u))
    assert len(cache._entries) == 8 and cache.uncached == 2
    cache.close(); assert not cache._entries
    with pytest.raises(ValueError, match='scope already closed'): cache.index(d, v, [0, 0, 1])
    new = FrameFloorIndexCache(); new.index(d, v, [0, 0, 1]); assert new.misses == 1 and new.hits == 0


def test_cached_geometry_preserves_full_outputs_and_retained_patch_bytes():
    d, v = depth_plane(); R = np.eye(3); p = np.zeros(3); h = -.32
    g = FloorFrameGeometry()
    equal(g.floor_coverage(d, v, R, p, h), floor_coverage(d, v, R, p, h))
    equal(g.sampled_floor_patch(d, v, R, p, h, ROWS, COLUMNS), sampled_floor_patch(d, v, R, p, h, ROWS, COLUMNS))
    plane = primary_floor_plane(d, v, R, p, h)
    equal(g.primary_floor_plane(d, v, R, p, h), plane)
    equal(g.confirm_auxiliary_floor(d, v, R, p, h, ROWS, COLUMNS, plane),
          confirm_auxiliary_floor(d, v, R, p, h, ROWS, COLUMNS, plane))
    old, new = RetainedFloorPatches(), RetainedFloorPatches()
    witness = dict(frame=0, measured_ns=1_500_000_000, rgb_sha256='a'*64, depth_sha256='b'*64)
    old.append(d, v, R, p, h, witness); g.append_patch(new, d, v, R, p, h, witness)
    equal(old.frames, new.frames)
    g.close(); equal(old.coverage([[1.5, 0.]]), new.coverage([[1.5, 0.]]))


def test_full_synthetic_controller_equality_and_cache_cleanup_on_failure(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    kwargs = dict(public_mission=dict(goal_initial_body_xy_m=[1., 0.],
        return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    old = JointFloorRegisteredRoundTripController(None, None, **kwargs)
    new = FrameCachedJointFloorRoundTripController(None, None, **kwargs)
    previous = None
    for frame in range(2):
        p, d, a, raw, now = packets(frame, previous)
        old.motion = SimpleNamespace(observe=lambda *args, **kw: deepcopy(raw))
        new.motion = SimpleNamespace(observe=lambda *args, **kw: deepcopy(raw))
        x = old.observe(p, d, None, now_ns=now, auxiliary_depth=a)
        y = new.observe(p, d, None, now_ns=now, auxiliary_depth=a)
        assert x['terminal'] is None, x['failure']
        equal(x, y)
        assert new.mapper.frame_geometry is None and new.memory.frame_geometry is None
        assert new.mapper.last_cache_counts['hits'] >= 7
        equal(old.memory.patches.frames, new.memory.patches.frames)
        equal(old.memory.auxiliary_patches.frames, new.memory.auxiliary_patches.frames)
        previous = raw
    p, d, a, raw, now = packets(2, previous)
    old.memory.failed = new.memory.failed = True
    x = old.observe(p, d, None, now_ns=now, auxiliary_depth=a)
    y = new.observe(p, d, None, now_ns=now, auxiliary_depth=a)
    assert x['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    equal(x, y)
    assert new.mapper.frame_geometry is None and new.memory.frame_geometry is None


DERIVATIVES = json.loads(Path('docs/go2_frame_cached_floor_source_derivatives_2026-09-09.json').read_text())['derivatives']


@pytest.mark.parametrize('entry', DERIVATIVES, ids=lambda e: e['target'])
def test_derivatives_only_route_calculations_through_explicit_frame_cache(entry):
    source = Path(entry['source']).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == entry['source_sha256']
    body = ast.parse(source).body
    for name in entry['node'].split('.'):
        node = next(n for n in body if getattr(n, 'name', None) == name); body = node.body
    expected = ast.get_source_segment(source, node)
    for before, after in entry['replacements']:
        assert before in expected
        expected = expected.replace(before, after)
    if '.' in entry['node']:
        lines = expected.splitlines()
        expected = lines[0]+'\n'+textwrap.dedent('\n'.join(lines[1:]))
        expected = lines[0]+'\n'+textwrap.indent(expected.split('\n', 1)[1], '    ')
    target_file = ('lewm/frame_cached_floor_geometry_development.py' if entry['target'].startswith('FloorFrameGeometry.')
        else 'lewm/frame_cached_floor_map_development.py')
    body = ast.parse(Path(target_file).read_text()).body
    for name in entry['target'].split('.'):
        actual = next(n for n in body if getattr(n, 'name', None) == name); body = actual.body
    expected_node = ast.parse(expected).body[0]
    # Moving a function into a class adds indentation inside multiline docs.
    # Normalize only its docstring; every executable statement stays exact.
    for node in (expected_node, actual):
        doc = ast.get_docstring(node)
        if doc is not None: node.body[0].value.value = doc
    assert ast.dump(expected_node) == ast.dump(actual)
