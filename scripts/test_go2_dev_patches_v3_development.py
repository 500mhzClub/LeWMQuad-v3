"""Synthetic tests for patches v3 (lewm/dev_dynamics_patches_v3_development.py): tinted marker, raised session guard, v9 parser.
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_patches_v3_development
"""
import numpy as np
import trimesh

from lewm import dev_dynamics_patches_v3_development as p3
from scripts import novel_maze_round_trip_physical_session_development as session_module


def floor(rng):
    quads, colours = [], []
    for i in range(32):
        for j in range(-6, 6):
            x, y = i*.125, j*.125
            quads += [[x, y, 0.], [x+.125, y, 0.], [x+.125, y+.125, 0.], [x, y+.125, 0.]]
            grey = int(rng.integers(60, 220))
            colours += [[grey, grey, grey, 255]]*4
    vertices = np.asarray(quads)
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[4*k, 4*k+1, 4*k+2] for k in range(len(vertices)//4)], process=False)
    mesh.visual.vertex_colors = np.asarray(colours, np.uint8)
    return mesh


def test_tinted_marker_keeps_texture():
    mesh = floor(np.random.default_rng(0))
    rect = [1., 3., -.65, .65]
    marked, count = p3.mark_floor(mesh, [rect])
    before, after = np.asarray(mesh.visual.vertex_colors)[::4, :3].astype(float), np.asarray(marked.visual.vertex_colors)[::4, :3].astype(float)
    inside = p3.on_patch(np.asarray(mesh.vertices).reshape(-1, 4, 3).mean(axis=1)[:, :2], [rect])
    assert count == int(inside.sum()) == 160
    assert np.array_equal(before[~inside], after[~inside]), 'outside the strip unchanged'
    assert np.allclose(after[inside], np.rint(before[inside]*np.asarray(p3.TINT)), atol=.5)
    assert np.std(after[inside, 2]) > .9*np.std(before[inside, 2]), 'quad-to-quad contrast kept (blue channel unscaled)'
    assert (after[inside, 2] > after[inside, 0]).all(), 'reads blue'


class FakeSession:
    def __init__(self, speed, contacts=(), xy=(0., 0.)):
        self.guard, self.guard_rows, self.samples, self.packets = dict(fake=list(contacts)), [], [0], [dict(a=np.zeros((1, 1)))]
        self.speed, self.xy, self.inner_guard_seen = speed, xy, []

    def _sample(self, requested, applied, timestamp_s):
        self.inner_guard_seen.append(self.guard)
        return dict(base_twist_world=np.array([self.speed, 0., 0., 0., 0., 0.]), base_pose_world=np.array([*self.xy, .3, 0, 0, 0, 1]))


def run(speed, contacts=(), xy=(0., 0.)):
    s = p3.raise_speed_guard(FakeSession(speed, contacts, xy))
    try:
        s._sample(None, None, 0.)
        return s, None
    except session_module.PhysicalStop as stop:
        return s, stop


def test_raised_guard():
    original = session_module.nonfoot_ground_contact_indices
    session_module.nonfoot_ground_contact_indices = lambda packet, **guard: guard['fake']
    try:
        s, stop = run(.35)
        assert stop is None and s.guard_rows[-1]['speed_limit_m_s'] == .40 and s.inner_guard_seen == [None]
        assert s.guard == dict(fake=[]), 'guard restored after the sample'
        assert run(.45)[1] is not None
        assert run(.1, contacts=[3])[1] is not None, 'non-foot ground contact still stops'
        assert run(.1, xy=(9., 0.))[1] is not None, 'leaving the domain still stops'
    finally:
        session_module.nonfoot_ground_contact_indices = original


def test_v9_parser():
    from scripts.run_go2_dev_mission_pinned_v9_development import parse_dynamics
    d = parse_dynamics('patches3:0.3:unmarked')
    assert d == dict(perturbation='low_friction_patches', mu=.3, marked=False, placement_version='v2', patches_version='v3')
    assert parse_dynamics('patches2:0.3:marked')['patches_version'] == 'v2'


if __name__ == '__main__':
    for test in (test_tinted_marker_keeps_texture, test_raised_guard, test_v9_parser):
        test()
        print('ok', test.__name__)
    print('3 passed')
