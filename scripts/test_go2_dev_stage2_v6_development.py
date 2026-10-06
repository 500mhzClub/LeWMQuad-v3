"""Synthetic tests for the stage-2 v6 additions (Andrew, 5 October 2026).

- Placement v2 (lewm/dev_dynamics_patches_v2_development.py) on hand-built layouts: corridor, junction, corner, no
  usable segment; margins, minimum length, 20-40% coverage, full corridor width, determinism.
- Strip geometry: on_patch, signed edge distance, floor marker on a quad mesh.
- C1A (lewm/dev_c1_adaptive_travel_development.py):
  - requested-command rebuild from the decision log;
  - travel ratio against a stand-in command model and tracker poses;
  - median, clip and minimum sample count;
  - the mixin scales C1's XY only and leaves it unchanged when the ratio is 1.
- Scorer bins (scripts/score_go2_dev_patch_edge_prediction_development.py) and the v6 entry's dynamics parser.
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_stage2_v6_development
"""
from contextlib import nullcontext

import numpy as np
import trimesh

from lewm import dev_c1_adaptive_travel_development as c1a
from lewm import dev_dynamics_patches_v2_development as p2

P = p2.PITCH_M


def layout(cells, edges, home, beacon):
    spec = dict(evaluation_layout=dict(pitch_m=P, cells=[list(c) for c in cells], edges=[[list(a), list(b)] for a, b in edges]))
    packet = dict(home_se2_world=[home[0]*P, home[1]*P, 0.], beacon_xy_world=[beacon[0]*P, beacon[1]*P])
    return spec, packet


def chain(cells):
    return list(zip(cells, cells[1:]))


def test_corridor():
    cells = [(i, 0) for i in range(7)]
    spec, packet = layout(cells, chain(cells), (0, 0), (6, 0))
    p = p2.place_patches(spec, packet, seed=1)
    assert p['cell_kinds'] == ['dead_end']+['straight']*5+['dead_end']
    (seg,) = p['usable_segments']
    assert abs(seg['start_m']-(0.5*P+.5)) < 1e-9 and abs(seg['end_m']-(5.5*P-.5)) < 1e-9
    assert len(p['patches']) == 1
    x0, x1, y0, y1 = p['patches'][0]['rect']
    assert x0 >= seg['start_m']-1e-9 and x1 <= seg['end_m']+1e-9 and x1-x0 >= p2.MIN_LENGTH_M-1e-9
    assert abs(y0+P/2) < 1e-9 and abs(y1-P/2) < 1e-9, 'full corridor width'
    assert .20 <= p['coverage'] <= .40, p['coverage']
    assert p == p2.place_patches(spec, packet, seed=1), 'deterministic'


def test_junction_margins():
    cells = [(i, 0) for i in range(7)]+[(3, 1)]
    spec, packet = layout(cells, chain(cells[:7])+[((3, 0), (3, 1))], (0, 0), (6, 0))
    p = p2.place_patches(spec, packet, seed=3)
    assert p['cell_kinds'][3] == 'junction'
    segs = p['usable_segments']
    assert [s['cells'] for s in segs] == [[[1, 0], [2, 0]], [[4, 0], [5, 0]]]
    assert abs(segs[0]['end_m']-(2.5*P-.5)) < 1e-9 and abs(segs[1]['start_m']-(3.5*P+.5)) < 1e-9
    for patch in p['patches']:
        x0, x1 = patch['rect'][:2]
        assert x1 <= 2.5*P-.5+1e-9 or x0 >= 3.5*P+.5-1e-9, 'at least 0.5 m from the junction cell'


def test_corner_and_vertical():
    cells = [(i, 0) for i in range(4)]+[(3, j) for j in range(1, 5)]
    spec, packet = layout(cells, chain(cells), (0, 0), (3, 4))
    p = p2.place_patches(spec, packet, seed=5)
    assert p['cell_kinds'][3] == 'corner'
    first, second = p['usable_segments']
    assert first['axis'] == 0 and abs(first['end_m']-(2.5*P-p2.CORNER_MARGIN_M)) < 1e-9, 'ends at the corner cell, minus the corner margin'
    assert second['axis'] == 1 and abs(second['start_m']-(0.5*P+p2.CORNER_MARGIN_M)) < 1e-9
    for patch in p['patches']:
        assert not p2.on_patch([[3*P, 0.]], [patch['rect']])[0], 'corner cell centre never patched'


def test_no_segment():
    cells = [(0, 0), (1, 0), (1, 1)]
    spec, packet = layout(cells, chain(cells), (0, 0), (1, 1))
    p = p2.place_patches(spec, packet, seed=7)
    assert p['patches'] == [] and p['coverage'] == 0. and 'note' in p


def test_strip_geometry_and_marker():
    rect = [1., 3., -.65, .65]
    assert list(p2.on_patch([[2., 0.], [3.5, 0.], [1., .65]], [rect])) == [True, False, True]
    assert abs(p2.signed_edge_distance([2., 0.], [rect])+.65) < 1e-9
    assert abs(p2.signed_edge_distance([3.5, 0.], [rect])-.5) < 1e-9
    quads = []
    for i in range(32):
        for j in range(-6, 6):
            x, y = i*.125, j*.125
            quads += [[x, y, 0.], [x+.125, y, 0.], [x+.125, y+.125, 0.], [x, y+.125, 0.]]
    vertices = np.asarray(quads)
    faces = np.asarray([[4*k, 4*k+1, 4*k+2] for k in range(len(vertices)//4)])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile([[90, 90, 90, 255]], (len(vertices), 1)).astype(np.uint8)
    marked, count = p2.mark_floor(mesh, [rect])
    centres = vertices.reshape(-1, 4, 3).mean(axis=1)[:, :2]
    assert count == int(p2.on_patch(centres, [rect]).sum()) == 16*10


def stand_in_model():
    features = 24+3+420
    return dict(mean=np.zeros((8, features)), scale=np.ones((8, features)), bias=np.zeros((8, 3)),
                coefficient=np.zeros((8, features, 3)))


def history_packets(now_ns):
    packets = []
    for i in range(4):
        t = now_ns+(i-3)*100_000_000
        measured = t-np.arange(15)[::-1]*100_000_000
        packets.append(dict(sensor_state=dict(decision_ns=t, control=dict(applied_command=dict(
            values=np.tile([.2, 0., 0.], (15, 1)).tolist(), valid=np.ones((15, 3), bool).tolist(),
            measured_ns=measured.tolist(), available_ns=measured.tolist())))))
    return packets


def test_requested_ticks():
    planning = [dict(measured_ns=0, committed_prefix=[[.1, 0, 0]]*3, selection=dict(requested_command=[.2, 0, 0])),
                dict(measured_ns=400_000_000, committed_prefix=[[.2, 0, 0]]*3,
                     selection=dict(requested_command=[0, 0, .5], command_duration_ns=100_000_000))]
    ticks = c1a.requested_ticks(planning, 0, 8)
    assert ticks[:3, 0].tolist() == [.1]*3 and ticks[3, 0] == .2
    assert ticks[4:7, 0].tolist() == [.2]*3 and ticks[7].tolist() == [0, 0, .5]
    assert c1a.requested_ticks(planning, -100_000_000, 2) is None


def poses_along(points):
    return {f: dict(position_initial_body_m=[x, y, 0.], rotation_initial_body_from_current_body=np.eye(3).tolist(),
                    measured_ns=f*100_000_000) for f, (x, y) in points.items()}


def test_travel_ratio():
    model = stand_in_model()
    planning = [dict(measured_ns=0, committed_prefix=[[.2, 0, 0]]*3, selection=dict(requested_command=[.2, 0, 0])),
                dict(measured_ns=400_000_000, committed_prefix=[[.2, 0, 0]]*3, selection=dict(requested_command=[.2, 0, 0]))]
    decisions = [dict(ns=0, frame=10, history=np.zeros(420))]
    poses = poses_along({10: (0., 0.), 18: (.08, 0.)})  # 0.2 m/s commanded for 0.8 s predicts 0.16 m; half tracked
    s = c1a.travel_sample(model, decisions, planning, poses, 800_000_000, 18)
    assert s['horizon_steps'] == 8 and abs(s['predicted_m'][0]-.16) < 1e-9 and abs(s['ratio']-.5) < 1e-9
    assert c1a.current_ratio([s], 800_000_000) == (1., 1), 'one sample is not enough'
    many = [dict(s, ns=800_000_000+k) for k in range(3)]+[dict(s, ns=1, ratio=.05)]
    ratio, used = c1a.current_ratio(many, 3_700_000_000)
    assert used == 3 and abs(ratio-.5) < 1e-9, 'a sample older than 3 s is dropped, the median kept'
    assert c1a.current_ratio([dict(s, ratio=.1)]*3, 800_000_000)[0] == c1a.CLIP[0]


class StandInC1:
    def __init__(self):
        self.pulse_prediction_source = 'command_history'
        self.command_model = stand_in_model()
        self.correction_poses, self.correction_pose_lock, self.planning = {}, nullcontext(), []

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        selected = prediction.copy()
        return selected, dict(command_history_forecast_xy_yaw=selected[..., :3].tolist())


class AdaptiveC1(c1a.AdaptiveTravelMixin, StandInC1):
    pass


class Packet:
    def __init__(self, ns, frame):
        self.measured_ns, self.frame, self.history = ns, frame, history_packets(ns)


def test_mixin():
    controller = AdaptiveC1()
    forecast = np.zeros((6, 8, 5))
    forecast[..., 0], forecast[..., 3] = .1, 1.
    out, receipt = controller._correct_prediction(forecast, Packet(0, 10), None, None)
    assert np.array_equal(out, forecast) and receipt['dev_adaptive_travel']['ratio'] == 1.
    controller.adaptive_samples.extend(dict(ns=0, ratio=.6) for _ in range(3))
    controller.planning = []
    out, receipt = controller._correct_prediction(forecast, Packet(400_000_000, 14), None, None)
    assert np.allclose(out[..., 0], .06) and np.array_equal(out[..., 2:], forecast[..., 2:]), 'XY scaled, yaw untouched'
    assert receipt['controller_variant'] == 'C1A' and np.allclose(np.asarray(receipt['dev_adaptive_forecast_xy_yaw'])[..., 0], .06)


def test_scorer_bins_and_parser():
    from scripts.score_go2_dev_patch_edge_prediction_development import edge_bin
    from scripts.run_go2_dev_mission_pinned_v6_development import parse_dynamics
    assert edge_bin(-.5, -.6) == 'on_patch' and edge_bin(.1, 0.) == 'entry' and edge_bin(.1, .2) == 'exit'
    assert edge_bin(1., .9) == 'approach' and edge_bin(1., 1.1) == 'off_patch' and edge_bin(2., 1.9) == 'off_patch'
    assert parse_dynamics('patches2:0.3:marked') == dict(perturbation='low_friction_patches', mu=.3, marked=True,
                                                         placement_version='v2')
    assert parse_dynamics('patches:0.3:unmarked')['placement_version'] == 'v1'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok', test.__name__)
    print(f'{len(tests)} passed')
