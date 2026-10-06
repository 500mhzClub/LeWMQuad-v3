"""Synthetic tests for C1A v2 (lewm/dev_c1_adaptive_travel_v2_development.py; parameters fixed by Andrew, 5 October 2026).

The stand-in command model has zero corrections, so C1's forecast is the nominal integration of the requested commands.
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_c1a_v2_development
"""
from contextlib import nullcontext

import numpy as np

from lewm import dev_c1_adaptive_travel_v2_development as c1a
from scripts.test_go2_dev_stage2_v6_development import Packet, stand_in_model


def yaw_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]


def pose(x, y, theta):
    return dict(position_initial_body_m=[x, y, 0.], rotation_initial_body_from_current_body=yaw_rotation(theta))


def plan(command):
    return [dict(measured_ns=0, committed_prefix=[command]*3, selection=dict(requested_command=command)),
            dict(measured_ns=400_000_000, committed_prefix=[command]*3, selection=dict(requested_command=command))]


def test_translation_and_rotation_samples():
    model, decisions = stand_in_model(), [dict(ns=0, frame=10, history=np.zeros(420))]
    # forward 0.2 m/s for 0.8 s predicts 0.16 m and no turn: translation sample only
    t, r = c1a.motion_samples(model, decisions, plan([.2, 0., 0.]), {10: pose(0, 0, 0), 18: pose(.12, 0, 0)}, 800_000_000, 18)
    assert abs(t['ratio']-.75) < 1e-9 and r is None
    # turn in place 0.5 rad/s for 0.8 s predicts 0.4 rad: rotation sample only
    t, r = c1a.motion_samples(model, decisions, plan([0., 0., .5]), {10: pose(0, 0, 0), 18: pose(0, 0, .2)}, 800_000_000, 18)
    assert t is None and abs(r['ratio']-.5) < 1e-9
    # all hold: no samples at all
    assert c1a.motion_samples(model, decisions, plan([0., 0., 0.]), {10: pose(0, 0, 0), 18: pose(.05, 0, .1)},
                              800_000_000, 18) == (None, None)
    # tiny commanded motion (under 2 cm and 2 degrees) gives no samples
    t, r = c1a.motion_samples(model, decisions, plan([.02, 0., .03]), {10: pose(0, 0, 0), 18: pose(.01, 0, 0)}, 800_000_000, 18)
    assert t is None and r is None


def test_ratio_rules():
    s = [dict(ns=1_000_000_000+k, ratio=v) for k, v in enumerate((.6, .7, .8))]
    assert c1a.current_ratio(s[:1], 1_000_000_000) == (1., 1), 'start at 1.0 until two samples'
    ratio, used = c1a.current_ratio(s, 1_500_000_000)
    assert used == 3 and abs(ratio-.7) < 1e-9, 'median'
    assert c1a.current_ratio(s, 4_100_000_000) == (1., 0), '3-s window'
    assert c1a.current_ratio([dict(ns=0, ratio=5.)]*2, 0)[0] == 1.5 and c1a.current_ratio([dict(ns=0, ratio=.01)]*2, 0)[0] == .3


def test_adapt_scales_xy_and_yaw_separately():
    f = np.zeros((6, 8, 5))
    f[..., 0], f[..., 2], f[..., 3] = .1, np.sin(.4), np.cos(.4)
    out, yaw = c1a.adapt(f, .5, .25)
    assert np.allclose(out[..., 0], .05) and np.allclose(yaw, .1) and np.allclose(out[..., 2], np.sin(.1))
    assert np.array_equal(out[..., 4], f[..., 4])
    same, _ = c1a.adapt(f, 1., 1.)
    assert np.allclose(same, f)


class StandInC1:
    def __init__(self):
        self.pulse_prediction_source, self.command_model = 'command_history', stand_in_model()
        self.correction_poses, self.correction_pose_lock, self.planning = {}, nullcontext(), []

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        return prediction.copy(), dict(command_history_forecast_xy_yaw=prediction[..., :3].tolist())


class C1A(c1a.AdaptiveTravelMixin, StandInC1):
    pass


def test_mixin():
    controller = C1A()
    f = np.zeros((6, 8, 5))
    f[..., 0], f[..., 2], f[..., 3] = .1, np.sin(.4), np.cos(.4)
    out, receipt = controller._correct_prediction(f, Packet(0, 10), None, None)
    assert np.allclose(out, f) and receipt['dev_adaptive_travel']['translation_ratio'] == 1.
    controller.adaptive_translation.extend(dict(ns=0, ratio=.6) for _ in range(2))
    controller.adaptive_rotation.extend(dict(ns=0, ratio=.5) for _ in range(2))
    out, receipt = controller._correct_prediction(f, Packet(400_000_000, 14), None, None)
    travel = receipt['dev_adaptive_travel']
    assert travel['translation_ratio'] == .6 and travel['rotation_ratio'] == .5 and travel['version'] == 'C1A-v2'
    assert np.allclose(out[..., 0], .06) and np.allclose(np.asarray(receipt['dev_adaptive_forecast_xy_yaw'])[..., 2], .2)


def test_v7_entry_uses_v2():
    text = open('scripts/run_go2_dev_mission_pinned_v7_development.py').read()
    assert 'from lewm.dev_c1_adaptive_travel_v2_development import AdaptiveTravelMixin' in text
    assert 'dev_c1_adaptive_travel_development import' not in text


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
        print('ok', test.__name__)
    print(f'{len(tests)} passed')
