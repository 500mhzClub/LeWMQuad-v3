"""Synthetic tests for C1A v3 (lewm/dev_c1_adaptive_travel_v3_development.py): per-movement-type ratios.
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_c1a_v3_development
"""
from contextlib import nullcontext

import numpy as np

from lewm import dev_c1_adaptive_travel_v3_development as c1a
from scripts.test_go2_dev_c1a_v2_development import plan, pose
from scripts.test_go2_dev_stage2_v6_development import Packet, stand_in_model


def test_sample_types():
    model, decisions = stand_in_model(), [dict(ns=1_500_000_000, frame=25, history=np.zeros(420))]
    log = [dict(r, measured_ns=r['measured_ns']+1_500_000_000) for r in plan([.2, 0., 0.])]
    poses = {25: pose(0, 0, 0), 33: pose(.12, 0, 0)}
    kind, t, r = c1a.motion_samples(model, decisions, log, poses, 2_300_000_000, 33)
    assert kind == 'rest_start' and abs(t['ratio']-.75) < 1e-9 and t['movement'] == 'rest_start'
    log = [dict(measured_ns=0, committed_prefix=[[.2, 0., 0.]]*3, selection=dict(requested_command=[.2, 0., 0.]))]+log
    kind, t, r = c1a.motion_samples(model, decisions, log, poses, 2_300_000_000, 33)
    assert kind == 'cruise', 'forward after forward history is cruise'
    log = plan([0., 0., .5])  # two decision records: the turn is requested for the whole 800 ms (0.4 rad predicted)
    kind, t, r = c1a.motion_samples(model, [dict(ns=0, frame=10, history=np.zeros(420))], log,
                                    {10: pose(0, 0, 0), 18: pose(0, 0, .2)}, 800_000_000, 18)
    assert kind == 'turn' and t is None and abs(r['ratio']-.5) < 1e-9


def test_adapt():
    f = np.zeros((6, 8, 5))
    f[..., 0], f[..., 2], f[..., 3] = .1, np.sin(.4), np.cos(.4)
    out, yaw = c1a.adapt(f, ['cruise', 'turn', 'hold', 'switch', 'arc_steady', 'rest_start'],
                         dict(cruise=.5, turn=1., switch=.8, arc_steady=1.), dict(cruise=1., turn=.25, switch=1., arc_steady=1.))
    assert np.allclose(out[0, :, 0], .05) and np.allclose(out[3, :, 0], .08), 'XY scaled by the candidate type'
    assert np.allclose(yaw[1], .1) and np.allclose(out[2], f[2]) and np.allclose(out[5], f[5]), 'turn yaw scaled; hold, rest start untouched'


def test_candidate_types_exact():
    from lewm.geometry_progress_pilot_development import ACTIONS
    past = np.tile([.2, 0., 0.], (15, 1))
    types = dict(zip(ACTIONS, c1a.candidate_types([[.2, 0., 0.]]*3, False, past)))
    assert types['forward'] == 'cruise' and types['hold'] == 'switch' and types['left_turn'] == 'switch'
    still = dict(zip(ACTIONS, c1a.candidate_types([[0., 0., 0.]]*3, False, np.zeros((15, 3)))))
    assert still['hold'] == 'hold' and still['left_turn'] == 'turn' and still['forward'] == 'rest_start', 'no forward motion is a turn'
    spinning = dict(zip(ACTIONS, c1a.candidate_types([[0., 0., .5]]*3, False, np.tile([0., 0., .5], (15, 1)))))
    assert spinning['left_turn'] == 'turn'


class StandInC1:
    def __init__(self):
        self.pulse_prediction_source, self.command_model = 'command_history', stand_in_model()
        self.correction_poses, self.correction_pose_lock, self.planning = {}, nullcontext(), []
        self.planning_translation_pulse = False

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        return prediction.copy(), dict(command_history_forecast_xy_yaw=prediction[..., :3].tolist())


class C1A(c1a.AdaptiveTravelMixin, StandInC1):
    pass


def test_mixin():
    from lewm.geometry_progress_pilot_development import ACTIONS
    controller = C1A()
    f = np.zeros((6, 8, 5))
    f[..., 0], f[..., 2], f[..., 3] = .1, np.sin(.4), np.cos(.4)
    out, receipt = controller._correct_prediction(f, Packet(0, 10), None, [[0., 0., 0.]]*3)
    assert np.allclose(out, f) and receipt['dev_adaptive_travel']['version'] == 'C1A-v3'
    controller.adaptive_translation['rest_start'].extend(dict(ns=0, ratio=.5) for _ in range(3))
    controller.adaptive_translation['turn'].extend(dict(ns=0, ratio=.6) for _ in range(3))
    out, receipt = controller._correct_prediction(f, Packet(400_000_000, 14), None, [[0., 0., 0.]]*3)
    kinds = receipt['dev_adaptive_travel']['candidate_movement']
    assert kinds[ACTIONS.index('forward')] == 'rest_start' and np.allclose(out[ACTIONS.index('forward'), :, 0], .1), 'rest start not adapted'
    assert kinds[ACTIONS.index('left_turn')] == 'turn' and np.allclose(out[ACTIONS.index('left_turn'), :, 0], .06), 'turn candidate uses the turn ratio'


if __name__ == '__main__':
    tests = [test_sample_types, test_adapt, test_candidate_types_exact, test_mixin]
    for test in tests:
        test()
        print('ok', test.__name__)
    print(f'{len(tests)} passed')
