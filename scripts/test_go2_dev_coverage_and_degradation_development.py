"""Synthetic checks for the coverage-rule fix and the forecast degradation (development, 2 October 2026).

Coverage: the forward footprint reaches one coarse cell that is observed as occupied, not floor.
The frozen rule rejects forward and requests no view; with the fix (inside the mixin) forward
passes. A truly unobserved cell is still rejected, and is targeted for a view.
Degradation: scale halves displacement and heading change exactly; noise hits its median
700-ms targets (E mm displacement, E/10 degrees heading) and is reproducible per frame.
"""
import math
import types

import numpy as np

from lewm import coverage_translation_view_development as coverage_rule
from lewm import dev_harness_fixes_development as fixes
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.observed_floor_waypoint_development import CELL_M


def selection():
    rows = [dict(action=a, utility_m=1.0 if a == 'forward' else 0.1*(i+1)) for i, a in enumerate(ACTIONS)]
    return dict(action='forward', action_index=ACTIONS.index('forward'), candidates=rows,
                planned_stopping_projection=dict(candidates=[dict(action=a, projection_clear=True) for a in ACTIONS]),
                memory_forecast_candidates=[dict(action=a, nominal_predicted_path_clear=True, reserve_recovery_path_clear=True) for a in ACTIONS])


def prediction():
    p = np.zeros((6, 8, 5))
    p[..., 3] = 1.0
    p[ACTIONS.index('forward'), :, 0] = np.linspace(.025, .2, 8)
    return p


def snapshot(special, occupied):
    cells = {(i, j) for i in range(-30, 31) for j in range(-30, 31) if math.hypot(i, j)*CELL_M <= 1.2}
    cells.discard(special)
    return types.SimpleNamespace(floor=frozenset(cells), occupied=frozenset({special} if occupied else ()))


class Base:
    def __init__(self, snap):
        self.snap = snap

    def _select_clear_prediction(self, selected, prediction, snap, position, rotation):
        return coverage_rule.filter_translation(selected, prediction, snap, position, rotation)


class Fixed(fixes.CoverageObservedObstacleMixin, Base):
    pass


def check_coverage():
    special = (12, 0)  # centre 0.625 m ahead: inside the 0.48-m swept footprint of the 0.2-m forward path only
    args = (prediction(), np.zeros(3), np.eye(3))
    frozen, target = coverage_rule.filter_translation(selection(), args[0], snapshot(special, True), *args[1:])
    assert frozen['translation_footprint_coverage']['rejected'] and target is None, 'frozen rule should reject with no view target'
    fixed, target = Fixed(None)._select_clear_prediction(selection(), args[0], snapshot(special, True), *args[1:])
    assert not fixed['translation_footprint_coverage']['rejected'] and target is None, 'fix should pass an observed obstacle cell'
    unknown, target = Fixed(None)._select_clear_prediction(selection(), args[0], snapshot(special, False), *args[1:])
    assert unknown['translation_footprint_coverage']['rejected'] and target == special, 'an unobserved cell must still be rejected and targeted'
    outside, _ = Base(None)._select_clear_prediction(selection(), args[0], snapshot(special, True), *args[1:])
    assert outside['translation_footprint_coverage']['rejected'], 'outside the mixin the frozen behaviour is unchanged'
    print('coverage: observed obstacle cell passes with the fix; unobserved cell still rejected and targeted; frozen behaviour unchanged outside the mixin')


class Source:
    def _correct_prediction(self, prediction, packet, evidence, prefix):
        return prediction, {}


def check_degradation():
    base = prediction()
    yaw = np.zeros((6, 8))
    base[..., 2], base[..., 3] = np.sin(.4*np.ones((6, 8))), np.cos(.4*np.ones((6, 8)))
    scaled = type('S', (fixes.degradation_mixin('scale:0.5'), Source), {})()
    p, c = scaled._correct_prediction(base.copy(), types.SimpleNamespace(frame=7), None, None)
    assert np.allclose(p[..., :2], base[..., :2]*.5) and np.allclose(np.arctan2(p[..., 2], p[..., 3]), .2)
    for e in (10., 40.):
        noisy = type('N', (fixes.degradation_mixin(f'noise:{e}'), Source), {})()
        xy, hd = [], []
        for frame in range(3000):
            q, c = noisy._correct_prediction(base.copy(), types.SimpleNamespace(frame=frame), None, None)
            xy.append(np.linalg.norm(q[0, 6, :2]-base[0, 6, :2]))
            hd.append(abs(math.degrees(math.atan2(q[0, 6, 2], q[0, 6, 3])-.4)))
        assert abs(np.median(xy)*1000/e-1) < .06 and abs(np.median(hd)/(e/10)-1) < .06, (np.median(xy), np.median(hd))
        again, _ = noisy._correct_prediction(base.copy(), types.SimpleNamespace(frame=5), None, None)
        first, _ = noisy._correct_prediction(base.copy(), types.SimpleNamespace(frame=5), None, None)
        assert np.array_equal(again, first)
        print(f'noise:{e:g}: median 700-ms error {np.median(xy)*1000:.1f} mm, heading {np.median(hd):.2f} deg; reproducible per frame')
    print('scale:0.5 halves displacement and heading change; degraded forecast logged:', 'dev_degraded_forecast_xy_yaw' in c)


if __name__ == '__main__':
    check_coverage()
    check_degradation()
