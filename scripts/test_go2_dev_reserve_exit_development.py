"""Synthetic tests for the next harness version (reserve exit, C2 aligned clearance; lewm/dev_harness_reserve_exit_development.py).

Cells are 1-cm squares (i, j) covering [i, i+1) x [j, j+1) cm; the robot starts at the origin facing +x.
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_reserve_exit_development
"""
import copy
import math

import numpy as np

from lewm import clearance_lookahead_development as lookahead
from lewm import dev_harness_reserve_exit_development as v5
from lewm import persistent_visual_baselines_development as c2
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.memory_forecast_clearance_development import select_clear_prediction as frozen

RNG = np.random.default_rng(20261003)


def wall_x(at_cm, span=range(-60, 61)):
    """A wall of cells whose near face is at x = at_cm (cm); behind the robot if negative."""
    i = at_cm if at_cm > 0 else at_cm-1
    return {(i, j) for j in span}


def selected(utilities):
    return dict(action='hold', candidates=[dict(action=a, utility_m=float(u)) for a, u in zip(ACTIONS, utilities)])


def straight(dx_per_tick):
    """Prediction (6, 8, 3): every translation moves along x by dx per tick after the 3-tick prefix; turns stay put."""
    p = np.zeros((6, 8, 3))
    for i, a in enumerate(ACTIONS):
        if a in v5.TRANSLATIONS:
            p[i, 3:, 0] = dx_per_tick*np.arange(1, 6)
    return p


def call(sel, pred, cells, exit_on):
    token = v5._EXIT.set(exit_on)
    try:
        return lookahead.select_clear_prediction(copy.deepcopy(sel), pred, cells, np.zeros(3), np.eye(3),
                                                 translation_reserve_m=.03, reserve_recovery=True)
    finally:
        v5._EXIT.reset(token)


def test_exit_rule():
    assert v5.exit_clear([.44, .44, .44, .45, .46, .47, .48, .49])
    assert v5.exit_clear([.44, .4395, .44, .44, .44, .44, .44, .441])          # within the 1-mm tolerance
    assert not v5.exit_clear([.44, .44, .44, .44, .44, .44, .44, .44])          # does not end higher
    assert not v5.exit_clear([.44, .43, .45, .46, .47, .48, .49, .50])          # dips by 1 cm
    assert not v5.exit_clear([.44, .4392, .4384, .4376, .4368, .45, .46, .47])  # creeping loss beyond 1 mm in total
    assert not v5.exit_clear([.44]*7)
    assert not v5.exit_clear([.44, None, .45, .46, .47, .48, .49, .50])


def test_frozen_when_off():
    for _ in range(300):
        cells = wall_x(int(RNG.integers(44, 70))) | wall_x(-int(RNG.integers(44, 70)))
        pred = RNG.normal(0, .02, (6, 8, 3))
        sel = selected(RNG.normal(0, .05, 6))
        assert call(sel, pred, cells, False) == frozen(copy.deepcopy(sel), pred, cells, np.zeros(3), np.eye(3),
                                                       translation_reserve_m=.03, reserve_recovery=True)


def test_same_action_when_no_exit_applies():
    same = 0
    for _ in range(300):
        cells = wall_x(int(RNG.integers(44, 70))) | wall_x(-int(RNG.integers(44, 70)))
        pred = RNG.normal(0, .02, (6, 8, 3))
        sel = selected(RNG.normal(0, .05, 6))
        on = call(sel, pred, cells, True)
        off = call(sel, pred, cells, False)
        if not on['reserve_exit_candidates']:
            assert (on['action'], on['memory_forecast_status']) == (off['action'], off['memory_forecast_status'])
            same += 1
    assert same > 100


def test_exit_inside_disc_moving_away():
    cells = wall_x(-44)                     # wall 0.44 m behind: inside the 0.45-m disc
    pred = straight(.012)                   # translations move 6 cm away over the commit
    sel = selected([0., .03, .02, .01, -.01, -.02])
    off, on = call(sel, pred, cells, False), call(sel, pred, cells, True)
    assert off['action'] == 'hold' and off['memory_forecast_status'] == 'NO_CLEAR_CANDIDATE_ZERO_REQUESTED'
    assert on['action'] == 'forward' and on['selected_reserve_exit']
    assert set(on['reserve_exit_candidates']) == set(v5.TRANSLATIONS)
    turns = [r for r in on['memory_forecast_candidates'] if r['action'] in ('left_turn', 'right_turn')]
    assert not any(r['reserve_exit_path_clear'] for r in turns)


def test_exit_inside_reserve_small_gain():
    cells = wall_x(-46)                     # 0.46 m: inside the reserve, outside the disc
    pred = straight(.002)                   # ends at 0.47 m: frozen recovery needs > 0.48 m
    sel = selected([0., .03, .02, .01, -.01, -.02])
    assert call(sel, pred, cells, False)['action'] != 'forward'
    assert call(sel, pred, cells, True)['action'] == 'forward'


def test_no_exit_toward_wall():
    cells = wall_x(44)                      # wall 0.44 m ahead
    pred = straight(.012)
    sel = selected([0., .03, .02, .01, -.01, -.02])
    on = call(sel, pred, cells, True)
    assert on['action'] == 'hold' and not on['reserve_exit_candidates']


def test_full_reserve_unchanged():
    cells = wall_x(90) | wall_x(-90)
    pred = straight(.012)
    sel = selected([0., .03, .02, .01, -.01, -.02])
    on, off = call(sel, pred, cells, True), call(sel, pred, cells, False)
    assert on['action'] == off['action'] == 'forward' and not on['selected_reserve_exit']


def test_nominal_prediction():
    p = v5.nominal_prediction([[0., 0., 0.]]*3)
    f, t = ACTIONS.index('forward'), ACTIONS.index('left_turn')
    assert np.allclose(p[f, :3], 0) and math.isclose(p[f, 5, 0], .06, abs_tol=1e-9) and math.isclose(p[f, 7, 0], .06, abs_tol=1e-9)
    assert np.allclose(p[t, :, :2], 0) and math.isclose(p[t, 7, 2], .45*.3, abs_tol=1e-9)
    assert np.allclose(p[ACTIONS.index('hold')], 0)
    q = v5.nominal_prediction([[.2, 0., 0.]]*3)
    assert math.isclose(q[f, 2, 0], .06, abs_tol=1e-9)    # the committed prefix moves every candidate
    try:
        v5.nominal_prediction([[0., 0., 0.]]*2)
        raise AssertionError('short prefix accepted')
    except ValueError:
        pass


def reactive(goal, cells, clearance, prefix=((0., 0., 0.),)*3, scan_error=None):
    token = v5._C2_CHECK.set(dict(prefix=[list(c) for c in prefix], cells=cells, position=np.zeros(3), rotation=np.eye(3)))
    try:
        return c2.select_reactive(goal, scan_error=scan_error, current_clearance_m=clearance)
    finally:
        v5._C2_CHECK.reset(token)


def test_c2_frozen_without_context():
    for clearance in (None, .30, .44, .46, .9):
        for goal in ((1., 0.), (0., 1.), (-1., .1)):
            assert c2.select_reactive(goal, scan_error=None, current_clearance_m=clearance) == v5._frozen_reactive(goal, scan_error=None, current_clearance_m=clearance)


def test_c2_inside_disc():
    cells = wall_x(-44)
    old = v5._frozen_reactive((1., 0.), scan_error=None, current_clearance_m=.44)
    assert old['action'] == 'hold' and not any(r['eligible'] for r in old['candidates'])
    new = reactive((1., 0.), cells, .44)
    assert new['action'] == 'forward' and new['all_actions_block_removed'] and new['c2_nominal_path_check']['selected_reserve_exit']
    assert new['current_stored_clearance_m'] == .44 and new['current_nominal_disk_clear'] is False
    behind = reactive((-1., .05), cells, .44)          # target behind: turns blocked inside the disc, an arc leaves
    assert behind['action'] in ('left_arc', 'right_arc')
    ahead = reactive((1., 0.), wall_x(44), .44)        # facing the wall inside the disc: nothing passes
    assert ahead['action'] == 'hold' and not any(r['eligible'] for r in ahead['candidates'])


def test_c2_clear_space_unchanged():
    cells = wall_x(150) | wall_x(-150)
    for goal in ((1., 0.), (.5, .5), (0., -1.), (-1., .2)):
        new = reactive(goal, cells, 1.5)
        old = v5._frozen_reactive(goal, scan_error=None, current_clearance_m=1.5)
        assert new['action'] == old['action'], goal
    scan = reactive((0., 0.), cells, 1.5, scan_error=.8)
    assert scan['action'] == v5._frozen_reactive((0., 0.), scan_error=.8, current_clearance_m=1.5)['action']


def test_c2_wall_ahead_outside_disc_blocks_forward():
    new = reactive((1., 0.), wall_x(47), .47)           # in the reserve facing the wall: forward would lose clearance
    assert new['action'] != 'forward'
    assert not next(r for r in new['candidates'] if r['action'] == 'forward')['eligible']


def test_composition():
    from lewm.dev_harness_fixes_development import compose, fixes_for
    from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    for arm, base in (('C1', owner.source.DenseNavigationRuntime), ('C2', owner.source.DenseReactiveNavigationRuntime)):
        mix = compose(fixes_for('off'), CompletedSupportRuntimeMixin, extra=v5.mixins_for(arm))
        runtime = type('Checked', (mix, base), {})
        owners = [k.__name__ for k in runtime.__mro__ if '_select_clear_prediction' in k.__dict__]
        assert owners[0] == 'ReserveExitMixin', owners
        assert 'CoverageObservedObstacleMixin' in owners and 'ReserveRecoveryLookaheadRuntime' in owners
        if arm == 'C2':
            assert owners[1] == 'C2NominalPathCheckMixin' and 'ReactiveFeedbackMixin' in owners
            assert [k.__name__ for k in runtime.__mro__ if '_select_action' in k.__dict__][0] == 'C2NominalPathCheckMixin'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok', test.__name__)
    print(f'{len(tests)} passed')
