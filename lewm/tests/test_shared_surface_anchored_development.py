"""Full recovery decisions, public ownership and original failure behavior."""
import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import FunctionType
import pytest
from lewm import shared_surface_anchored_selection_development as candidate
from lewm.shared_surface_anchored_controller_development import (
    SharedSurfaceAnchoredController, SharedSurfaceAnchoredSelector, FLAG)
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)
from lewm.tests.test_receipt_copied_anchored_development import SOURCE_SHA256
from lewm.tests.test_residual_anchored_continuation_development import fixture
from lewm.tests.test_residual_hold_feasibility_development import fixture as hold_fixture
from lewm.tests.test_residual_first_interval_feasibility_development import NOW


def apply(s, r, m):
    return candidate.reconsider_with_shared_surface(s, r, m, object(), now_ns=NOW)


def test_bound_original_calculations_and_selector_code_are_unchanged():
    for name, expected in SOURCE_SHA256.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    old = ast.parse(textwrap.dedent(inspect.getsource(ResidualAnchoredContinuationSelector.choose)))
    new = ast.parse(textwrap.dedent(inspect.getsource(SharedSurfaceAnchoredSelector.choose)))
    assert ast.dump(old) == ast.dump(new)


def test_only_private_copy_and_function_bindings_change():
    workspace = candidate.SurfaceReceiptWorkspace()
    snapshots = {f: f.__globals__.copy() for f in candidate.ORIGINAL_FUNCTIONS}
    clones = candidate.isolated_selection_functions(workspace)
    assert len(clones) == 6
    for original, copied in clones.items():
        assert copied.__code__ is original.__code__
        assert copied.__globals__ is not original.__globals__
        assert copied.__defaults__ is original.__defaults__
        assert copied.__kwdefaults__ == original.__kwdefaults__
        assert copied.__closure__ is original.__closure__ is None
        assert copied.__globals__.keys() == original.__globals__.keys()
        for name, value in original.__globals__.items():
            expected = workspace.copy if name == 'deepcopy' else (
                clones[value] if isinstance(value, FunctionType) and value in clones else value)
            assert copied.__globals__[name] == expected
            assert snapshots[original][name] is value
    with pytest.raises(TypeError):
        clones[candidate.reconsider_anchored_continuation] = None


def test_borrowed_receipts_are_detached_at_the_public_boundary():
    workspace = candidate.SurfaceReceiptWorkspace()
    surface = [{'possible_intersection': False, 'witness': [1., 2.]}]
    original = dict(surface_checks=surface, candidates=[{'utility_m': 1.}])
    intermediate = workspace.copy(original)
    assert intermediate['surface_checks'] is surface
    assert intermediate['candidates'] is not original['candidates']
    second = workspace.copy(intermediate)
    second['candidates'][0]['utility_m'] = 9.
    result = workspace.finish(second, original)
    assert result == second and result['surface_checks'] is not surface
    result['surface_checks'][0]['witness'][0] = 7.
    assert surface[0]['witness'][0] == 1.
    assert original['candidates'][0]['utility_m'] == 1.


@pytest.mark.parametrize('alias_kind', ['candidate', 'other_field', 'surface_root'])
def test_cross_field_aliases_use_standard_copy(alias_kind):
    row = {'utility_m': 1., 'witness': [1., 2.]}
    surface = [row]
    original = dict(surface_checks=surface, candidates=[{'utility_m': 0.}])
    if alias_kind == 'candidate': original['candidates'] = [row]
    elif alias_kind == 'other_field': original['other'] = row['witness']
    else: original['other'] = surface
    w = candidate.SurfaceReceiptWorkspace(); result = w.copy(original)
    assert not w.borrowed and result == deepcopy(original)
    if alias_kind == 'candidate':
        result['candidates'][0]['utility_m'] = 5.
        assert result['surface_checks'][0]['utility_m'] == 5.
        assert original['surface_checks'][0]['utility_m'] == 1.
    elif alias_kind == 'other_field':
        assert result['other'] is result['surface_checks'][0]['witness']
    else: assert result['other'] is result['surface_checks']


def test_internal_receipt_aliases_remain_aliases_after_detachment():
    row = {'witness': [1., 2.]}
    original = dict(surface_checks=[row, row], candidates=[])
    w = candidate.SurfaceReceiptWorkspace(); copied = w.copy(original)
    assert w.borrowed
    result = w.finish(copied, original)
    assert result['surface_checks'][0] is result['surface_checks'][1]
    assert result['surface_checks'][0] is not row


@pytest.mark.parametrize('location', ['surface', 'other'])
def test_cycles_fall_back_without_recursing_forever(location):
    cycle = []; cycle.append(cycle)
    s = dict(surface_checks=[{}], candidates=[])
    if location == 'surface': s['surface_checks'][0]['cycle'] = cycle
    else: s['cycle'] = cycle
    w = candidate.SurfaceReceiptWorkspace(); copied = w.copy(s)
    assert not w.borrowed
    c = copied['surface_checks'][0]['cycle'] if location == 'surface' else copied['cycle']
    assert c is c[0] and c is not cycle


def test_custom_copying_is_not_suppressed_or_executed_twice():
    class Counting:
        calls = 0
        def __deepcopy__(self, memo):
            type(self).calls += 1
            return Counting()
    obj = Counting(); s = dict(surface_checks=[{}], custom=obj)
    w = candidate.SurfaceReceiptWorkspace(); result = w.copy(s)
    assert not w.borrowed and Counting.calls == 1
    assert w.finish(result, s) is result and Counting.calls == 1
    assert result['custom'] is not obj


@pytest.mark.parametrize('make_fixture', [fixture, hold_fixture])
def test_complete_recovery_and_public_mutation_isolation(make_fixture):
    s, r, m = make_fixture(); before = deepcopy(s)
    expected = candidate.reconsider_anchored_continuation(s, r, m, object(), now_ns=NOW)
    result = apply(s, r, m)
    assert result == expected and result['action'] == 'forward' and s == before
    result['surface_checks'][0]['possible_intersection'] = True
    result['candidates'][0]['utility_m'] = 123.
    name = 'residual_anchored_continuation' if 'residual_anchored_continuation' in result else 'residual_hold_feasibility'
    result[name]['corrected_surface_checks'][0]['possible_intersection'] = True
    result[name]['corrected_nominal_path_checks'][0]['segments'][0]['radius_m'] = 9.
    assert s == before
    assert expected['surface_checks'][0]['possible_intersection'] is False
    assert apply(s, r, m) == expected


@pytest.mark.parametrize('veto', ['later_segment', 'contact_cost', 'original_surface', 'corrected_surface', 'phase', 'tie'])
def test_original_gate_rejections_preserve_input_identity(veto):
    s, r, m = fixture(late_x=.02 if veto == 'later_segment' else .01, expensive=veto == 'contact_cost')
    if veto == 'original_surface':
        for row in s['surface_checks'][1:]: row['possible_intersection'] = True
    elif veto == 'corrected_surface': m.surface.block = True
    elif veto == 'phase': s['phase_allowed_actions'] = ['hold']
    elif veto == 'tie':
        for row in s['candidates'][1:]: row['utility_m'] = s['candidates'][0]['utility_m']
    before = deepcopy(s)
    assert candidate.reconsider_anchored_continuation(s, r, m, object(), now_ns=NOW) is s
    assert apply(s, r, m) is s and s == before


@pytest.mark.parametrize('fault', ['future', 'bias', 'path', 'score', 'clock', 'failed_map'])
def test_invalid_evidence_preserves_the_original_exception(fault):
    s, r, m = fixture()
    if fault == 'future': r['residuals'][-1]['available_tick'] = 11
    elif fault == 'bias': r['correction_xy_m'][0] = .2
    elif fault == 'path': s['nominal_path_checks'][1]['segments'][4]['radius_m'] = .4
    elif fault == 'score': s['candidates'][1]['causal_scoring_body_xy_m'] = [1., 0.]
    elif fault == 'clock': m.surface.last_ns -= 100_000_000
    else: m.failed = True
    with pytest.raises(ValueError) as original:
        candidate.reconsider_anchored_continuation(s, r, m, object(), now_ns=NOW)
    with pytest.raises(ValueError) as changed:
        apply(s, r, m)
    assert str(changed.value) == str(original.value)


def test_controller_inherits_observation_mission_and_failure_behavior():
    for name in ('observe', 'advance'):
        assert getattr(SharedSurfaceAnchoredController, name) is getattr(ResidualAnchoredContinuationController, name)
    controller = SharedSurfaceAnchoredController(object(), object(),
        public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
            require_return_after_goal=True), navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    assert controller.selector.residual is controller.residual
    assert controller.memory is controller.mapper.surface
    failed = controller.observe({}, {}, {}, now_ns=1)
    assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert failed['requested_command'] == [0., 0., 0.]
    assert failed[FLAG] is True and failed['residual_anchored_continuation_enabled'] is True
