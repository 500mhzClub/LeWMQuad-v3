"""Copy ownership, unchanged calculation code and actual nominal-policy gates."""
import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import FunctionType
import pytest
from lewm import receipt_copied_anchored_selection_development as candidate
from lewm.receipt_copy_development import copy_receipt
from lewm.receipt_copied_anchored_controller_development import (
    ReceiptCopiedAnchoredController, ReceiptCopiedAnchoredSelector)
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)
from lewm.tests.test_residual_anchored_continuation_development import fixture
from lewm.tests.test_residual_hold_feasibility_development import fixture as hold_fixture
from lewm.tests.test_residual_first_interval_feasibility_development import NOW


SOURCE_SHA256 = {
    'lewm/receipt_copy_development.py': '6237bc01257e2eacf3b2ef2effa6c3aa24af6eb902eb5ce6f40a757db232c6ea',
    'lewm/observation_horizon_nominal_constraint_development.py': 'eda24d382bb69d5678961517976a8bd102ce9dc03f77af639e03e5ef090acec1',
    'lewm/eight_step_planning_development.py': '6c0a6062196b603a8f8e3935d3f204f8dd487f4e952375def626a6be8bfcd2cc',
    'lewm/observation_horizon_surface_filter_development.py': '591c2372a2177f6a2051a4ba9284777b510c56b559a0d2156292525687e8a337',
    'lewm/residual_hold_feasibility_development.py': '62dc7768b1b644b0c79c8471a3972543a5c1cf3ac4c90a154943b01b5ee6443c',
    'lewm/residual_anchored_continuation_development.py': '7360e27e6c24444ed3149350c371cf81ce2baeb3005663d7ecb2594cf7bae5b2',
    'lewm/residual_anchored_continuation_controller_development.py': 'f35ff18c78b4db81c0c3c766eed1015823bf972d484600907954941ef7fb946a',
}


def apply(selection, receipt, mapper):
    return candidate.reconsider_with_receipt_copy(selection, receipt, mapper, object(), now_ns=NOW)


def test_bound_predecessor_code_is_unchanged():
    for name, expected in SOURCE_SHA256.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    original = ast.parse(textwrap.dedent(inspect.getsource(ResidualAnchoredContinuationSelector.choose)))
    copied = ast.parse(textwrap.dedent(inspect.getsource(ReceiptCopiedAnchoredSelector.choose)))
    assert ast.dump(original) == ast.dump(copied)


def test_private_namespaces_change_only_declared_copy_and_function_bindings():
    before = {f: f.__globals__.copy() for f in candidate.ORIGINAL_FUNCTIONS}
    clones = candidate.isolated_selection_functions()
    assert len(clones) == 6
    for original, copied in clones.items():
        assert copied.__code__ is original.__code__
        assert copied.__globals__ is not original.__globals__
        assert copied.__defaults__ is original.__defaults__
        assert copied.__kwdefaults__ == original.__kwdefaults__
        assert copied.__closure__ is original.__closure__ is None
        assert copied.__globals__.keys() == original.__globals__.keys()
        for name, value in original.__globals__.items():
            expected = copy_receipt if name == 'deepcopy' else (
                clones[value] if isinstance(value, FunctionType) and value in clones else value)
            assert copied.__globals__[name] is expected
            assert before[original][name] is value
    with pytest.raises(TypeError):
        clones[candidate.plan] = candidate.plan


@pytest.mark.parametrize('make_fixture', [fixture, hold_fixture])
def test_recovery_values_aliasing_and_source_isolation_match(make_fixture):
    selection, receipt, mapper = make_fixture()
    shared = {'points': [1., 2.]}
    cycle = []; cycle.append(cycle)
    selection['copy_witness'] = [shared, shared, cycle]
    expected = candidate.reconsider_anchored_continuation(selection, receipt, mapper, object(), now_ns=NOW)
    result = apply(selection, receipt, mapper)
    assert result is not selection and result['action'] == expected['action'] == 'forward'
    a, b = result.pop('copy_witness'), expected.pop('copy_witness')
    assert result == expected
    for witness in (a, b):
        assert witness[0] is witness[1] and witness[0] is not shared
        assert witness[2] is witness[2][0] and witness[2] is not cycle
        witness[0]['points'][0] = 9.
    result['candidates'][0]['utility_m'] = 123.
    result['surface_checks'][0]['possible_intersection'] = True
    assert shared['points'] == [1., 2.]
    assert selection['candidates'][0]['utility_m'] != 123.
    assert selection['surface_checks'][0]['possible_intersection'] is False


@pytest.mark.parametrize('veto', ['later_segment', 'contact_cost', 'original_surface', 'corrected_surface', 'phase', 'tie'])
def test_original_gate_rejections_and_input_identity_are_retained(veto):
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
def test_invalid_evidence_retains_original_exception(fault):
    s, r, m = fixture()
    if fault == 'future': r['residuals'][-1]['available_tick'] = 11
    elif fault == 'bias': r['correction_xy_m'][0] = .2
    elif fault == 'path': s['nominal_path_checks'][1]['segments'][4]['radius_m'] = .4
    elif fault == 'score': s['candidates'][1]['causal_scoring_body_xy_m'] = [1., 0.]
    elif fault == 'clock': m.surface.last_ns -= 100_000_000
    else: m.failed = True
    with pytest.raises(ValueError) as original:
        candidate.reconsider_anchored_continuation(s, r, m, object(), now_ns=NOW)
    with pytest.raises(ValueError) as copied:
        apply(s, r, m)
    assert str(copied.value) == str(original.value)


def test_controller_inherits_observation_mission_and_failure_behavior():
    for name in ('observe', 'advance'):
        assert getattr(ReceiptCopiedAnchoredController, name) is getattr(ResidualAnchoredContinuationController, name)
    controller = ReceiptCopiedAnchoredController(object(), object(),
        public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
            require_return_after_goal=True), navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    assert controller.selector.residual is controller.residual
    assert controller.memory is controller.mapper.surface
    failed = controller.observe({}, {}, {}, now_ns=1)
    assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert failed['requested_command'] == [0., 0., 0.]
    assert failed['anchored_selection_receipt_copy_enabled'] is True
    assert failed['residual_anchored_continuation_enabled'] is True
