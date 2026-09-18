"""Restoration and actual synthetic image-to-action instrumentation equivalence."""
from copy import deepcopy
from contextlib import nullcontext
from types import MethodType

import pytest
import torch

from lewm.controller_phase_timing_development import PhaseTiming
from lewm.measured_plane_phase_timing_development import controller_bindings, time_methods


@pytest.mark.parametrize('fails', [False, True])
def test_original_return_exception_side_effects_and_instance_override_restored(fails):
    class Target:
        def outer(self, value):
            return self.inner(value)

        def inner(self, value):
            raise AssertionError('instance override must be retained')

    target = Target()
    result = []
    def override(self, value):
        result.append(value)
        if fails:
            raise RuntimeError('original failure')
        return result
    target.inner = MethodType(override, target)
    before = vars(target).copy()
    t = PhaseTiming()
    with pytest.raises(RuntimeError, match='original failure') if fails else nullcontext():
        with time_methods(((target, 'outer', 'outer'), (target, 'inner', 'inner')), t):
            assert target.outer(7) is result
    assert vars(target) == before and target.inner is before['inner']
    assert result == [7]
    phases = t.snapshot()
    assert phases['inner']['calls'] == phases['outer']['calls'] == 1
    assert sum(p['exclusive_ns'] for p in phases.values()) == phases['outer']['inclusive_ns']


def test_invalid_bindings_are_rejected_before_any_installation():
    class Target:
        def call(self):
            return 1
    target = Target()
    t = PhaseTiming()
    for bindings in (((target, 'call', 'one'), (target, 'missing', 'two')),
                     ((target, 'call', 'one'), (target, 'call', 'two'))):
        with pytest.raises((AttributeError, ValueError)):
            with time_methods(bindings, t):
                pytest.fail('invalid bindings accepted')
        assert vars(target) == {}


@pytest.mark.parametrize('condition,variant', [('jepa', 'full'), ('direct', 'no_rgb')])
def test_actual_current_controller_decisions_states_model_and_objects_unchanged(condition, variant):
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.measured_plane_single_pass_controller_development import MeasuredPlaneSinglePassController
    from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
    from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
    from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
    from scripts.measured_plane_full_history_timing_development import observed_state

    models = [FixedHeadModel(condition), FixedHeadModel(condition)]
    initial = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    arms = [MeasuredPlaneSinglePassController(model, ArticulatedCollisionGeometry(URDF),
        public_mission=deepcopy(MISSION), navigation_ticks=40, condition=condition,
        variant=variant, persistent=True) for model in models]
    t = PhaseTiming()
    bindings = controller_bindings(arms[1])
    identities = tuple((id(target), type(target)) for target, _, _ in bindings)
    original_methods = [(target, name, name in vars(target), vars(target).get(name))
                        for target, name, _ in bindings]
    forward_calls = 0
    packets = list(sequence())
    # A duplicate final packet exercises the actual sensor-failure latch too.
    for original in packets + [packets[-1]]:
        p, d, f, options = [move_test_origin(deepcopy(v)) for v in original]
        expected = arms[0].observe(p, d, f, **options)
        p, d, f, options = [move_test_origin(deepcopy(v)) for v in original]
        t.reset()
        with time_methods(bindings, t):
            actual = arms[1].observe(p, d, f, **options)
        assert actual == expected
        assert fingerprint(observed_state(arms[0])) == fingerprint(observed_state(arms[1]))
        phases = t.snapshot()
        assert sum(row['exclusive_ns'] for row in phases.values()) == phases['controller.observe']['inclusive_ns']
        forward_calls += phases.get('model.forward', {}).get('calls', 0)
        assert tuple((id(target), type(target)) for target, _, _ in controller_bindings(arms[1])) == identities
        for target, name, existed, value in original_methods:
            assert (name in vars(target)) is existed
            if existed:
                assert vars(target)[name] is value
    assert actual['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert forward_calls == 1 and [len(m.calls) for m in models] == [1, 1]
    for model, before in zip(models, initial, strict=True):
        assert all(torch.equal(before[k], value) for k, value in model.state_dict().items())
