"""Fresh composition and short real sensor-to-action checks, not native trials."""
from copy import deepcopy

import pytest
import torch

from lewm import extended_return_budget_controller_development as new
from lewm import extended_return_budget_memory_development as memory
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorVisualMotion
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMission
from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from lewm.packed_fused_scoped_controller_development import index_owners
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.measured_plane_full_history_timing_development import observer_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


OPTIONS = dict(public_mission=MISSION, navigation_ticks=8000,
    condition='direct', variant='no_rgb', persistent=True)
TYPE_PAIRS = (
    (new.ExtendedReturnBudgetFloorMap, new.BodyProjectedTiledFloorMap),
    (new.ExtendedReturnBudgetMemory, new.MeasuredFloorTransportMemory),
    (new.ExtendedReturnBudgetLaterFloorEvidence, new.LaterFloorEvidence),
    (new.ExtendedReturnBudgetResidual, new.MeasuredFloorTransportResidual),
    (new.ExtendedReturnBudgetFloorRegistration, new.TiledDensityFloorRegistration),
    (new.ExtendedReturnBudgetSelector, new.receipts.ReceiptCopiedFootprintSelector),
    (new.ExtendedReturnBudgetMeasuredMission, MeasuredFloorTransportMission))
TYPE_NAMES = {a.__module__+'.'+a.__name__:b.__module__+'.'+b.__name__ for a,b in TYPE_PAIRS}


def normalized_state(controller):
    # Compare every retained controller field; model and articulated geometry
    # identities/weights are checked separately by the calling test.
    state = observer_state_tree({k:v for k,v in vars(controller).items() if k not in ('model', 'geometry')})
    def visit(value):
        if isinstance(value, dict):
            result = {k:visit(v) for k,v in value.items()}
            if set(result) == {'type', 'fields'} and result['type'] in TYPE_NAMES:
                result['type'] = TYPE_NAMES[result['type']]
            return result
        if isinstance(value, list): return [visit(v) for v in value]
        return value
    return visit(state)


@pytest.mark.parametrize('budget', [-1, 0, 8001, True, 1.5, None])
def test_constructor_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match='bounded extended'):
        new.ExtendedReturnBudgetChainedController(None, None, **(OPTIONS | {'navigation_ticks':budget}))


def test_fresh_composition_preserves_empty_indices_aliases_and_read_only_view():
    controller = new.ExtendedReturnBudgetChainedController(None, None, **OPTIONS)
    assert controller.mission.navigation_ticks == 8000 and controller.tick == -1
    assert controller.memory is controller.mapper.surface
    assert controller.residual is controller.selector.residual
    assert type(controller.motion) is MeasuredPlaneChainedAnchorVisualMotion
    assert type(controller.registration) is new.ExtendedReturnBudgetFloorRegistration
    assert type(controller.memory.later_floor_evidence) is new.ExtendedReturnBudgetLaterFloorEvidence
    indices = [getattr(owner, name) for owner,name in index_owners(controller.memory)]
    assert len(indices) == len({id(i) for i in indices}) == 8
    assert all(type(i) is SinglePassMeasuredSampleBoundsIndex and not any(vars(i).values()) for i in indices)
    view = new.footprint_view(controller.memory)
    assert vars(view) is vars(controller.memory)
    with pytest.raises(TypeError): view.failed = True
    with pytest.raises(ValueError): new._install_fresh_components(controller)


@pytest.mark.parametrize('condition,variant', [('direct', 'no_rgb'), ('jepa', 'full')])
def test_complete_short_sensor_action_state_equivalence_and_optimized_dispatch(monkeypatch, condition, variant):
    models = [FixedHeadModel(condition), FixedHeadModel(condition)]
    weights = [{k:v.clone() for k,v in model.state_dict().items()} for model in models]
    options = OPTIONS | dict(navigation_ticks=40, condition=condition, variant=variant)
    controllers = [cls(model, ArticulatedCollisionGeometry(URDF), **options)
        for cls,model in zip((new.MeasuredPlaneChainedSinglePassController,
            new.ExtendedReturnBudgetChainedController), models, strict=True)]
    calls = []
    original_init = new.ExtendedReturnBudgetFootprintScope.__init__
    def record_scope(self, supplied_memory, geometry):
        calls.append(type(supplied_memory))
        original_init(self, supplied_memory, geometry)
    monkeypatch.setattr(new.ExtendedReturnBudgetFootprintScope, '__init__', record_scope)
    assert fingerprint(normalized_state(controllers[0])) == fingerprint(normalized_state(controllers[1]))
    items = list(sequence())
    for frame, source in enumerate(items+[items[-1]]):
        decisions = []
        for controller in controllers:
            p,d,f,kwargs = [move_test_origin(deepcopy(value)) for value in source]
            decisions.append(controller.observe(p,d,f,**kwargs))
        old, actual = decisions
        assert actual['controller'] == new.CONTROLLER and actual['extended_return_budget_enabled'] is True
        normalized = deepcopy(actual); normalized.pop('extended_return_budget_enabled')
        normalized['controller'] = old['controller']
        assert normalized == old
        assert fingerprint(normalized_state(controllers[0])) == fingerprint(normalized_state(controllers[1]))
        assert fingerprint(observer_state_tree(controllers[0].geometry)) == fingerprint(observer_state_tree(controllers[1].geometry))
        if frame < len(items): assert actual['terminal'] is None, actual.get('failure')
        else: assert actual['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and actual['requested_command'] == [0.,0.,0.]
    assert calls and all(kind is new.ExtendedReturnBudgetMemory for kind in calls)
    assert [len(model.calls) for model in models] == [1, 1]
    for model, before in zip(models, weights, strict=True):
        assert all(torch.equal(before[k], value) for k,value in model.state_dict().items())
        assert all(parameter.grad is None for parameter in model.parameters())
    assert controllers[1].mapper.frame_geometry is None and controllers[1].memory.frame_geometry is None


def test_controller_map_uses_larger_recording_context():
    function = new.ExtendedReturnBudgetFloorMap.observe
    assert function.__globals__['RecordingFloorGeometry'] is memory.ExtendedReturnBudgetRecordingFloorGeometry
    assert new.ExtendedReturnBudgetChainedController.advance.__globals__['current_measured_floor_pose'] is new.current_measured_floor_pose
