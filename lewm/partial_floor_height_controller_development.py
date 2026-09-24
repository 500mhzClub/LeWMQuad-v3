"""Typed height observation in every existing measured-pose consumer."""
from types import FunctionType
from lewm.partial_floor_height_development import partial_height_transport, current_partial_height_pose
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.measured_floor_transport_controller_development import (
    MeasuredFloorTransportMemory, MeasuredFloorTransportMap, MeasuredFloorTransportResidual, MeasuredFloorTransportController)
from lewm.direct_flow_floor_transport_controller_development import DirectFlowFloorTransportController


def bind(original, **dependencies):
    if any(name not in original.__globals__ for name in dependencies):
        raise ValueError('only existing explicit method dependencies may differ')
    namespace = original.__globals__.copy(); namespace.update(dependencies)
    result = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__; result.__annotations__ = original.__annotations__
    result.__qualname__ = original.__qualname__; result.__doc__ = original.__doc__
    return result


class PartialHeightRegistration(MeasuredFloorTransportRegistration):
    observe = bind(MeasuredFloorTransportRegistration.observe, transport_evidence=partial_height_transport)


class PartialHeightMemory(MeasuredFloorTransportMemory):
    observe = bind(MeasuredFloorTransportMemory.observe, current_measured_floor_pose=current_partial_height_pose)


class PartialHeightMap(MeasuredFloorTransportMap):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity); self.surface = PartialHeightMemory(identity=identity)


class PartialHeightResidual(MeasuredFloorTransportResidual):
    observe = bind(MeasuredFloorTransportResidual.observe, current_measured_floor_pose=current_partial_height_pose)


class PartialHeightDirectFlowController(DirectFlowFloorTransportController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registration = PartialHeightRegistration(identity=(0, 0, 0))
        self.mapper = PartialHeightMap(identity=(0, 0, 0)); self.memory = self.mapper.surface
        self.residual = PartialHeightResidual(); self.selector.residual = self.residual

    advance = bind(MeasuredFloorTransportController.advance, current_measured_floor_pose=current_partial_height_pose)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs)|dict(controller='partial_height_direct_flow_controller_v1',
            partial_floor_height_constraint_enabled=True)
