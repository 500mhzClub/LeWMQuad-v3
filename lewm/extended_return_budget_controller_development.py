"""Fresh longer-budget chained controller; prospective validation is required.

No existing instance is upgraded. Rendering, packet acquisition, native
recording and evaluator integration are outside this controller composition.
"""
from lewm.measured_plane_chained_single_pass_controller_development import MeasuredPlaneChainedSinglePassController
from lewm.body_projected_tiled_controller_development import BodyProjectedTiledFloorMap
from lewm.measured_floor_transport_controller_development import (
    MeasuredFloorTransportMemory, MeasuredFloorTransportResidual)
from lewm.later_floor_evidence_development import LaterFloorEvidence
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMap
from lewm.tiled_density_floor_registration_development import TiledDensityFloorRegistration
from lewm import receipt_copied_footprint_development as receipts
from lewm.extended_return_budget_mission_development import (
    NAVIGATION_TICKS, ExtendedReturnBudgetMeasuredMission)
from lewm.extended_return_budget_transport_development import (
    current_measured_floor_pose, ExtendedReturnBudgetFloorRegistration)
from lewm.extended_return_budget_memory_development import (
    ExtendedReturnBudgetMemory, ExtendedReturnBudgetResidual,
    ExtendedReturnBudgetLaterFloorEvidence, ExtendedReturnBudgetRecordingFloorGeometry)

CONTROLLER = 'extended_return_budget_chained_single_pass_controller_v1'
fork = receipts.fork


class ExtendedReturnBudgetFloorMap(BodyProjectedTiledFloorMap):
    observe = fork(LaterResolvedFloorMap.observe,
        RecordingFloorGeometry=ExtendedReturnBudgetRecordingFloorGeometry)


footprint_view = fork(receipts.footprint_view, MeasuredFloorTransportMemory=ExtendedReturnBudgetMemory)


class ExtendedReturnBudgetFootprintScope(receipts.ReceiptCopiedFootprintScope):
    __init__ = fork(receipts.ReceiptCopiedFootprintScope.__init__, footprint_view=footprint_view)


class ExtendedReturnBudgetSelector(receipts.ReceiptCopiedFootprintSelector):
    choose = fork(receipts.ReceiptCopiedFootprintSelector.choose,
        MeasuredFloorTransportMemory=ExtendedReturnBudgetMemory,
        FusedScopedFootprintReuse=ExtendedReturnBudgetFootprintScope)


def _copy_fresh_fields(original, cls):
    result = object.__new__(cls)
    result.__dict__ = vars(original).copy()
    return result


def _install_fresh_components(controller):
    mapper, memory = controller.mapper, controller.memory
    if (controller.tick != -1 or controller.terminal is not None
            or type(mapper) is not BodyProjectedTiledFloorMap
            or type(memory) is not MeasuredFloorTransportMemory
            or mapper.surface is not memory or memory.route or memory.last_ns is not None
            or memory.failed or mapper.failed or mapper.frame_geometry is not None
            or memory.frame_geometry is not None or controller.history
            or type(memory.later_floor_evidence) is not LaterFloorEvidence
            or memory.later_floor_evidence.frame != -1
            or memory.later_floor_evidence._records or memory.later_floor_evidence._cell_observations
            or type(controller.registration) is not TiledDensityFloorRegistration
            or controller.registration.frame != -1 or controller.registration.failed
            or type(controller.residual) is not MeasuredFloorTransportResidual
            or controller.residual.frame != -1 or controller.residual.pending is not None
            or controller.residual.history
            or type(controller.selector) is not receipts.ReceiptCopiedFootprintSelector
            or controller.selector.residual is not controller.residual):
        raise ValueError('only untouched original chained single-pass components may be composed')
    revised_memory = _copy_fresh_fields(memory, ExtendedReturnBudgetMemory)
    revised_memory.later_floor_evidence = _copy_fresh_fields(
        memory.later_floor_evidence, ExtendedReturnBudgetLaterFloorEvidence)
    revised_mapper = _copy_fresh_fields(mapper, ExtendedReturnBudgetFloorMap)
    revised_mapper.surface = revised_memory
    revised_residual = _copy_fresh_fields(controller.residual, ExtendedReturnBudgetResidual)
    revised_registration = _copy_fresh_fields(controller.registration, ExtendedReturnBudgetFloorRegistration)
    revised_selector = _copy_fresh_fields(controller.selector, ExtendedReturnBudgetSelector)
    revised_selector.residual = revised_residual
    controller.mapper, controller.memory = revised_mapper, revised_memory
    controller.residual, controller.selector = revised_residual, revised_selector
    controller.registration = revised_registration


class ExtendedReturnBudgetChainedController(MeasuredPlaneChainedSinglePassController):
    def __init__(self, model, geometry, *, public_mission, navigation_ticks, **kwargs):
        if type(navigation_ticks) is not int or not 1 <= navigation_ticks <= NAVIGATION_TICKS:
            raise ValueError('explicit bounded extended return budget required')
        # Build the original empty optimized components before any observation.
        # Their temporary mission is replaced before this constructor returns.
        super().__init__(model, geometry, public_mission=public_mission,
            navigation_ticks=min(navigation_ticks, 4000), **kwargs)
        _install_fresh_components(self)
        self.mission = ExtendedReturnBudgetMeasuredMission(public_mission, navigation_ticks=navigation_ticks)

    advance = fork(MeasuredPlaneChainedSinglePassController.advance,
        current_measured_floor_pose=current_measured_floor_pose)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller=CONTROLLER, extended_return_budget_enabled=True)
