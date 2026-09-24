"""Use primitive-leaf freezing only; retain the original cached-receipt clone."""
from lewm.atomic_leaf_fused_footprint_development import freeze_ordinary_footprint
from lewm.fused_scoped_footprint_development import FusedScopedFootprintReuse
from lewm.receipt_copied_footprint_development import (
    ReceiptCopiedFootprintScope, ReceiptCopiedFootprintSelector, fork)
from lewm.tiled_density_progressive_floor_controller_development import (
    TiledDensityProgressiveFloorController, CONTROLLER as BASELINE)

CONTROLLER = 'atomic_leaf_freeze_tiled_controller_v1'
FLAG = 'atomic_leaf_footprint_freeze_enabled'


class AtomicLeafFreezeFootprintScope(ReceiptCopiedFootprintScope):
    footprint = fork(FusedScopedFootprintReuse.footprint,
        freeze_ordinary_footprint=freeze_ordinary_footprint)


class AtomicLeafFreezeFootprintSelector(ReceiptCopiedFootprintSelector):
    choose = fork(ReceiptCopiedFootprintSelector.choose,
        FusedScopedFootprintReuse=AtomicLeafFreezeFootprintScope)


class AtomicLeafFreezeTiledController(TiledDensityProgressiveFloorController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.selector
        if (type(old) is not ReceiptCopiedFootprintSelector or old.residual is not self.residual
                or self.memory is not self.mapper.surface or self.memory.route or self.last_ns is not None):
            raise ValueError('fresh original tiled controller and selector aliases required')
        selector = object.__new__(AtomicLeafFreezeFootprintSelector)
        selector.__dict__ = vars(old).copy()
        self.selector = selector

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller':CONTROLLER, FLAG:True}


def normalize_to_tiled(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit atomic-leaf-freeze controller identity required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = BASELINE
    return result
