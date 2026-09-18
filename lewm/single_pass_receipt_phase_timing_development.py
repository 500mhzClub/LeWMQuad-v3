"""Replay-only timers around the verified combined current controller."""
from lewm.controller_phase_timing_development import PhaseTiming, PhaseProxy, model_forward_timing
from lewm.single_pass_receipt_copied_controller_development import (
    SinglePassMeasuredFloorMemory, SinglePassMeasuredFloorMap, SinglePassReceiptCopiedController)


class PhaseTimedSinglePassMemory(SinglePassMeasuredFloorMemory):
    def __init__(self,*,identity,timing):
        super().__init__(identity=identity);self.timing=timing

    def observe(self,*args,**kwargs):
        with self.timing.scope('memory.primary_insert'):return super().observe(*args,**kwargs)

    def classify_current(self,*args,**kwargs):
        with self.timing.scope('memory.primary_classification'):return super().classify_current(*args,**kwargs)

    def _observe_auxiliary_original(self,*args,**kwargs):
        with self.timing.scope('memory.original_auxiliary'):return super()._observe_auxiliary_original(*args,**kwargs)

    def observe_auxiliary(self,*args,**kwargs):
        with self.timing.scope('memory.auxiliary_confirmation'):return super().observe_auxiliary(*args,**kwargs)

    def footprint(self,*args,**kwargs):
        with self.timing.scope('memory.contact_query'):return super().footprint(*args,**kwargs)


class PhaseTimedSinglePassMap(SinglePassMeasuredFloorMap):
    def __init__(self,*,identity,timing):
        super().__init__(identity=identity);self.timing=timing
        self.surface=PhaseTimedSinglePassMemory(identity=identity,timing=timing)

    def observe(self,*args,**kwargs):
        with self.timing.scope('map.observe'):return super().observe(*args,**kwargs)

    def _observe_primary(self,*args,**kwargs):
        with self.timing.scope('map.primary_coverage'):return super()._observe_primary(*args,**kwargs)

    def waypoint(self,*args,**kwargs):
        with self.timing.scope('map.waypoint'):return super().waypoint(*args,**kwargs)


class PhaseTimedSinglePassReceiptController(SinglePassReceiptCopiedController):
    def __init__(self,*args,timing,**kwargs):
        super().__init__(*args,**kwargs);self.timing=timing
        self.mapper=PhaseTimedSinglePassMap(identity=(0,0,0),timing=timing);self.memory=self.mapper.surface
        self.motion=PhaseProxy(self.motion,timing,{'observe':'motion.observe'})
        self.registration=PhaseProxy(self.registration,timing,{'observe':'floor_registration.observe'})
        self.selector=PhaseProxy(self.selector,timing,{'choose':'selector.choose'})

    def observe(self,*args,**kwargs):
        with self.timing.scope('controller.observe'):return super().observe(*args,**kwargs)

    def advance(self,*args,**kwargs):
        with self.timing.scope('controller.advance'):return super().advance(*args,**kwargs)

    def _result(self,*args,**kwargs):
        with self.timing.scope('controller.result'):return super()._result(*args,**kwargs)
