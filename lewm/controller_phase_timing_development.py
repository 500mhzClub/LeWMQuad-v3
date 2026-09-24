"""Explicit replay-only phase timers; no global tracing or scientific changes."""
from contextlib import contextmanager
import time
from lewm.later_floor_resolution_controller_development import (
    LaterFloorResolutionRoundTripController, LaterResolvedFloorMap, LaterResolvedFloorMemory)


class PhaseTiming:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock=clock; self.stack=[]; self.values={}

    def reset(self):
        if self.stack: raise ValueError('cannot reset active timing scopes')
        self.values={}

    def begin(self, label):
        self.stack.append(dict(label=label, start=self.clock(), child_ns=0))

    def end(self, label):
        if not self.stack or self.stack[-1]['label'] != label:
            raise ValueError('properly nested explicit timing scopes required')
        end=self.clock(); frame=self.stack.pop(); elapsed=end-frame['start']
        if elapsed < frame['child_ns']: raise ValueError('monotone nested timing clock required')
        row=self.values.setdefault(label,dict(calls=0,inclusive_ns=0,exclusive_ns=0))
        row['calls']+=1; row['inclusive_ns']+=elapsed; row['exclusive_ns']+=elapsed-frame['child_ns']
        if self.stack:self.stack[-1]['child_ns']+=elapsed

    @contextmanager
    def scope(self,label):
        self.begin(label)
        try:yield
        finally:self.end(label)

    def snapshot(self):
        if self.stack:raise ValueError('completed timing scopes required')
        return {k:dict(v) for k,v in self.values.items()}


class PhaseProxy:
    """Delegate original objects and returns, timing only listed bound calls."""
    def __init__(self,target,timing,methods):
        object.__setattr__(self,'_timing_target',target)
        object.__setattr__(self,'_timing_sink',timing)
        object.__setattr__(self,'_timing_methods',dict(methods))

    def __getattr__(self,name):
        target=object.__getattribute__(self,'_timing_target'); value=getattr(target,name)
        methods=object.__getattribute__(self,'_timing_methods')
        if name not in methods:return value
        timing=object.__getattribute__(self,'_timing_sink')
        def measured(*args,**kwargs):
            with timing.scope(methods[name]):return value(*args,**kwargs)
        return measured

    def __setattr__(self,name,value):
        setattr(object.__getattribute__(self,'_timing_target'),name,value)


@contextmanager
def model_forward_timing(model,timing):
    def before(module,args):timing.begin('model.forward')
    def after(module,args,output):timing.end('model.forward')
    pre=model.register_forward_pre_hook(before)
    try:
        post=model.register_forward_hook(after,always_call=True)
        try:yield
        finally:post.remove()
    finally:pre.remove()


class PhaseTimedFloorMemory(LaterResolvedFloorMemory):
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


class PhaseTimedFloorMap(LaterResolvedFloorMap):
    def __init__(self,*,identity,timing):
        super().__init__(identity=identity);self.timing=timing
        self.surface=PhaseTimedFloorMemory(identity=identity,timing=timing)

    def observe(self,*args,**kwargs):
        with self.timing.scope('map.observe'):return super().observe(*args,**kwargs)

    def _observe_primary(self,*args,**kwargs):
        with self.timing.scope('map.primary_coverage'):return super()._observe_primary(*args,**kwargs)

    def waypoint(self,*args,**kwargs):
        with self.timing.scope('map.waypoint'):return super().waypoint(*args,**kwargs)


class PhaseTimedLaterFloorController(LaterFloorResolutionRoundTripController):
    def __init__(self,*args,timing,**kwargs):
        super().__init__(*args,**kwargs);self.timing=timing
        self.mapper=PhaseTimedFloorMap(identity=(0,0,0),timing=timing);self.memory=self.mapper.surface
        self.motion=PhaseProxy(self.motion,timing,{'observe':'motion.observe'})
        self.registration=PhaseProxy(self.registration,timing,{'observe':'floor_registration.observe'})
        self.selector=PhaseProxy(self.selector,timing,{'choose':'selector.choose'})

    def observe(self,*args,**kwargs):
        with self.timing.scope('controller.observe'):return super().observe(*args,**kwargs)

    def advance(self,*args,**kwargs):
        with self.timing.scope('controller.advance'):return super().advance(*args,**kwargs)

    def _result(self,*args,**kwargs):
        with self.timing.scope('controller.result'):return super()._result(*args,**kwargs)
