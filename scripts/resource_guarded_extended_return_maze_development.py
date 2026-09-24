"""Prospective lifecycle guards around the tested longer-budget pipeline."""
from scripts import extended_return_budget_maze_pipeline_development as pipeline
from scripts import extended_return_budget_resource_guard_development as resources

artifacts = pipeline.artifacts


def resource_artifacts(episode):
    return [name for phase in ('collection','audit') for name in resources.names(episode,phase)]


class CheckedController:
    """The original controller receives all inputs and produces every decision."""
    def __init__(self,controller,guard):
        self.controller=controller;self.guard=guard;self.frame=0

    def observe(self,*args,**kwargs):
        self.guard.check('before_controller',self.frame)
        result=self.controller.observe(*args,**kwargs)
        self.guard.check('after_controller',self.frame)
        self.frame+=1
        return result


def controller_factory(guard):
    def create(*args,**kwargs):
        return CheckedController(pipeline.ExtendedReturnBudgetChainedController(*args,**kwargs),guard)
    return create


def session_type(guard):
    class CheckedSession(pipeline.ExtendedReturnBudgetRendererSession):
        def sensor_packets(self):
            frame=(len(self.samples)-750)//50
            guard.check('before_packet',frame)
            packet=super().sensor_packets()
            guard.check('after_packet',frame)
            return packet
    return CheckedSession


def _execute(function,args,kwargs,*,root,episode,phase):
    guard=resources.ResourceGuard(root,episode,phase);error=None
    try:
        first=guard.check('begin')
        if phase=='collection':resources.admission(first)
        bindings=dict(ResidualAnchoredContinuationController=controller_factory(guard))
        if phase=='collection':
            bindings['RendererWitnessDualCameraMazeSession']=session_type(guard)
        else:
            def audit_sensors(*a,**k):
                guard.check('before_sensor_audit')
                result=pipeline.audit_sensors(*a,**k)
                guard.check('after_sensor_audit')
                return result
            bindings['audit_sensors']=audit_sensors
        bound=pipeline.bind(function,**bindings)
        result=bound(*args,**kwargs)
        guard.check('completed')
        return result
    except BaseException as failure:
        error=failure
        raise
    finally:
        guard.finish(error)


def collect(*args,**kwargs):
    return _execute(pipeline.collect,args,kwargs,root=kwargs['output'],episode=kwargs['episode_name'],phase='collection')


def audit(*args,**kwargs):
    return _execute(pipeline.audit,args,kwargs,root=kwargs['input_root'],episode=kwargs['episode_name'],phase='audit')


def definition():
    return pipeline.definition()|dict(sampled_resource_guards_enabled=True,
        collection_persistence_and_audit_resource_receipts_required=True,
        minimum_initial_available_ram_bytes=resources.INITIAL_RAM_BYTES,
        minimum_initial_artifact_free_bytes=resources.INITIAL_DISK_BYTES,
        original_controller_decision_fields_unchanged=True,
        operating_system_memory_limit_enforced=False,between_sample_peak_bounded=False)
