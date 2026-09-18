"""Isolate receipt copying in the existing anchored-continuation calculation.

The six functions keep their original code objects. Only private function
namespaces route the existing deepcopy calls through copy_receipt; imported
modules, geometry providers and upstream selection remain untouched.
"""
from copy import deepcopy
from types import FunctionType, MappingProxyType
from lewm.receipt_copy_development import copy_receipt
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.residual_hold_feasibility_development import reconsider_hold_feasibility
from lewm.residual_anchored_continuation_development import (
    _reconsider_anchored_hold, reconsider_anchored_continuation)

ORIGINAL_FUNCTIONS = (constrain, plan, filter_selection,
    reconsider_hold_feasibility, _reconsider_anchored_hold,
    reconsider_anchored_continuation)


def isolated_selection_functions():
    clones = {}
    for original in ORIGINAL_FUNCTIONS:
        if original.__closure__ is not None or original.__globals__.get('deepcopy') is not deepcopy:
            raise ValueError('explicit original module functions and standard copying required')
        namespace = original.__globals__.copy()
        copied = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
        copied.__kwdefaults__ = None if original.__kwdefaults__ is None else original.__kwdefaults__.copy()
        copied.__annotations__ = original.__annotations__.copy()
        copied.__qualname__ = original.__qualname__
        copied.__module__ = original.__module__
        copied.__doc__ = original.__doc__
        clones[original] = copied
    for copied in clones.values():
        namespace = copied.__globals__
        namespace['deepcopy'] = copy_receipt
        for name, target in tuple(namespace.items()):
            if isinstance(target, FunctionType) and target in clones:
                namespace[name] = clones[target]
    return MappingProxyType(clones)


COPIED_FUNCTIONS = isolated_selection_functions()
reconsider_with_receipt_copy = COPIED_FUNCTIONS[reconsider_anchored_continuation]
