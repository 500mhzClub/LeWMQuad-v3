"""Temporary, same-object phase instrumentation for a future owned replay.

This helper does not execute a replay or attach to a running controller. Use it
only around synchronous calls on a controller owned by the caller, and compare
decisions and retained state after the context has restored every method.
Durations include instrumentation overhead and are not real-time qualification.
"""
from contextlib import contextmanager
from functools import wraps
from inspect import ismethod


@contextmanager
def time_methods(bindings, timing):
    """Time bound instance methods without replacing objects or class methods."""
    timing.snapshot()  # Reject installation inside an unfinished timing scope.
    prepared = []
    seen = set()
    for target, name, label in bindings:
        key = (id(target), name)
        original = getattr(target, name)
        if (key in seen or not ismethod(original) or original.__self__ is not target
                or not isinstance(label, str) or not label
                or getattr(original, '_temporary_phase_wrapper', False)):
            raise ValueError('distinct original bound methods and named phases required')
        seen.add(key)
        namespace = vars(target)
        prepared.append((namespace, name, name in namespace, namespace.get(name), original, label))

    def measured(original, label):
        @wraps(original)
        def call(*args, **kwargs):
            with timing.scope(label):
                return original(*args, **kwargs)
        call._temporary_phase_wrapper = True
        return call

    installed = []
    try:
        for namespace, name, existed, value, original, label in prepared:
            namespace[name] = measured(original, label)
            installed.append((namespace, name, existed, value))
        yield timing
    finally:
        for namespace, name, existed, value in reversed(installed):
            if existed:
                namespace[name] = value
            else:
                del namespace[name]


def controller_bindings(controller):
    """Explicit outer phases; nested exclusive times avoid double counting."""
    if controller.memory is not controller.mapper.surface:
        raise ValueError('original shared map/memory identity required')
    groups = (
        (controller, (('observe', 'controller.observe'), ('advance', 'controller.advance'),
                      ('_result', 'controller.result'))),
        (controller.motion, (('observe', 'motion.observe'),)),
        (controller.registration, (('observe', 'floor_registration.observe'),)),
        (controller.selector, (('choose', 'selector.choose'),)),
        (controller.model, (('forward', 'model.forward'),)),
        (controller.mapper, (('observe', 'map.observe'), ('_observe_primary', 'map.primary_coverage'),
                            ('waypoint', 'map.waypoint'))),
        (controller.memory, (('observe', 'memory.primary_insert'),
                             ('classify_current', 'memory.primary_classification'),
                             ('_observe_auxiliary_original', 'memory.original_auxiliary'),
                             ('observe_auxiliary', 'memory.auxiliary_confirmation'),
                             ('footprint', 'memory.contact_query'))),
    )
    return tuple((target, name, label) for target, methods in groups for name, label in methods)
