"""Private function bindings change only the floor-index implementation."""
from types import FunctionType

from lewm import floor_pose_registration_development as plane
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.eligible_floor_cell_index_development import observed_floor_cell_index


def bind(function, **replacements):
    if function.__closure__ is not None:
        raise ValueError('closure-free original floor function required')
    copied = FunctionType(function.__code__, function.__globals__ | replacements,
        function.__name__, function.__defaults__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


measured_candidates = bind(plane.measured_candidates, observed_floor_cell_index=observed_floor_cell_index)


class EligibleFloorRegistration(MeasuredFloorTransportRegistration):
    observe = bind(MeasuredFloorTransportRegistration.observe, measured_candidates=measured_candidates)
