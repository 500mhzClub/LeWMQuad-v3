"""Test a 250-ms age bound with the existing age-charged stopping connector.

100-ms camera cadence plus roughly 114-ms measured delivery exceeds the
predecessor's 200-ms bound between updates. This is an exposed-development
timing experiment, not a calibrated latency or stopping-distance guarantee.
"""
from types import SimpleNamespace

from lewm import fresh_obstacle_dispatch_development as fresh
from lewm import fine_obstacle_round_trip_development as fine
from lewm import stopping_margin_dispatch_development as stopping
from lewm import auxiliary_only_turn_recovery_development as auxiliary
from lewm.eligible_floor_registration_development import bind
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.short_pulse_navigation_runtime_development import PulsePredictiveRuntime

MAX_OBSERVATION_AGE_NS = 250_000_000

_fresh = bind(fresh.dispatch_request,
    MAX_OBSERVATION_AGE_NS=MAX_OBSERVATION_AGE_NS, nominal_connector=fine.connector)
_fine = bind(fine.original, _original=_fresh)
_stopping = bind(stopping.dispatch_request, original=_fine, nominal_connector=fine.connector)
_auxiliary = bind(auxiliary.dispatch_request,
    fine=SimpleNamespace(dispatch_request=_stopping))


def dispatch_request(plan, current, *, now_ns):
    return _auxiliary(plan, current, now_ns=now_ns) | dict(
        observation_age_limit_ns=MAX_OBSERVATION_AGE_NS,
        observation_age_charged_to_translation_stopping_connector=True,
        sensor_latency_bound_calibrated=False)


class PipelineAgeDispatch(auxiliary.AuxiliaryTurnDispatch):
    request = bind(PacedMultirateController.request, dispatch_request=dispatch_request)


class PipelineAgeRuntime(PulsePredictiveRuntime, PipelineAgeDispatch):
    pass
