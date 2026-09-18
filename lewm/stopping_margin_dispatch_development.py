"""Development stopping allowance for translating commands, retaining the disk veto."""
import numpy as np
from lewm.fresh_obstacle_dispatch_development import dispatch_request as original,NOMINAL_RADIUS_M
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.eligible_floor_registration_development import bind
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.process_registered_round_trip_development import ProcessRegisteredRoundTripRuntime

STOPPING_ALLOWANCE_S=.5


def dispatch_request(plan,current,*,now_ns):
    result=original(plan,current,now_ns=now_ns)
    if result['reason']!='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED' or not any(plan.command[:2]):
        return result
    age=(now_ns-current.measured_ns)/1e9
    horizon=(plan.expires_ns-now_ns)/1e9+age+STOPPING_ALLOWANCE_S
    p=np.asarray(current.position_map);R=np.asarray(current.rotation_map_from_body)
    endpoint=p+R@np.array([plan.command[0]*horizon,plan.command[1]*horizon,0.])
    check=nominal_connector(p[:2],endpoint[:2],sorted(current.occupied),radius_m=NOMINAL_RADIUS_M)
    result=result|dict(stopping_margin_connector=check,stopping_allowance_s=STOPPING_ALLOWANCE_S,
        observation_age_allowance_s=age,stopping_distance_bound_calibrated=False)
    if not check['nominal_disk_connector_clear']:
        result=result|dict(requested_command=[0.,0.,0.],reason='CURRENT_STOPPING_MARGIN_VETO')
    return result


class _StoppingDispatch(PacedMultirateController):
    request=bind(PacedMultirateController.request,dispatch_request=dispatch_request)


class StoppingMarginRoundTripRuntime(ProcessRegisteredRoundTripRuntime,_StoppingDispatch):
    pass
