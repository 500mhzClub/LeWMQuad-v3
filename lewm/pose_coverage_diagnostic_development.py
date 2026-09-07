"""Keep diagnostic coverage unavailable without changing an accepted pose.

This is not a navigation fallback. A coverage contract failure never grants
clearance and never projects, resets, normalizes or relaxes the supplied pose.
"""
import numpy as np

from lewm.causal_sensor_state import SensorContractError


def query_coverage(surface, geometry, joints, state, errors):
    if state is None:
        return dict(status='POSE_UNAVAILABLE', floor_coverage=None, reason=None)
    if surface is None or surface.status!='BOUNDED_MEASURED_SURFACE_AVAILABLE':
        return dict(status='SURFACE_UNAVAILABLE', floor_coverage=None,
            reason=None if surface is None else surface.status)
    rotation=np.asarray(state['rotation_initial_body_from_current_body'])
    defect=dict(orthogonality_max_abs=float(np.max(abs(rotation.T@rotation-np.eye(3)))),
                determinant_abs_error=abs(float(np.linalg.det(rotation))-1.))
    try:
        result=surface.query(geometry,joints,rotation_observation_from_body=rotation,
            translation_observation_from_body=state['position_initial_body_m'],point_error_by_shape=errors)
    except SensorContractError as error:
        return dict(status='COVERAGE_CONTRACT_REJECTED',floor_coverage=None,reason=str(error),**defect)
    return dict(status='CONDITIONAL_ZERO_ADDITIONAL_ERROR_COVERAGE',floor_coverage=result['floor_coverage'],
                reason=None,**defect)
