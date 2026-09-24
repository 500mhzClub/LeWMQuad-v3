"""Restore only the known identity tuple lost by JSON encoding of audited evidence."""
from copy import deepcopy
from lewm.causal_sensor_state import _identity, SensorContractError


def restore_identity(evidence,expected_identity):
    expected=_identity(expected_identity)
    if (not isinstance(evidence,dict) or type(evidence.get('identity')) is not list
            or len(evidence['identity'])!=3 or any(type(v) is not int for v in evidence['identity'])
            or evidence['identity']!=list(expected)):
        raise SensorContractError('exact JSON identity matching current public packet required')
    result=deepcopy(evidence);result['identity']=tuple(result['identity'])
    _identity(result['identity'])
    return result
