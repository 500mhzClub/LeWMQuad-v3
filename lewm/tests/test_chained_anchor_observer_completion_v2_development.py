"""Only validated serialized identity is adapted; original pose validation remains."""
from copy import deepcopy

import pytest

from lewm.causal_sensor_state import SensorContractError
from scripts import verify_go2_chained_anchor_observer_completion_v2 as check


def test_json_identity_adapter_preserves_payload_and_original_validator(monkeypatch):
    evidence = dict(identity=[0, 0, 0], current_pose={'position': [1., 2., 3.]})
    original = deepcopy(evidence)
    called = []
    def validate(value, *args, **kwargs):
        called.append((value, args, kwargs))
        assert value['identity'] == (0, 0, 0)
        assert value['current_pose'] is evidence['current_pose']
        return 'validated'
    monkeypatch.setattr(check.run, 'current_dual_camera_pose', validate)
    assert check.check_serialized_pose(evidence, 'policy', now_ns=1) == 'validated'
    assert len(called) == 1 and evidence == original


@pytest.mark.parametrize('identity', [(0, 0, 0), '000', None, [], [0, 0], [0, 0, 0, 0], [True, 0, 0], [-1, 0, 0]])
def test_invalid_serialized_identity_is_not_repaired(identity):
    with pytest.raises(SensorContractError): check.check_serialized_pose({'identity': identity})


def test_runtime_pose_failure_is_propagated(monkeypatch):
    def reject(*args, **kwargs): raise SensorContractError('bad measured pose')
    monkeypatch.setattr(check.run, 'current_dual_camera_pose', reject)
    with pytest.raises(SensorContractError, match='bad measured pose'):
        check.check_serialized_pose({'identity': [0, 0, 0]})
