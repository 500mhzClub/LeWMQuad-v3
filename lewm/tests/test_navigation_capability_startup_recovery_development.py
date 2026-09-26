import threading
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from lewm import navigation_capability_startup_recovery_development as startup
from lewm.causal_sensor_state import SensorContractError
from lewm.fresh_obstacle_dispatch_development import CurrentObstacles
from scripts.check_go2_capability_map_domain_contract_development import check


def test_generator_domain_contract():check()


def test_absent_initial_floor_can_recover_but_other_errors_latch():
    mapper=SimpleNamespace(latest=None,floor_height=None,failed=True)
    packet=SimpleNamespace(frame=0)
    with patch.object(startup.process,'_mapper',mapper),patch.object(startup.process,'mapping_update',side_effect=SensorContractError(startup.INITIAL_FLOOR_MISSING)):
        assert startup.mapping_update(packet,{})[0] is None
        assert not mapper.failed
    mapper.failed=True
    with patch.object(startup.process,'_mapper',mapper),patch.object(startup.process,'mapping_update',side_effect=SensorContractError('bad packet binding')):
        with pytest.raises(SensorContractError):startup.mapping_update(packet,{})
        assert mapper.failed


class Base:
    def __init__(self):self.lock=threading.RLock()
    def _command_gate(self,result,now_ns):return result


class Runtime(startup.StartupRecoveryRuntimeMixin,Base):pass


def test_startup_scan_preserves_obstacle_guard_and_has_time_and_gyro_bounds():
    r=Runtime();original=dict(requested_command=[.1,0.,0.],reason='original')
    assert r._command_gate(original,1_500_000_000) is original
    r.startup_missing=True;r.startup_first_ns=1_500_000_000;r.startup_gyro_ns=2_600_000_000
    def obstacle(cells):
        return CurrentObstacles(11,r.startup_gyro_ns,(0.,0.,0.),tuple(map(tuple,np.eye(3))),
            frozenset(cells),(100,100),'current_body_1cm_grid')
    r.latest_obstacles=obstacle([(0,0)])
    assert r._command_gate(original,r.startup_gyro_ns)['requested_command']==[0.,0.,0.]
    r.latest_obstacles=obstacle([(90,90)])
    assert r._command_gate(original,r.startup_gyro_ns)['requested_command']==[0.,0.,.45]
    assert r._command_gate(original,r.startup_gyro_ns+120_000_000)['requested_command']==[0.,0.,0.]
    r.startup_gyro_turn=startup.MAX_GYRO_TURN_RAD
    assert r._command_gate(original,r.startup_gyro_ns)['requested_command']==[0.,0.,0.]
    assert r.mission_terminal=='INITIAL_FLOOR_RECOVERY_BUDGET_EXHAUSTED'
    r.startup_gyro_turn=0.
    assert r._command_gate(original,r.startup_first_ns+startup.MAX_STARTUP_NS)['requested_command']==[0.,0.,0.]
