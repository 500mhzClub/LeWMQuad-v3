import numpy as np
import pytest

from scripts.run_go2_fast_gyro_scan_development_v1 import scan_scenes, sensor_decision
from scripts.fast_gyro_scan_session_development import FastGyroSession, RouteSession
from scripts.run_go2_multijunction_route_development_v1 import PhysicalStop
from lewm.fast_gyro_development import FastGyroBuffer
from lewm.simulated_fast_gyro_development import IdealFastGyro
from lewm.fast_gyro_scan_development import FastGyroScan
from lewm.active_gyro_scan_development import ActiveGyroScan
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.tests.test_fast_gyro_development import buffers
from lewm.tests.test_relative_gyro_turn_development import packet


def test_eight_exact_paired_fixtures_keep_narrow_wall_challenge():
    specs = scan_scenes()
    assert len(specs) == 8 and len({s['scene_id'] for s in specs}) == 8
    for i in range(0, 8, 2):
        a, b = specs[i:i+2]
        assert a['geometry'] == b['geometry'] and a['procedural_seed'] == b['procedural_seed']
        assert a['rate_hz'] == 50 and b['rate_hz'] == 500
        assert a['initial_heading_rad'] == b['initial_heading_rad'] == 0.
        assert a['geometry']['spawn_se2_world'] == [0., 0., 0.]
    assert {(s['motif'], s['width_m']) for s in specs} == {('dead_end', .9), ('dead_end', 1.2), ('cross', .9), ('cross', 1.2)}


def test_native_stop_sample_still_acquires_fast_measurement(monkeypatch):
    session = FastGyroSession.__new__(FastGyroSession)
    session.samples = []
    session.fast_sensor = IdealFastGyro()
    session.fast_buffer = FastGyroBuffer((0, 0, 0))
    session.fast_rows = []
    def stop(self, requested, applied, timestamp):
        self.samples.append({'timestamp_s': timestamp, 'base_pose_world': np.array([0, 0, .3, 0, 0, 0, 1]),
                             'base_twist_world': np.array([0, 0, 0, 0, 0, .2])})
        raise PhysicalStop('DISALLOWED_CONTACT')
    monkeypatch.setattr(RouteSession, '_sample', stop)
    with pytest.raises(PhysicalStop): session._sample([0, 0, 0], [0, 0, 0], .002)
    assert len(session.fast_rows) == 1 and session.fast_rows[0]['measured_ns'] == 2_000_000
    assert np.array_equal(session.fast_rows[0]['values'], [0, 0, .2])


@pytest.mark.parametrize('rate', [50, 500])
def test_initial_decision_and_ground_observer_use_separate_channels(rate):
    slow, fast = buffers()
    p = packet(slow, 80)
    controller = ActiveGyroScan() if rate == 50 else FastGyroScan()
    f = None if rate == 50 else fast.packet(now_ns=1_600_000_000)
    decision, state, observation = sensor_decision(controller, CausalGravityFeedbackGround('transported_feedback'),
                                                 p, tick=0, observation_id='frame-0', fast_packet=f)
    assert decision['requested_command'] == [0, 0, .35]
    assert not state['ground_plane_qualified'] and not observation['metric_clearance_qualified']
    assert set(p) == {'image', 'sensor_state'}
