"""Real image/plane fits with explicit acquisition gaps and continuous gyro."""
import cv2
import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.gapped_camera_plane_tracker_development import GappedCameraPlaneTracker
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneChainedPose
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run

cv2.setNumThreads(1)


def ingest(model, item):
    return model.ingest_gyro(item[0], item[2], now_ns=item[3]['now_ns'])


def test_consecutive_frames_retain_complete_original_raw_pose_results():
    old = SampledPlaneChainedPose(); new = GappedCameraPlaneTracker()
    for item in sequence():
        ingest(new, item)
        expected = run(old, item); actual = run(new, item)
        comparable = {k:actual[k] for k in expected}
        comparable['original_temporal_gate_values_unchanged'] = True
        assert comparable == expected
        assert new.last_continuity == old.last_continuity
        assert actual['acquisition_frame'] == actual['frame']


def test_gaps_use_real_timestamps_and_current_auxiliary_image_without_reset():
    model = GappedCameraPlaneTracker(); chosen = (0, 3, 5, 9); accepted = []
    for frame, item in enumerate(sequence(10, blank_primary_at=(3,))):
        ingest(model, item)
        if frame not in chosen:
            continue
        result = run(model, item); accepted.append(result)
        assert result['frame'] == len(accepted)-1 and result['acquisition_frame'] == frame
        assert result['measured_ns'] == item[3]['now_ns']
        assert result['measured_plane_evidence']['measured_ns'] == item[3]['now_ns']
        assert result['continuous_gyro_intervals'] == frame*50
        assert result['global_history_reset'] is False
        np.testing.assert_allclose(result['position_initial_body_m'], 0., atol=1e-9)
        if frame:
            assert result['previous_visual_measured_ns'] == accepted[-2]['measured_ns']
            assert model.last_continuity['previous_measured_ns'] == accepted[-2]['measured_ns']
    assert accepted[1]['camera_selection']['selected_camera'] == 'auxiliary'
    assert accepted[2]['camera_selection']['selected_camera'] == 'primary'
    assert [r['visual_interval_ns'] for r in accepted] == [None, 300_000_000, 200_000_000, 400_000_000]


@pytest.mark.parametrize('fault', ['duplicate', 'overlong_camera_gap', 'missing_gyro', 'delayed_origin'])
def test_invalid_sequence_latches_instead_of_resetting_or_retiming(fault):
    model = GappedCameraPlaneTracker(); items = list(sequence(7))
    ingest(model, items[0])
    if fault == 'delayed_origin':
        ingest(model, items[1])
        with pytest.raises(ValueError): run(model, items[1])
    else:
        run(model, items[0]); reference = model.previous
        if fault == 'duplicate':
            with pytest.raises(ValueError): run(model, items[0])
        elif fault == 'missing_gyro':
            with pytest.raises(ValueError): ingest(model, items[2])
        else:
            for item in items[1:]: ingest(model, item)
            with pytest.raises(ValueError): run(model, items[-1])
        assert model.previous is reference
    assert model.failed
    with pytest.raises(ValueError): ingest(model, items[-1])
    with pytest.raises(ValueError): run(model, items[-1])


def test_gapped_measured_bridges_keep_original_one_second_elapsed_limit(monkeypatch):
    model = GappedCameraPlaneTracker(); items = list(sequence(16))
    ingest(model, items[0]); run(model, items[0]); original_reference = model.references[0]
    def no_anchor(self, current, G):
        self.last_selection = dict(status='NO_QUALIFIED_REFERENCE', selected_reference=None)
        raise SensorContractError('synthetic missing retained anchor; real incremental image fit remains')
    monkeypatch.setattr(MultiReferenceRGBDPose, '_choose', no_anchor)
    for frame, item in enumerate(items[1:], 1):
        ingest(model, item)
        if frame in (5, 10):
            run(model, item)
            assert model.last_continuity['status'] == 'MEASURED_INCREMENT_BRIDGE'
            assert model.bridge_frames == frame//5
        elif frame == 15:
            with pytest.raises(SensorContractError): run(model, item)
    assert model.failed and model.previous.measured_ns == items[10][3]['now_ns']
    assert model.bridge_frames == 2 and model.references == [original_reference]
