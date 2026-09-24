from lewm import batched_patch_agreement_development as batch
from lewm.batched_patch_tracker_development import BatchedPatchPose
from lewm.full_consensus_tracker_development import _Dual
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run


def test_pose_fallback_uses_batched_photometry_on_actual_image_pairs(monkeypatch):
    model = BatchedPatchPose(); items = list(sequence(2)); run(model, items[0])
    original = batch.patches_agree; calls = []
    def observed(*args): calls.append(len(args[2])); return original(*args)
    monkeypatch.setattr(batch, 'patches_agree', observed)
    def missing(*args): raise SensorContractError('synthetic missing descriptor support; actual images retained')
    monkeypatch.setattr(_Dual, '_candidate', missing)
    result = run(model, items[1])
    assert calls and max(calls) >= 12
    assert model.last_direct_flow_fallback['accepted']
    assert result['frame'] == 1 and result['global_history_reset'] is False
