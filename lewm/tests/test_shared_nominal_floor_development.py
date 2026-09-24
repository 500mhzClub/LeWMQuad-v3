from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_floor_evidence_development import PairedFloorEvidence
from lewm.correlated_moment_sensitivity_development import RAW_SHAPES
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.shared_nominal_floor_development import SharedNominalFloorEvidence
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth


def test_single_nominal_update_supplies_original_evidence_floor_and_ray_memory(monkeypatch):
    shared = SharedNominalFloorEvidence(['range_scale'], difference_step=.01)
    reference = PairedFloorEvidence(['range_scale'], difference_step=.01)
    original = DepthRelativeState(); rays = FusedRayEvidenceMemory()
    counter = []; observe = shared.observer.models[0].depth.observe
    def counted(*args, **kwargs):
        counter.append(kwargs['now_ns']); return observe(*args, **kwargs)
    monkeypatch.setattr(shared.observer.models[0].depth, 'observe', counted)
    for tick, p, f, _ in stream(4):
        d = room_depth(p); now = d['measured_ns']
        loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
        loading['depth_m'][..., 0] = .002 * d['depth_m']
        result = shared.observe(p, d, f, loading)
        baseline = reference.observe(p, d, f, loading)
        assert result['nominal_depth_state'] == original.observe(p, d, f, now_ns=now)
        rays.observe(p, d, result['nominal_depth_state'], now_ns=now)
        assert rays.fusion == result['nominal_fusion'] == baseline['nominal_fusion']
        shared.retain(str(tick)); reference.retain(str(tick))
        a = shared.query('0', [[1.5, .013, -.27]], np.array([True]), now_ns=now)
        b = reference.query('0', [[1.5, .013, -.27]], np.array([True]), now_ns=now)
        for key, expected in b.items():
            if isinstance(expected, np.ndarray): np.testing.assert_array_equal(a[key], expected)
            else: assert a[key] == expected
        snapshot = shared.nominal_depth_state(now_ns=now)
        snapshot['local_surfaces']['surface_segments'].clear()
        assert shared.nominal_depth_state(now_ns=now)['local_surfaces']['surface_segments']
        assert len(counter) == tick + 1


def test_stale_or_failed_observer_cannot_export_cached_nominal_evidence():
    shared = SharedNominalFloorEvidence(['zero'])
    _, p, f, _ = next(stream(1)); d = room_depth(p); now = d['measured_ns']
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    with pytest.raises(SensorContractError): shared.nominal_depth_state(now_ns=now)
    shared.observe(p, d, f, loading)
    with pytest.raises(SensorContractError): shared.nominal_depth_state(now_ns=now + 1)
    with pytest.raises(SensorContractError): shared.observe(p, d, f, loading)
    with pytest.raises(SensorContractError): shared.nominal_depth_state(now_ns=now)


def test_shared_weak_depth_remains_partial_and_ray_consumer_still_latches_budget_stop():
    shared = SharedNominalFloorEvidence(['zero']); rays = FusedRayEvidenceMemory()
    stopped = None
    for tick, p, f, _ in stream(20):
        d = room_depth(p); now = d['measured_ns']
        if tick >= 2:
            d['valid'][:, :280] = False; d['valid'][:, 360:] = False
            d['depth_m'][~d['valid']] = 0.
        loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
        result = shared.observe(p, d, f, loading)
        if tick >= 2:
            assert result['nominal_depth_state']['motion']['rank'] == 2
            assert result['nominal_depth_state']['motion']['translation_previous_body_m'] is None
            assert result['nominal_depth_state']['position_initial_body_m'] is None
            assert result['nominal_fusion']['kind'] == 'INERTIALLY_PREDICTED_WEAK_COMPONENT'
        try: rays.observe(p, d, result['nominal_depth_state'], now_ns=now)
        except SensorContractError:
            assert not result['nominal_fusion']['usable_under_declared_proxy_budget']
            stopped = tick; break
    assert stopped == 15 and rays.failed
    with pytest.raises(SensorContractError): rays.query([[1, 0, 0]], np.array([False]), now_ns=now)
