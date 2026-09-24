from copy import deepcopy
from functools import partial
import numpy as np
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.dual_camera_settled_controller_development import DualCameraSettledController
from lewm.single_pass_dual_camera_controller_development import SinglePassDualCameraController
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion
from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
from lewm.tests.test_batched_sample_bounds_development import assert_equal
from lewm.tests.test_frame_floor_cache_development import equal


def indices(memory):
    return [memory.index, memory.auxiliary_index, *[v for p in
        (memory.partition, memory.auxiliary_partition, memory.confirmed_auxiliary_partition)
        for v in (p.floor, p.other)]]


def test_new_controller_owns_empty_indices_and_preserves_dual_camera_and_mission_types():
    old, new = [cls(None, None, **kwargs()) for cls in (DualCameraSettledController, SinglePassDualCameraController)]
    values = indices(new.memory)
    assert len(values) == len({id(v) for v in values}) == 8
    assert all(type(v) is SinglePassMeasuredSampleBoundsIndex and not v.cells for v in values)
    assert new.memory is new.mapper.surface and new.residual is new.selector.residual
    assert type(new.motion) is type(old.motion) is DualCameraVisualMotion
    assert type(new.mission) is type(old.mission)


def test_actual_dual_camera_floor_mapping_and_missing_rgb_stop_remain_exact(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.fast_gyro_development import FastGyroBuffer
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    old, new = [cls(None, None, **kwargs()) for cls in (DualCameraSettledController, SinglePassDualCameraController)]
    p, d, a, _, now = packets()
    image = from_captured_rgb(p['image']['rgb'], a, p, measured_ns=now, available_ns=now, now_ns=now)
    gyro = FastGyroBuffer((0, 0, 0))
    for t in range(now-100_000_000, now+1, 2_000_000):
        gyro.append(np.zeros(3), np.ones(3, bool), measured_ns=t, available_ns=t)
    f = gyro.packet(now_ns=now)
    rows = []
    for controller in (old, new):
        pp, dd, ff, aa, ii = deepcopy((p, d, f, a, image))
        rows.append(controller.observe(pp, dd, ff, auxiliary_rgb=ii, auxiliary_depth=aa, now_ns=now))
    assert rows[0]['terminal'] is None and rows[1]['terminal'] is None
    equal(*rows)
    for aindex, bindex in zip(indices(old.memory), indices(new.memory), strict=True):
        assert_equal(aindex, bindex)
        assert all(v.base is None and v.flags.owndata for v in bindex.bounds.values())
    counts = [c.memory.partition.total_returns for c in (old, new)]
    rows = [c.observe(p, d, f, auxiliary_rgb=None, auxiliary_depth=a, now_ns=now+100_000_000)
        for c in (old, new)]
    equal(*rows)
    assert rows[0]['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and rows[0]['requested_command'] == [0., 0., 0.]
    assert counts == [c.memory.partition.total_returns for c in (old, new)]
