"""Synthetic session composition with real public array/packet validation."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

from lewm.tests.test_causal_auxiliary_rgb_observation_development import inputs
from lewm.auxiliary_downward45_depth_geometry_development import CALIBRATION_ID
from scripts import single_read_auxiliary_maze_session_development as new


def same(a, b):
    assert type(a) is type(b)
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True): same(x, y)
    else:
        assert a == b


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    policy, auxiliary, image, now, native, rgb = inputs()
    index = (now-1_500_000_000)//100_000_000
    assert index == 1
    primary_depth = dict(untouched_primary_fixture=True)
    fast = dict(untouched_fast_fixture=True)
    row = dict(frame=index, measured_ns=now, physical_sample_index=749+50*index,
        calibration_id=CALIBRATION_ID,
        native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        rgb_sha256=image['rgb_sha256'], world_from_optical='evaluator-only fixture')
    def persist():
        np.savez_compressed(tmp_path/f'auxiliary_depth_{index:04d}.npz',
            native_optical_depth_m=native, depth_m=auxiliary['depth_m'], valid=auxiliary['valid'],
            diagnostic_segmentation=np.array([{'privileged': True}], dtype=object))
        Image.fromarray(rgb).save(tmp_path/f'auxiliary_rgb_{index:04d}.png')
    persist()
    calls = []
    def primary(session):
        calls.append('primary')
        return policy, primary_depth, fast, now
    def capture(session, directory, frame):
        assert directory == tmp_path and frame == index
        calls.append('capture')
        return deepcopy(row)
    monkeypatch.setattr(new.NovelMazeBaseSession, 'sensor_packets', primary)
    monkeypatch.setattr(new, 'capture', capture)
    monkeypatch.setitem(new.extended.ExtendedBudgetPairedSession.sensor_packets.__globals__,
        'capture', capture)
    def session(cls):
        s = object.__new__(cls)
        s.output = tmp_path; s.samples = range(750+50*index)
        s.model_manifest = [{} for _ in range(index+1)]
        s.auxiliary_audit = [{} for _ in range(index)]
        return s
    return NS(policy=policy, auxiliary=auxiliary, image=image, now=now, native=native,
        rgb=rgb, index=index, row=row, persist=persist, calls=calls, session=session,
        primary_depth=primary_depth, fast=fast, directory=tmp_path)


def test_same_complete_packets_with_one_archive_read_and_no_diagnostic_access(fixture, monkeypatch):
    f = fixture; reads = []; original_load = np.load
    def load(path, *args, **kwargs):
        reads.append(Path(path).name)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', load)
    before = deepcopy(f.policy)
    baseline = f.session(new.extended.ExtendedBudgetDualSession)
    candidate = f.session(new.SingleReadAuxiliaryDualSession)
    expected = baseline.sensor_packets()
    assert reads == [f'auxiliary_depth_{f.index:04d}.npz']*2
    reads.clear(); f.calls.clear()
    actual = candidate.sensor_packets()
    assert reads == [f'auxiliary_depth_{f.index:04d}.npz']
    assert f.calls == ['primary', 'capture']
    same(expected, actual); same(before, f.policy)
    assert actual[0] is f.policy and actual[1] is f.primary_depth and actual[2] is f.fast
    assert not actual[3]['valid'][0, 0] and actual[3]['depth_m'][0, 0] == 0.
    assert 'world_from_optical' not in actual[3] and 'world_from_optical' not in actual[4]
    assert candidate.auxiliary_audit == baseline.auxiliary_audit
    # Each call rereads its current bound arrays; it does not cache mutable packets.
    actual[3]['depth_m'][1, 1] = 4.; actual[4]['rgb'][1, 1] = 0
    f.calls.clear(); reads.clear()
    again = candidate.sensor_packets()
    same(expected, again)
    assert f.calls == ['primary'] and len(reads) == 1


@pytest.mark.parametrize('fault', ['native_hash', 'rgb_hash', 'depth_mask', 'depth_values',
    'rgb_pixels', 'frame', 'calibration', 'stale', 'physical_pair', 'primary_index',
    'auxiliary_gap', 'auxiliary_extra'])
def test_candidate_and_original_reject_corrupt_or_misaligned_acquisition(fixture, fault):
    f = fixture
    if fault == 'native_hash': f.row['native_depth_sha256'] = '0'*64
    if fault == 'rgb_hash': f.row['rgb_sha256'] = '0'*64
    if fault == 'depth_mask': f.auxiliary['valid'][0, 0] = True; f.persist()
    if fault == 'depth_values': f.auxiliary['depth_m'][1, 1] = 4.; f.persist()
    if fault == 'rgb_pixels': f.rgb[1, 1] = 0; f.persist()
    if fault == 'frame': f.row['frame'] += 1
    if fault == 'calibration': f.row['calibration_id'] = 'incorrect'
    if fault == 'stale': f.row['measured_ns'] -= 1
    if fault == 'physical_pair': f.row['physical_sample_index'] += 1
    for cls in (new.extended.ExtendedBudgetDualSession, new.SingleReadAuxiliaryDualSession):
        session = f.session(cls)
        if fault == 'primary_index': session.model_manifest.append({})
        if fault == 'auxiliary_gap': session.auxiliary_audit.clear()
        if fault == 'auxiliary_extra': session.auxiliary_audit.extend([{}, {}])
        with pytest.raises(ValueError): session.sensor_packets()


@pytest.mark.parametrize('index', [-1, 0, 3014, 4013, 4014])
def test_renderer_mro_keeps_budget_guard_and_terminal_failure_latch(monkeypatch, index):
    calls = []
    class PrimaryReached(Exception): pass
    def primary(session):
        calls.append('primary')
        raise PrimaryReached()
    monkeypatch.setattr(new.NovelMazeBaseSession, 'sensor_packets', primary)
    session = object.__new__(new.SingleReadAuxiliaryRendererSession)
    session.samples = range(750+50*index)
    session.renderer_witnesses = dict(primary=[], paired=[], failures=[])
    if 0 <= index < 4014:
        with pytest.raises(PrimaryReached): session.sensor_packets()
        assert calls == ['primary']
    else:
        with pytest.raises(ValueError, match='bounded prospective'): session.sensor_packets()
        assert not calls
    assert len(session.renderer_witnesses['failures']) == 1
    prior = list(calls)
    with pytest.raises(ValueError, match='terminal'): session.sensor_packets()
    assert calls == prior


def install_witnesses(f, monkeypatch):
    e = new.extended
    identity = dict(camera_uid='synthetic', framebuffer_matches_camera_depth_target=True,
        raster_error_bound_proven=False, source_implementation_equivalence_proven=False)
    monkeypatch.setitem(e.capture_witness.__globals__, 'renderer_identity_readback',
        lambda camera: deepcopy(identity))
    monkeypatch.setitem(e.capture_witness.__globals__, 'sampling_readback',
        lambda camera: deepcopy(e.witness.SAMPLING))
    monkeypatch.setitem(e.capture_witness.__globals__, 'precision_readback', lambda camera: {})
    session = f.session(new.SingleReadAuxiliaryRendererSession)
    session.ctx = NS(build=NS(camera=NS(uid='synthetic', transform=np.eye(4))),
        runner=NS(_sim_time_ns=f.now))
    session.depth_audit = [{}, dict(native_depth_sha256='b'*64)]
    primary = e.capture_witness(session, frame=f.index, phase=e.witness.PHASES[0],
        pixel_hashes=dict(primary_rgb_sha256=f.image['primary_rgb_sha256'], primary_native_depth_sha256='b'*64))
    session.renderer_witnesses = dict(primary=[{}, primary], paired=[{}], failures=[])
    return session


def test_complete_packet_path_preserves_paired_witness_and_detects_drift(fixture, monkeypatch):
    f = fixture; session = install_witnesses(f, monkeypatch)
    packets = session.sensor_packets()
    assert len(session.renderer_witnesses['paired']) == 2
    paired = session.renderer_witnesses['paired'][f.index]
    assert paired['pixel_hashes']['auxiliary_rgb_sha256'] == f.row['rgb_sha256']
    assert paired['pixel_hashes']['auxiliary_native_depth_sha256'] == f.row['native_depth_sha256']
    assert paired['pixel_hashes']['primary_rgb_sha256'] == packets[4]['primary_rgb_sha256']
    new.extended.witness.validate_pair(session.renderer_witnesses['primary'][f.index], paired)
    assert not session.renderer_witnesses['failures']
    before = deepcopy(session.renderer_witnesses)
    session.sensor_packets()
    assert before == session.renderer_witnesses
    fresh = install_witnesses(f, monkeypatch)
    fresh.ctx.build.camera.transform[0, 3] = .1
    with pytest.raises(ValueError, match='endpoint drift'): fresh.sensor_packets()
    assert len(fresh.renderer_witnesses['failures']) == 1


def test_corrupt_frame_latches_before_any_reacquisition(fixture, monkeypatch):
    f = fixture; session = install_witnesses(f, monkeypatch)
    f.row['native_depth_sha256'] = '0'*64
    with pytest.raises(ValueError, match='raw array'): session.sensor_packets()
    previous = list(f.calls)
    with pytest.raises(ValueError, match='terminal'): session.sensor_packets()
    assert f.calls == previous and len(session.renderer_witnesses['failures']) == 1
