"""Recording compaction must preserve live packets and archived evidence."""
import copy
import hashlib
import zipfile

import numpy as np
import pytest

from scripts.compact_depth_retention_session_development import (
    CompactDepthRetentionMixin, compact_depth_row, persist_compact_pair,
)
from scripts.in_memory_paired_camera_session_development import persist_camera_pair
from scripts.raw_depth_archive_development import packet_digest


def test_compaction_preserves_live_packets_and_exact_archive(tmp_path):
    native = np.resize(np.array([np.nan, np.inf, -.1, .2, 1., 5., 6.],
                               dtype=np.float32), (480, 640))
    valid = np.isfinite(native) & (native >= .2) & (native <= 5.)
    depth = dict(schema='synthetic_depth', measured_ns=100,
        depth_m=np.where(valid, native, np.float32(0.)), valid=valid)
    auxiliary = dict(depth, schema='synthetic_auxiliary_depth')
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    row = dict(frame=0, measured_ns=100, physical_sample_index=0,
        transforms=[], acquisition_wall_ms=1., images=[(rgb, native), (rgb, native)],
        depth=depth, auxiliary_depth=auxiliary)
    compact = copy.copy(row)
    before = [packet_digest(depth), packet_digest(auxiliary)]
    compact_depth_row(compact)
    assert [compact['depth'], compact['auxiliary_depth']] == before
    assert row['depth'] is depth and row['auxiliary_depth'] is auxiliary
    assert [packet_digest(depth), packet_digest(auxiliary)] == before
    assert compact['images'] is row['images']
    assert depth['depth_m'].nbytes + depth['valid'].nbytes == 640*480*5
    old = tmp_path/'old'; new = tmp_path/'compact'
    old.mkdir(); new.mkdir()
    options = dict(native_depth_only=True, compression=zipfile.ZIP_LZMA, compresslevel=None)
    expected = persist_camera_pair(row, old, **options)
    actual = persist_compact_pair(compact, new, **options)
    assert actual == expected
    for name in ('rgb_0000.png', 'auxiliary_rgb_0000.png'):
        assert (old/name).read_bytes() == (new/name).read_bytes()
    for label in ('primary', 'auxiliary'):
        with zipfile.ZipFile(old/f'{label}_depth_0000.npz') as a, \
                zipfile.ZipFile(new/f'{label}_depth_0000.npz') as b:
            assert a.namelist() == b.namelist() == ['native_optical_depth_m.npy']
            assert a.read(a.namelist()[0]) == b.read(b.namelist()[0])
        assert actual['pixel_sha256'][label]['native_depth_sha256'] == hashlib.sha256(native.tobytes()).hexdigest()


def test_compact_archive_rejects_derived_array_output(tmp_path):
    with pytest.raises(ValueError, match='native-depth-only'):
        persist_compact_pair({}, tmp_path, native_depth_only=False)


def test_session_returns_original_packets_and_charges_hashing_time():
    depth = dict(depth_m=np.ones((480, 640), dtype=np.float32),
                 valid=np.ones((480, 640), dtype=bool))
    packets = (object(), depth, object(), depth.copy(), object(), 100)

    class Producer:
        def sensor_packets(self):
            self.captured_pairs = [dict(depth=packets[1], auxiliary_depth=packets[3],
                                       acquisition_wall_ms=2.)]
            return packets

    class Session(CompactDepthRetentionMixin, Producer):
        pass

    session = Session()
    returned = session.sensor_packets()
    assert returned is packets
    assert returned[1]['depth_m'] is depth['depth_m']
    assert returned[3]['valid'] is depth['valid']
    assert session.captured_pairs[0]['depth'] == packet_digest(depth)
    assert session.captured_pairs[0]['acquisition_wall_ms'] > 2.
