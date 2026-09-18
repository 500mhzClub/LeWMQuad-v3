"""Lossless native storage, public reconstruction and bounded malformed input."""
from copy import deepcopy
from io import BytesIO
import zipfile

import numpy as np
import pytest

from scripts import compact_native_depth_archive_development as compact
from lewm.causal_depth_observation_development import from_native_depth as primary_packet
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_packet
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.tests.test_causal_auxiliary_rgb_observation_development import inputs
from lewm.tests.test_single_read_auxiliary_maze_session_development import same


def arrays():
    native = np.full(compact.SHAPE, 2., np.float32)
    native.flat[:9] = [np.nan, np.inf, -np.inf, -0., .199, .2, 5., 5.001, -1.]
    native.view(np.uint32).flat[0] = 0x7fc01234
    segmentation = np.arange(np.prod(compact.SHAPE), dtype=np.int64).reshape(compact.SHAPE) % 23
    segmentation[0, 0] = -1
    return native, segmentation


def test_bitwise_round_trip_and_complete_existing_public_packet_reconstruction(tmp_path):
    policy, _, _, now, _, rgb = inputs()
    native, segmentation = arrays()
    original_native = native.tobytes(); original_segmentation = segmentation.tobytes()
    bindings = {}
    for role, constructor in (('primary', primary_packet), ('auxiliary', auxiliary_packet)):
        binding, evaluator = compact.write(tmp_path, role=role, frame=1, native_depth=native,
            diagnostic_segmentation=segmentation if role == 'auxiliary' else None)
        bindings[role] = binding
        decoded = compact.read_native(tmp_path, binding)
        assert decoded.tobytes() == original_native
        assert not np.shares_memory(decoded, native)
        expected = constructor(native, policy, measured_ns=now, available_ns=now, now_ns=now)
        actual = constructor(decoded, policy, measured_ns=now, available_ns=now, now_ns=now)
        same(expected, actual)
        if role == 'auxiliary':
            same(from_captured_rgb(rgb, expected, policy, measured_ns=now, available_ns=now, now_ns=now),
                 from_captured_rgb(rgb, actual, policy, measured_ns=now, available_ns=now, now_ns=now))
            recovered = compact.read_evaluator_segmentation(tmp_path, binding, evaluator)
            assert recovered.tobytes() == original_segmentation
            assert recovered.dtype == segmentation.dtype
        else:
            assert evaluator is None
        decoded.fill(7)
        assert compact.read_native(tmp_path, binding).tobytes() == original_native
        assert native.tobytes() == original_native and segmentation.tobytes() == original_segmentation
    # Equivalent legacy payloads contain both raw and derivable arrays.
    primary = primary_packet(native, policy, measured_ns=now, available_ns=now, now_ns=now)
    auxiliary = auxiliary_packet(native, policy, measured_ns=now, available_ns=now, now_ns=now)
    np.savez_compressed(tmp_path/'legacy_native.npz', optical_depth_m=native)
    np.savez_compressed(tmp_path/'legacy_primary.npz', depth_m=primary['depth_m'], valid=primary['valid'])
    np.savez_compressed(tmp_path/'legacy_auxiliary.npz', native_optical_depth_m=native,
        depth_m=auxiliary['depth_m'], valid=auxiliary['valid'], diagnostic_segmentation=segmentation)
    old_bytes = sum((tmp_path/name).stat().st_size for name in (
        'legacy_native.npz', 'legacy_primary.npz', 'legacy_auxiliary.npz'))
    assert sum(binding['archive_bytes'] for binding in bindings.values()) < old_bytes


def test_public_reader_never_decodes_segmentation_and_evaluator_is_separate(tmp_path, monkeypatch):
    native, segmentation = arrays()
    binding, evaluator = compact.write(tmp_path, role='auxiliary', frame=0,
        native_depth=native, diagnostic_segmentation=segmentation)
    opened = []; original = zipfile.ZipFile.open
    def tracked(archive, name, *args, **kwargs):
        opened.append(name)
        return original(archive, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'open', tracked)
    compact.read_native(tmp_path, binding)
    assert opened == [compact.NATIVE]
    opened.clear()
    compact.read_evaluator_segmentation(tmp_path, binding, evaluator)
    assert opened == [compact.SEGMENTATION]


@pytest.mark.parametrize('frame', [0, 4013, 8013])
def test_valid_boundary_frames_and_exclusive_preservation(tmp_path, frame):
    native, _ = arrays()
    binding, _ = compact.write(tmp_path, role='primary', frame=frame, native_depth=native)
    path = tmp_path/binding['filename']; before = path.read_bytes()
    with pytest.raises(FileExistsError):
        compact.write(tmp_path, role='primary', frame=frame, native_depth=native+1)
    assert path.read_bytes() == before
    assert compact.read_native(tmp_path, binding).tobytes() == native.tobytes()


@pytest.mark.parametrize('fault', ['negative', 'too_many', 'bool_frame', 'float_frame', 'role',
    'native_dtype', 'native_shape', 'native_object', 'missing_seg', 'seg_float', 'seg_shape', 'primary_seg'])
def test_invalid_capture_rejected_before_any_file_write(tmp_path, fault):
    native, seg = arrays()
    kwargs = dict(role='auxiliary', frame=1, native_depth=native, diagnostic_segmentation=seg)
    if fault == 'negative': kwargs['frame'] = -1
    if fault == 'too_many': kwargs['frame'] = 8014
    if fault == 'bool_frame': kwargs['frame'] = True
    if fault == 'float_frame': kwargs['frame'] = 1.
    if fault == 'role': kwargs['role'] = 'unknown'
    if fault == 'native_dtype': kwargs['native_depth'] = native.astype(np.float64)
    if fault == 'native_shape': kwargs['native_depth'] = native[:2]
    if fault == 'native_object': kwargs['native_depth'] = native.astype(object)
    if fault == 'missing_seg': kwargs['diagnostic_segmentation'] = None
    if fault == 'seg_float': kwargs['diagnostic_segmentation'] = seg.astype(float)
    if fault == 'seg_shape': kwargs['diagnostic_segmentation'] = seg[:2]
    if fault == 'primary_seg': kwargs['role'] = 'primary'
    with pytest.raises(ValueError): compact.write(tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fault', ['frame', 'role', 'filename', 'extra', 'native_sha', 'archive_sha', 'size'])
def test_binding_tampering_rejected(tmp_path, fault):
    native, _ = arrays()
    binding, _ = compact.write(tmp_path, role='primary', frame=1, native_depth=native)
    bad = deepcopy(binding)
    if fault == 'frame': bad['frame'] = True
    if fault == 'role': bad['role'] = 'auxiliary'
    if fault == 'filename': bad['filename'] = '../escape.npz'
    if fault == 'extra': bad['world_from_optical'] = 'not part of storage binding'
    if fault == 'native_sha': bad['native_depth_sha256'] = '0'*64
    if fault == 'archive_sha': bad['archive_sha256'] = '0'*64
    if fault == 'size': bad['archive_bytes'] += 1
    with pytest.raises(ValueError): compact.read_native(tmp_path, bad)


def test_changed_archive_rejected_before_zip_decode(tmp_path, monkeypatch):
    native, _ = arrays()
    binding, _ = compact.write(tmp_path, role='primary', frame=1, native_depth=native)
    path = tmp_path/binding['filename']; data = bytearray(path.read_bytes()); data[-10] ^= 1
    path.write_bytes(data)
    def forbidden(*args, **kwargs): raise AssertionError('decode reached')
    monkeypatch.setattr(compact.zipfile, 'ZipFile', forbidden)
    with pytest.raises(ValueError, match='differs from its binding'): compact.read_native(tmp_path, binding)


@pytest.mark.parametrize('fault', ['wrong_frame', 'bool_frame', 'sha', 'dtype', 'archive', 'extra'])
def test_evaluator_identity_cannot_be_substituted(tmp_path, fault):
    native, seg = arrays()
    binding, evaluator = compact.write(tmp_path, role='auxiliary', frame=0,
        native_depth=native, diagnostic_segmentation=seg)
    if fault == 'wrong_frame': evaluator['frame'] = 1
    if fault == 'bool_frame': evaluator['frame'] = False
    if fault == 'sha': evaluator['segmentation_sha256'] = '0'*64
    if fault == 'dtype': evaluator['dtype'] = '<u8'
    if fault == 'archive': evaluator['archive_sha256'] = '0'*64
    if fault == 'extra': evaluator['world_from_optical'] = 'private'
    with pytest.raises(ValueError): compact.read_evaluator_segmentation(tmp_path, binding, evaluator)


@pytest.mark.parametrize('fault', ['huge_shape', 'object', 'fortran', 'trailing', 'extra_member', 'missing_member'])
def test_malformed_numpy_or_member_structure_rejected_before_allocation(tmp_path, fault):
    native, _ = arrays()
    header = dict(descr=np.dtype(np.float32).str, fortran_order=False, shape=compact.SHAPE)
    if fault == 'huge_shape': header['shape'] = (10**12,)
    if fault == 'object': header['descr'] = '|O'
    if fault == 'fortran': header['fortran_order'] = True
    payload = BytesIO(); np.lib.format.write_array_header_1_0(payload, header)
    payload.write(native.tobytes())
    if fault == 'trailing': payload.write(b'extra')
    path = tmp_path/'compact_primary_depth_0000.npz'
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        if fault != 'missing_member': archive.writestr(compact.NATIVE, payload.getvalue())
        if fault == 'extra_member': archive.writestr('valid.npy', b'extra')
    binding = dict(schema=compact.SCHEMA, role='primary', frame=0, filename=path.name,
        archive_bytes=path.stat().st_size, archive_sha256=compact.digest(path.read_bytes()),
        native_depth_sha256=compact.digest(native.tobytes()))
    with pytest.raises(ValueError): compact.read_native(tmp_path, binding)


def test_protected_names_and_symlink_leaves_rejected_before_access(tmp_path):
    native, _ = arrays()
    protected = tmp_path/'sealed_synthetic'
    with pytest.raises(ValueError, match='protected'):
        compact.write(protected, role='primary', frame=0, native_depth=native)
    # The protected path is never created or opened.
    target = tmp_path/'untouched'; target.write_bytes(b'keep')
    (tmp_path/'compact_primary_depth_0000.npz').symlink_to(target)
    with pytest.raises(ValueError, match='symlink'):
        compact.write(tmp_path, role='primary', frame=0, native_depth=native)
    assert target.read_bytes() == b'keep'
