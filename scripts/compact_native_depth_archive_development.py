"""Prospective raw-depth storage, separate public and evaluator array readers.

No capture session, replay or launcher adopts this format. Callers must bind
these receipts to the actual acquisition clock/calibration and whole-trial
artifact roster. Existing bound archives are never replaced or removed.
"""
from io import BytesIO
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from lewm.causal_rgb_dataset_development import _leaf, _protected

SCHEMA = 'compact_native_depth_archive.v1'
SEGMENTATION_SCHEMA = 'compact_native_depth_segmentation.v1'
SHAPE = (480, 640)
MAX_FRAMES = 8014
MAX_ARCHIVE_BYTES = 5*1024**2
MAX_HEADER_BYTES = 512
NATIVE = 'optical_depth_m.npy'
SEGMENTATION = 'diagnostic_segmentation.npy'
FIELDS = {'schema', 'role', 'frame', 'filename', 'archive_bytes',
          'archive_sha256', 'native_depth_sha256'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _sha(value):
    return (type(value) is str and len(value) == 64
        and all(c in '0123456789abcdef' for c in value))


def _name(role, frame):
    if (type(role) is not str or role not in ('primary', 'auxiliary')
            or type(frame) is not int or not 0 <= frame < MAX_FRAMES):
        raise ValueError('explicit camera role and bounded integer frame required')
    return f'compact_{role}_depth_{frame:04d}.npz'


def _path(directory, name):
    directory = Path(directory).absolute()
    if _protected(directory) or _protected(directory.resolve()):
        raise ValueError('protected depth archive path forbidden')
    directory = directory.resolve()
    if (directory/name).is_symlink():
        raise ValueError('depth archive must not be a symlink')
    return _leaf(directory, name)


def _array(value, *, segmentation=False):
    if (type(value) is not np.ndarray or value.shape != SHAPE
            or (value.dtype.kind not in 'iu' or value.dtype.itemsize > 8
                if segmentation else value.dtype != np.dtype(np.float32))):
        raise ValueError('exact raster shape and native array dtype required')
    # Preserve every bit, including nonfinite raw depth and integer labels.
    return value.copy(order='C')


def _binding(binding):
    if (type(binding) is not dict or set(binding) != FIELDS
            or binding['schema'] != SCHEMA
            or binding['filename'] != _name(binding['role'], binding['frame'])
            or type(binding['archive_bytes']) is not int
            or not 0 < binding['archive_bytes'] <= MAX_ARCHIVE_BYTES
            or not _sha(binding['archive_sha256']) or not _sha(binding['native_depth_sha256'])):
        raise ValueError('exact bounded compact depth archive binding required')


def _bytes(path):
    with path.open('rb') as stream:
        data = stream.read(MAX_ARCHIVE_BYTES+1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError('bounded compact depth archive required')
    return data


def write(directory, *, role, frame, native_depth, diagnostic_segmentation=None):
    """Create one new archive exclusively; return storage and evaluator receipts.

    No derived depth or validity array is stored. An auxiliary archive retains
    its complete diagnostic segmentation, read only by the evaluator function.
    Neither receipt replaces the caller's acquisition or artifact admission.
    """
    name = _name(role, frame)
    native = _array(native_depth)
    if role == 'primary' and diagnostic_segmentation is not None:
        raise ValueError('primary archive has no diagnostic segmentation')
    segmentation = _array(diagnostic_segmentation, segmentation=True) if role == 'auxiliary' else None
    arrays = dict(optical_depth_m=native)
    if segmentation is not None: arrays['diagnostic_segmentation'] = segmentation
    path = _path(directory, name)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    data = _bytes(path)
    binding = dict(schema=SCHEMA, role=role, frame=frame, filename=name,
        archive_bytes=len(data), archive_sha256=digest(data), native_depth_sha256=digest(native.tobytes()))
    evaluator = None if segmentation is None else dict(schema=SEGMENTATION_SCHEMA,
        frame=frame, archive_sha256=binding['archive_sha256'],
        dtype=segmentation.dtype.str, segmentation_sha256=digest(segmentation.tobytes()))
    return binding, evaluator


def _archive(directory, binding):
    _binding(binding)
    data = _bytes(_path(directory, binding['filename']))
    if len(data) != binding['archive_bytes'] or digest(data) != binding['archive_sha256']:
        raise ValueError('compact depth archive differs from its binding')
    archive = zipfile.ZipFile(BytesIO(data))
    try:
        expected = {NATIVE} if binding['role'] == 'primary' else {NATIVE, SEGMENTATION}
        infos = archive.infolist()
        if set(archive.namelist()) != expected or len(infos) != len(expected):
            raise ValueError('exact compact raw-only archive members required')
        for info in infos:
            limit = np.prod(SHAPE)*(4 if info.filename == NATIVE else 8)+MAX_HEADER_BYTES
            if (info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits & 1
                    or not 0 < info.file_size <= limit):
                raise ValueError('bounded unencrypted native NumPy members required')
        return archive
    except BaseException:
        archive.close()
        raise


def _decode(archive, name):
    with archive.open(name) as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError('explicit NumPy V1 native array header required')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream,
            max_header_size=MAX_HEADER_BYTES)
        valid_dtype = (dtype == np.dtype(np.float32) if name == NATIVE
            else dtype.kind in 'iu' and dtype.itemsize <= 8)
        if shape != SHAPE or fortran or not valid_dtype:
            raise ValueError('exact bounded native array shape and encoding required')
        size = int(np.prod(SHAPE))*dtype.itemsize
        if archive.getinfo(name).file_size != stream.tell()+size:
            raise ValueError('exact native array byte population required')
        data = stream.read(size+1)
        if len(data) != size: raise ValueError('complete native array payload required')
    return np.frombuffer(data, dtype=dtype).copy().reshape(SHAPE)


def read_native(directory, binding):
    """Decode raw optical depth only; never decode evaluator segmentation."""
    with _archive(directory, binding) as archive:
        native = _decode(archive, NATIVE)
    if digest(native.tobytes()) != binding['native_depth_sha256']:
        raise ValueError('native depth differs from its acquisition array identity')
    return native


def read_evaluator_segmentation(directory, binding, evaluator):
    """Separate evaluator-only entry point retaining full native integer labels."""
    _binding(binding)
    if (binding['role'] != 'auxiliary' or type(evaluator) is not dict
            or set(evaluator) != {'schema', 'frame', 'archive_sha256', 'dtype', 'segmentation_sha256'}
            or evaluator['schema'] != SEGMENTATION_SCHEMA
            or type(evaluator['frame']) is not int or evaluator['frame'] != binding['frame']
            or evaluator['archive_sha256'] != binding['archive_sha256']
            or type(evaluator['dtype']) is not str or not _sha(evaluator['segmentation_sha256'])):
        raise ValueError('exact separate evaluator segmentation binding required')
    with _archive(directory, binding) as archive:
        segmentation = _decode(archive, SEGMENTATION)
    if (segmentation.dtype.str != evaluator['dtype']
            or digest(segmentation.tobytes()) != evaluator['segmentation_sha256']):
        raise ValueError('diagnostic segmentation differs from its evaluator identity')
    return segmentation
