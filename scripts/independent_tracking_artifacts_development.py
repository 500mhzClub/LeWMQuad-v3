"""Explicit, exclusive, bounded persistence for a new tracking episode.

No directory discovery, source export, launcher, retries or predecessor access.
External renderer writes require a reservation and exact post-write accounting;
they are not OS-quota-enforced and a faulty renderer can exceed its reservation.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np

from lewm.independent_tracking_challenge_development import TRIALS, MAX_FRAMES
from lewm.independent_tracking_recording_budget_development import numeric_envelope,reservation_from_nbytes
from scripts.navigation_artifact_root_development import validate_root

EPISODE_BYTES = 5 * 1024**3
RESERVE_BYTES = 40 * 1024**3
DEFERRED_BYTES = 2 * 1024**3
FRAME_BYTES = 6 * 1024**2
MESH_BYTES = 128 * 1024**2
SETUP_BYTES = 8 * 1024**2
JSON_BYTES = 32 * 1024**2
STATIC = (
    'specification.json', 'actuator_identity.json', 'floor_roles.json',
    'terminal_actuator_gains.json', 'terminal_native_robot_geometry.json',
    'terminal_environment_identity.json', 'result.json', 'failure.json',
    'physics_trace.npz', 'native_contacts.npz', 'contact_events.json', 'contact_topology.json',
    'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json', 'camera_audit.json',
    'depth_observations.json', 'depth_camera_audit.json', 'fast_gyro_samples.npz', 'fast_gyro_histories.npz',
    'floor_visual_collision_identity.json', 'tracking_decisions.json', 'command_tape.json',
    'native_guard_rows.json', 'friction_checks.json', 'static_objects.json',
    'startup_native_robot_geometry.json', 'setup_checks.json', 'persistence.json',
    'visual_meshes/ground_visual.ply', 'visual_meshes/wall_union_visual.ply',
)


def episode_resource_contract():
    """Cover all declared operations, without asserting native/metadata bounds.

    Deferred recording still requires source/memory review before native launch.
    This arithmetic is an allocation envelope, not a measured compression ratio
    or evidence that a renderer respects its external-write reservation.
    """
    external = MESH_BYTES + SETUP_BYTES + MAX_FRAMES * FRAME_BYTES
    if external + DEFERRED_BYTES > EPISODE_BYTES:
        raise ValueError('complete episode reservations exceed episode allocation')
    internal=sum(r['serialization_ceiling_bytes'] for r in numeric_envelope().values())
    internal+=sum(name.endswith('.json') for name in STATIC)*JSON_BYTES
    if internal>DEFERRED_BYTES:
        raise ValueError('bounded static recording outputs exceed deferred allocation')
    return dict(episode_bytes=EPISODE_BYTES, maximum_frames=MAX_FRAMES,
        frame_reservation_bytes=FRAME_BYTES, mesh_reservation_bytes=MESH_BYTES,
        setup_reservation_bytes=SETUP_BYTES, deferred_reservation_bytes=DEFERRED_BYTES,
        all_external_reservations_bytes=external,
        all_reservations_bytes=external+DEFERRED_BYTES,
        static_recording_serialization_ceiling_bytes=internal,
        json_file_ceiling_bytes=JSON_BYTES,
        internal_serialized_file_ceilings_enforced=True,
        native_recording_bound_proved=False, memory_bound_proved=False,
        external_os_quota_enforced=False)


def frame_names(index):
    if type(index) is not int or not 0 <= index < MAX_FRAMES:
        raise ValueError('bounded frame index required')
    return tuple(f'{prefix}_{index:04d}.{suffix}' for prefix, suffix in (
        ('rgb', 'png'), ('depth', 'npz'), ('native_depth', 'npz'), ('raster', 'json')))


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError('unsupported artifact JSON value: ' + type(value).__name__)


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=serial) + '\n').encode()


class StorageStop(RuntimeError):
    pass


def bounded_json(value):
    """Bound accumulated output without building a complete oversized payload.

    The encoder can still create a large single string/default-conversion chunk;
    this is not an object-graph or process-memory quota. Never truncate a record.
    """
    buffer=io.BytesIO()
    encoder=json.JSONEncoder(sort_keys=True,indent=2,allow_nan=False,default=serial)
    for chunk in encoder.iterencode(value):
        payload=chunk.encode()
        if buffer.tell()+len(payload)+1>JSON_BYTES:raise StorageStop('JSON_RECORDING_BYTE_CEILING')
        buffer.write(payload)
    buffer.write(b'\n')
    return buffer.getvalue()


class EpisodeStore:
    def __init__(self, output, trial):
        root = validate_root(output)
        if trial not in TRIALS:
            raise ValueError('exact challenge trial required')
        self.directory = root / trial
        if self.directory.exists() or self.directory.is_symlink():
            raise ValueError('exclusive fresh episode; no retry or resume')
        self.directory.mkdir()
        self.allowed = set(STATIC) | {n for i in range(MAX_FRAMES) for n in frame_names(i)}
        self.bindings = {}; self.sizes = {}; self.failed_external = False; self.failed_internal = False

    @property
    def used(self):
        return sum(self.sizes.values())

    def path(self, name):
        if type(name) is not str or name not in self.allowed:
            raise ValueError('explicit challenge artifact required')
        p = self.directory / name
        if self.directory.resolve() != self.directory or p.resolve() != p:
            raise ValueError('symlink artifact forbidden')
        return p

    def check(self, allocation, *, deferred=DEFERRED_BYTES):
        if any(type(v) is not int or v < 0 for v in (allocation, deferred)):
            raise ValueError('nonnegative integer allocation required')
        required = allocation + deferred
        if self.used + required > EPISODE_BYTES:
            raise StorageStop('EPISODE_ARTIFACT_BUDGET_STOP')
        if shutil.disk_usage(self.directory).free < RESERVE_BYTES + required:
            raise StorageStop('STORAGE_RESERVE_STOP')

    def write_bytes(self, name, value):
        if type(value) is not bytes:
            raise ValueError('pre-serialized bytes required')
        p = self.path(name)
        ceiling=JSON_BYTES if name.endswith('.json') else (
            numeric_envelope()[name]['serialization_ceiling_bytes'] if name in numeric_envelope() else None)
        if ceiling is not None and len(value)>ceiling:
            raise StorageStop('SERIALIZED_RECORDING_FILE_CEILING')
        if name in self.bindings or p.exists():
            raise ValueError('artifact overwrite or second write forbidden')
        self.check(len(value), deferred=0)
        try:
            with p.open('xb') as stream:
                if stream.write(value) != len(value): raise OSError('short artifact write')
                stream.flush(); os.fsync(stream.fileno())
            descriptor = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        except BaseException:
            self.failed_internal = True
            raise
        finally:
            # A failed write/flush can still leave evidence. Bind its actual
            # bytes, not the intended payload; never retry or overwrite it.
            if p.is_file():
                self.path(name)
                self.sizes[name] = p.stat().st_size
                with p.open('rb') as stream:
                    self.bindings[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if self.sizes[name] != len(value) or self.bindings[name] != hashlib.sha256(value).hexdigest():
            self.failed_internal = True
            raise ValueError('written artifact differs from serialized payload')

    def json(self, name, value):
        self.write_bytes(name, bounded_json(value))

    def npz(self, name, arrays):
        # Reserve a conservative compression/metadata allowance and check the
        # actual serialized size again. This is not a memory quota.
        if (not isinstance(arrays, dict) or any(not isinstance(k, str) or not k.isidentifier() or len(k) > 128
                                               for k in arrays)):
            raise ValueError('explicit numeric array members required')
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        if any(v.dtype.hasobject or v.ndim > 8 for v in arrays.values()):
            raise ValueError('bounded nonobject arrays required')
        estimate = reservation_from_nbytes([v.nbytes for v in arrays.values()])
        if name in numeric_envelope() and estimate>numeric_envelope()[name]['serialization_ceiling_bytes']:
            raise StorageStop('NUMERIC_RECORDING_BYTE_CEILING')
        self.check(estimate, deferred=0)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        if len(payload) > estimate:
            raise StorageStop('SERIALIZED_ARRAY_BOUND_EXCEEDED')
        self.write_bytes(name, payload)

    @contextmanager
    def external(self, names, allowance):
        """Reserve before native writer; retain partial files even on exceptions.

        Missing files are allowed only for a failed writer. A second external
        operation is forbidden after failure. Rechecking adopted hashes happens
        in verify(), not by re-opening already written paths for replacement.
        """
        names = tuple(names)
        if not names or len(set(names)) != len(names) or self.failed_external:
            raise ValueError('new explicit nonempty external operation required')
        paths = {n: self.path(n) for n in names}
        if any(p.exists() or n in self.bindings for n, p in paths.items()):
            raise ValueError('external overwrite forbidden')
        self.check(allowance)
        failed = False
        try:
            yield
        except BaseException:
            failed = self.failed_external = True
            raise
        finally:
            written = 0; missing = []; sync_errors = []
            for name, p in paths.items():
                self.path(name)
                if not p.exists(): missing.append(name); continue
                if not p.is_file() or p.stat().st_uid != os.getuid():
                    self.failed_external = True
                    raise ValueError('owned regular native artifact required')
                size = p.stat().st_size
                with p.open('rb') as stream:
                    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                    self.sizes[name] = size; self.bindings[name] = sha
                    try: os.fsync(stream.fileno())
                    except OSError as error: sync_errors.append((name, repr(error)))
                written += size
            if sync_errors:
                self.failed_external = True
                raise OSError('external evidence synchronization failed: ' + repr(sync_errors))
            if written > allowance:
                self.failed_external = True
                raise StorageStop('EXTERNAL_WRITER_EXCEEDED_RESERVED_BYTES')
            if missing and not failed:
                self.failed_external = True
                raise ValueError('external writer omitted expected artifacts: ' + repr(missing))

    def verify(self):
        for name, sha in self.bindings.items():
            p = self.path(name)
            with p.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if p.stat().st_size != self.sizes[name] or actual != sha:
                raise ValueError('persisted artifact changed: ' + name)
        if self.used > EPISODE_BYTES:
            raise StorageStop('TERMINAL_EPISODE_BUDGET_EXCEEDED')
        return dict(artifact_sha256=dict(self.bindings), artifact_bytes=self.used,
                    artifact_sizes=dict(self.sizes), failed_external=self.failed_external,
                    failed_internal=self.failed_internal)
