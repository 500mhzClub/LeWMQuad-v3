"""Post-execution RGB/physics retention without redundant raw depth archives.

Live acquisition and perception are unchanged. Depth hashes and noise recipes
remain recorded, but direct depth replay requires regeneration and is not
claimed to be bitwise reproducible. No existing artifact is deleted.
"""
import hashlib
from PIL import Image

from lewm.eligible_floor_registration_development import bind
from scripts.in_memory_paired_camera_session_development import InMemoryPairedCameraSession, write


def persist_rgb_pair(row, directory, *, native_depth_only, **kwargs):
    if not native_depth_only:
        raise ValueError('compact native-depth recording source required')
    hashes = {}
    frame = row['frame']
    for label, (rgb, native), digest in zip(('primary','auxiliary'), row['images'],
            (row['depth'],row['auxiliary_depth']), strict=True):
        if not isinstance(digest,str) or len(digest)!=64:
            raise ValueError('captured derived-depth packet digest required')
        name = f'rgb_{frame:04d}.png' if label=='primary' else f'auxiliary_rgb_{frame:04d}.png'
        Image.fromarray(rgb).save(directory/name, compress_level=1)
        hashes[label] = dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
            derived_packet_sha256=digest)
    return {k:row[k] for k in ('frame','measured_ns','physical_sample_index','transforms','acquisition_wall_ms')} | dict(
        pixel_sha256=hashes, live_depth_noise=row['live_depth_noise'], native_depth_arrays_saved=False)


def write_rgb_metadata(path, value):
    if path.name=='in_memory_camera_observations.json':
        value = dict(value)
        for key in ('lossless_archive_compression_level','lossless_archive_compression_method',
                'stored_native_depth_is_unperturbed','replay_class'):
            value.pop(key,None)
        workers = value.pop('post_run_archive_workers',None)
        value = value | dict(schema='rgb_and_depth_hashes_navigation_development.v1',
            native_depth_arrays_saved=False, live_noisy_depth_arrays_saved=False,
            live_depth_requires_recorded_perturbation=True, direct_depth_replay_available=False,
            depth_regeneration_bitwise_equivalence_established=False,
            retention_applied_after_physical_execution=True,
            rgb_png_compression_level=1, post_run_recording_workers=workers)
    write(path,value)


class RGBNavigationRetentionMixin:
    persist_observations = bind(InMemoryPairedCameraSession.persist_observations,
        persist_camera_pair=persist_rgb_pair, write=write_rgb_metadata)
