"""Perturb live depth once; retain compact native pixels and exact noisy digests."""
import json
import time

from lewm.eligible_floor_registration_development import bind
from scripts.compact_depth_retention_session_development import persist_compact_pair
from scripts.in_memory_paired_camera_session_development import (
    InMemoryPairedCameraSession, write as write_json)
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.raw_depth_archive_development import packet_digest
from scripts.replay_go2_depth_noise_tracking_development import SEED, perturbed_packet

SCHEMA = 'live_synthetic_depth_noise_native_pixels_development.v1'


def persist_pair(row, directory, **kwargs):
    return persist_compact_pair(row, directory, **kwargs) | dict(
        live_depth_noise=row['live_depth_noise'])


def write_metadata(path, value):
    if path.name == 'in_memory_camera_observations.json':
        value = value | dict(schema=SCHEMA,
            stored_native_depth_is_unperturbed=True,
            live_depth_requires_recorded_perturbation=True,
            replay_class='scripts.live_depth_noise_session_development.NoisyPublicReplay')
    write_json(path, value)


class LiveDepthNoiseMixin:
    def __init__(self, *args, noise_layout_index, noise_sigma_mm, **kwargs):
        if noise_layout_index not in range(4) or noise_sigma_mm not in (0, 2):
            raise ValueError('fixed development noise assignment required')
        self.noise_layout_index = noise_layout_index
        self.noise_sigma_mm = noise_sigma_mm
        super().__init__(*args, **kwargs)

    def sensor_packets(self):
        # Native session order differs from the public replay tuple order.
        p, d, fast, auxiliary, rgb, now = super().sensor_packets()
        started = time.perf_counter_ns()
        row = self.captured_pairs[-1]
        p, d, fast, rgb, auxiliary, now = perturbed_packet(
            (p, d, fast, rgb, auxiliary, now), layout=self.noise_layout_index,
            frame=row['frame'], sigma_m=self.noise_sigma_mm / 1000)
        row['live_depth_noise'] = dict(seed=SEED, layout_index=self.noise_layout_index,
            sigma_mm=self.noise_sigma_mm, primary_sha256=packet_digest(d),
            auxiliary_sha256=packet_digest(auxiliary))
        row['acquisition_wall_ms'] += (time.perf_counter_ns() - started) / 1e6
        return p, d, fast, auxiliary, rgb, now

    persist_observations = bind(InMemoryPairedCameraSession.persist_observations,
        persist_camera_pair=persist_pair, write=write_metadata)


class NoisyPublicReplay(PublicReplay):
    """Reconstruct the packets actually delivered, checking both camera digests.

    The ordinary reader rejects this distinct schema rather than silently
    replaying the unperturbed recording as the live sensor stream.
    """
    def __init__(self, directory):
        bind(PublicReplay.__init__, RAW_DEPTH_SCHEMA=SCHEMA)(self, directory)
        metadata = json.loads((self.directory / 'in_memory_camera_observations.json').read_text())
        if metadata['schema'] != SCHEMA:
            raise ValueError('explicit live-noise recording required')
        self.noise_rows = metadata['frames']

    def packet(self, frame):
        recipe = self.noise_rows[frame]['live_depth_noise']
        if recipe['seed'] != SEED:
            raise ValueError('recorded noise seed differs from implementation')
        result = perturbed_packet(super().packet(frame), layout=recipe['layout_index'],
            frame=frame, sigma_m=recipe['sigma_mm'] / 1000)
        if (packet_digest(result[1]) != recipe['primary_sha256']
                or packet_digest(result[4]) != recipe['auxiliary_sha256']):
            raise ValueError('reconstructed live depth differs from delivered packets')
        return result
