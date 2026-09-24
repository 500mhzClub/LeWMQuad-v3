"""Keep native pixels and captured packet digests instead of duplicate depth arrays."""
import time

from lewm.eligible_floor_registration_development import bind
from scripts.in_memory_paired_camera_session_development import (
    InMemoryPairedCameraSession, persist_camera_pair,
)
from scripts.raw_depth_archive_development import packet_digest


def compact_depth_row(row):
    """Hash the actual packets before releasing only their recording references."""
    for key in ('depth', 'auxiliary_depth'):
        row[key] = packet_digest(row[key])


def saved_digest(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError('captured depth packet SHA-256 required')
    return value


persist_hashed_pair = bind(persist_camera_pair, packet_digest=saved_digest)


def persist_compact_pair(row, directory, **kwargs):
    if kwargs.get('native_depth_only') is not True:
        raise ValueError('compact retention requires the native-depth-only archive')
    return persist_hashed_pair(row, directory, **kwargs)


class CompactDepthRetentionMixin:
    """Same live packets; charge capture-time hashing to measured execution."""
    def sensor_packets(self):
        packets = super().sensor_packets()
        started = time.perf_counter_ns()
        row = self.captured_pairs[-1]
        compact_depth_row(row)
        row['acquisition_wall_ms'] += (time.perf_counter_ns()-started)/1e6
        return packets

    persist_observations = bind(InMemoryPairedCameraSession.persist_observations,
        persist_camera_pair=persist_compact_pair)
