"""Run remaining matched layouts with compact recordings and separate CPU groups."""
import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_matched_training_navigation_development as matched
from scripts.compact_depth_retention_session_development import CompactDepthRetentionMixin


CPU_GROUPS = {2: list(range(8)) + list(range(16, 24)),
              3: list(range(8, 16)) + list(range(24, 32))}


class CompactFreshCameraSession(CompactDepthRetentionMixin, matched.fresh.FreshCameraSession):
    """Preserve the fresh physical initializer and all live sensor packets."""


def finish_write(name, value):
    if name == 'launch.json':
        layout = value['layout_index']
        affinity = sorted(os.sched_getaffinity(0))
        if layout not in CPU_GROUPS or affinity != CPU_GROUPS[layout]:
            raise ValueError('remaining layout must use its assigned CPU group')
        profile = dict(recording='native_pixels_with_captured_derived_packet_digests',
            derived_depth_arrays_retained=False, packet_hashing_during_capture=True,
            hashing_cost_charged_to_execution=True, cpu_affinity=affinity,
            paired_layout_index=5-layout, maximum_planned_native_owners=2,
            actual_concurrent_owners_not_inferred_from_profile=True,
            real_time_qualified=False)
        value = value | dict(execution_profile=profile,
            extra_sources=value['extra_sources'] | {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'scripts/compact_depth_retention_session_development.py')})
        RAW_WRITE('parallel_execution_profile.json', profile)
    bind(matched.finish_write, TRAINING_CONDITION=TRAINING_CONDITION,
        CORRECTION_HASH=CORRECTION_HASH, CORRECTION_ROOT=CORRECTION_ROOT,
        RAW_WRITE=RAW_WRITE)(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(2, 3), required=True)
    parser.add_argument('--condition', choices=tuple(matched.FITS), required=True)
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != CPU_GROUPS[args.layout_index]:
        raise ValueError('launch with taskset using the assigned layout CPU group')
    fresh = SimpleNamespace(**(vars(matched.fresh) |
        dict(FreshCameraSession=CompactFreshCameraSession)))
    bind(matched.main, fresh=fresh, finish_write=finish_write)()


if __name__ == '__main__':
    main()
