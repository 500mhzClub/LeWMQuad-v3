"""Reconstruct the failed short-gap prefix solely to retain its inner cause.

The camera schedule and tracker are unchanged. This is a read-only sensor
diagnostic, not a replacement navigation attempt or a new parameter trial.
"""
from copy import deepcopy
from pathlib import Path
import sys

from lewm.eligible_floor_registration_development import bind
from scripts import check_gapped_camera_recorded_prefix_development as prefix

OUTPUT = prefix.BASE/'go2_gapped_camera_short_gap_failure_diagnostic_v1_attempt_001'
_write = bind(prefix.write, OUTPUT=OUTPUT)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(diagnostic_source_sha256=prefix.source.digest(Path(__file__)),
            purpose='capture inner exception from the unchanged failed frame-684 prefix',
            original_failure_preserved='go2_gapped_camera_short_gap_journey_v1_attempt_001')
    elif name == 'failure.json':
        error = sys.exc_info()[1]; chain = []; current = error
        while current is not None:
            chain.append(dict(type=type(current).__name__, message=str(current)))
            current = current.__cause__
        tb = error.__traceback__; tracker = None
        while tb is not None:
            if 'tracker' in tb.tb_frame.f_locals:
                tracker = tb.tb_frame.f_locals['tracker']; break
            tb = tb.tb_next
        value = value | dict(exception_chain=chain)
        if tracker is not None:
            value['tracker_state'] = deepcopy(dict(frame=tracker.frame,
                previous_frame=tracker.previous.frame,
                previous_measured_ns=tracker.previous.measured_ns,
                continuity=tracker.last_continuity, camera_selection=tracker.last_camera_selection,
                direct_flow=tracker.last_direct_flow_fallback,
                chained_flow=tracker.last_chained_anchor_fallback,
                plane_conflict=tracker._plane_conflict,
                current_plane=None if tracker._pending_plane is None else tracker._pending_plane[2],
                bridge_frames=tracker.bridge_frames))
    _write(name, value)


main = bind(prefix.main, COUNT=685, GAPS=(1, 2, 3), OUTPUT=OUTPUT, write=write)


if __name__ == '__main__':
    main()
