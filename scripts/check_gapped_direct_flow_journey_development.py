"""Test actual-interval direct flow on the same failed 100/200/300-ms schedule."""
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.gapped_direct_flow_tracker_development import GappedDirectFlowTracker
from scripts import check_gapped_camera_recorded_prefix_development as prefix
from scripts import diagnose_gapped_camera_short_gap_failure_development as diagnostic

OUTPUT = prefix.BASE/'go2_gapped_direct_flow_journey_v1_attempt_001'
_write = bind(prefix.write, OUTPUT=OUTPUT)
_diagnostic_write = bind(diagnostic.write, _write=_write)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(successor_source_sha256=prefix.source.digest(Path(__file__)),
            tracker_source_sha256=prefix.source.digest(Path('lewm/gapped_direct_flow_tracker_development.py')),
            predecessor_failure='go2_gapped_camera_short_gap_journey_v1_attempt_001',
            change='direct optical-flow fallback uses actual bounded camera interval',
            camera_schedule_unchanged=True, image_fit_limits_unchanged=True,
            selected_after_predecessor_failure=True, new_native_navigation=False)
        _write(name, value)
    elif name == 'failure.json':
        _diagnostic_write(name, value)
    else:
        _write(name, value)


main = bind(prefix.main, COUNT=3440, GAPS=(1, 2, 3), OUTPUT=OUTPUT,
    write=write, GappedCameraPlaneTracker=GappedDirectFlowTracker)


if __name__ == '__main__':
    main()
