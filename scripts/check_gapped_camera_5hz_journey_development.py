"""Fixed 5-Hz camera tracking on the complete recorded JEPA journey.

Chosen after actual-interval association still failed over a 300-ms pair.
All 100-ms gyro packets remain consumed. This tests a camera operating rate,
not wall-clock asynchronous control or a replay of the native navigation.
"""
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.gapped_chained_flow_tracker_development import GappedChainedFlowTracker
from scripts import check_gapped_camera_recorded_prefix_development as prefix
from scripts import diagnose_gapped_camera_short_gap_failure_development as diagnostic

OUTPUT = prefix.BASE/'go2_gapped_camera_5hz_journey_v1_attempt_001'
_write = bind(prefix.write, OUTPUT=OUTPUT)
_diagnostic_write = bind(diagnostic.write, _write=_write)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(successor_source_sha256=prefix.source.digest(Path(__file__)),
            tracker_source_sha256=prefix.source.digest(Path('lewm/gapped_chained_flow_tracker_development.py')),
            direct_flow_source_sha256=prefix.source.digest(Path('lewm/gapped_direct_flow_tracker_development.py')),
            predecessor_failure='go2_gapped_chained_flow_journey_v1_attempt_001',
            change='fixed 200-ms camera period; actual-interval tracker unchanged',
            nominal_camera_hz=5, gyro_packet_hz=10, image_fit_limits_unchanged=True,
            selected_after_predecessor_failure=True, new_native_navigation=False)
        _write(name, value)
    elif name == 'failure.json': _diagnostic_write(name, value)
    else: _write(name, value)


main = bind(prefix.main, COUNT=3440, GAPS=(2,), OUTPUT=OUTPUT,
    write=write, GappedCameraPlaneTracker=GappedChainedFlowTracker)


if __name__ == '__main__': main()
