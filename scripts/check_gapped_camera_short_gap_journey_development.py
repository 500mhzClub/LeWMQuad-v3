"""Fixed 100/200/300 ms camera schedule after the retained long-gap failure.

At the recorded 0.45 rad/s turn command, 500 ms spans more than the unchanged
0.20 rad increment limit. Shorten the schedule without widening image-fit gates.
This is a declared development successor, not a retry of the failed schedule.
"""
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from scripts import check_gapped_camera_recorded_prefix_development as prefix

OUTPUT = prefix.BASE/'go2_gapped_camera_short_gap_journey_v1_attempt_001'
_write = bind(prefix.write, OUTPUT=OUTPUT)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(extension_source_sha256=prefix.source.digest(Path(__file__)),
            predecessor_failure='go2_gapped_camera_plane_tracker_jepa_journey_v1_attempt_001',
            change='camera gaps repeat 100, 200, 300 ms; tracker and fit limits unchanged',
            scope='recorded observations through return-arrival frame 3439',
            selected_after_predecessor_failure=True, new_native_navigation=False)
    _write(name, value)


main = bind(prefix.main, COUNT=3440, GAPS=(1, 2, 3), OUTPUT=OUTPUT, write=write)


if __name__ == '__main__':
    main()
