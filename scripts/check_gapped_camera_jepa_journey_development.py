"""Extend the fixed camera-gap check through the JEPA return-arrival sensor tick.

The existing 1,2,3,4,5-tick pattern is unchanged. The last selected visual frame
is 3438; gyro input continues through return-arrival acquisition frame 3439.
"""
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from scripts import check_gapped_camera_recorded_prefix_development as prefix

OUTPUT = prefix.BASE/'go2_gapped_camera_plane_tracker_jepa_journey_v1_attempt_001'
_write = bind(prefix.write, OUTPUT=OUTPUT)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(extension_source_sha256=prefix.source.digest(Path(__file__)),
            scope='recorded sensor span through return arrival; final selected image is frame 3438',
            new_native_navigation=False)
    _write(name, value)


main = bind(prefix.main, COUNT=3440, OUTPUT=OUTPUT, write=write)


if __name__ == '__main__':
    main()
