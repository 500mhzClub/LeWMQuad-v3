# Long camera-gap schedule: retained moving-recording failure

The fixed repeating 100/200/300/400/500 ms camera schedule failed at source
frame 285 of the completed layout-0 JEPA recording. It had consumed 286 gyro
packets and accepted 95 visual observations, most recently source frame 280.
The last accepted pose was never reset or replaced. Maximum position error
over accepted observations was 2.496 mm against simulator truth.

The failed interval was 500 ms long. In the original recording, each command
from frames 280 through 285 requested a right turn at 0.45 rad/s. Independent
native evaluation measured 0.223726 rad rotation between frames 280 and 285,
and 0.222471 rad from the last accepted visual attitude to the later native
attitude. Both exceed the unchanged 0.20 rad incremental image-fit limit.
Native translation over the interval was 10.916 mm.

The saved exception is the outer `dual-camera RGBD pose unavailable; terminal
failure` error; it does not identify which inner check rejected first. The
native motion nevertheless establishes a concrete mismatch between this camera
schedule and the preserved angular envelope: an accurate estimate of the later
attitude would lie outside that envelope. Native poses were loaded only after
tracking stopped and were not estimator inputs.

A separate development successor now uses the fixed repeating 100/200/300 ms
schedule. The tracker, geometry limits and gyro processing are unchanged.
At a nominal 0.45 rad/s, 300 ms corresponds to 0.135 rad, leaving room below
the 0.20 rad gate; this arithmetic is not a calibrated motion bound. The new
schedule was chosen after this failure and is explicitly a development change.
Its result is pending at preparation.

The failed run completed in 19.136 seconds, session 9526, exit code 1. PID was
3193964 with creation time 1789284898.3. Its `failure.json`, partial accuracy
readout and all accepted visual results remain in
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_gapped_camera_plane_tracker_jepa_journey_v1_attempt_001`.
Sources are `scripts/check_gapped_camera_jepa_journey_development.py` and the
unchanged `scripts/check_gapped_camera_recorded_prefix_development.py`.
The successor has a distinct output directory and source:
`scripts/check_gapped_camera_short_gap_journey_development.py`.
Neither run executes a new native trajectory or establishes continuous control.
