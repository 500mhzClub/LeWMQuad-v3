# Camera-gap tracker: first moving-recording result

`GappedCameraPlaneTracker` accepts camera intervals from 100 to 500 ms while
`CameraIndependentGyro` continues to consume every 100 ms gyro packet. Processed
visual frame numbers and actual acquisition timestamps are distinct. The
original image/rigid-fit, measured-plane and anchor/increment disagreement
limits remain in force. Missing visual frames are not fabricated or retimed.

The original ten-observation bridge limit is supplemented by its original
one-second elapsed-time limit. Thus larger camera intervals cannot extend
unanchored tracking to five seconds. Direct and chained image-flow fallbacks
still require their original 100 ms image links; the new tracker does not claim
those methods support absent intermediate images. Sensor or tracking failures
remain terminal for the instance.

Seven focused tests passed in 5.11 seconds. They exercised complete original
raw-pose equality on consecutive frames, real timestamp gaps, auxiliary-camera
fallback, missing gyro, duplicate/overlong camera intervals, a delayed initial
reference, and the one-second measured-bridge ceiling.

The recorded moving test used source frames 0–200 of the completed independent
layout-0 JEPA journey. It consumed all 201 gyro packets and selected camera
frames with a fixed repeating gap pattern of 1, 2, 3, 4 and 5 acquisition ticks.
All 68 selected visual observations were accepted, ending at source frame 198.
Each output preserved its real measurement timestamp and had integrated every
gyro interval since the original reference. No high-level model was loaded.

Only after tracking finished, the evaluator loaded simulator poses and compared
them in the original body frame. At the same selected observation times:

| Position error against simulator truth | Gapped tracker | Original dense tracker |
| --- | ---: | ---: |
| Median | 0.960 mm | 0.833 mm |
| Maximum | 1.880 mm | 1.711 mm |

Gapped-tracker rotation error was 0.181 mrad median and 0.547 mrad maximum.
These are observed errors over this short recording under ideal development
sensing, not calibrated error bounds. The original dense tracker's recorded
poses supply a comparison, not an input to the new tracker. Neither native pose
nor dense pose was passed to the estimator.

Median tracking computation was 134.952 ms, maximum 298.006 ms. The complete
recorded check took 15.185 seconds, session 8837, exit code 0. No computation
delay was inserted into the sensor timeline. This is camera subsampling on an
existing recording, not continuous native execution or a new navigation result.
Longer journeys, delayed processing and realistic sensing remain untested for
this tracker.

The candidate exposes raw tracker results and explicitly declares incompatibility
with the existing fixed-cadence controller admission. Integration still needs
timestamp-aware pose availability, mapping/mission updates and command-age
handling, followed by a physical simulation with computation in flight.
No live or queued controller was changed.

Implementation: `lewm/gapped_camera_plane_tracker_development.py`.
Recorded check: `scripts/check_gapped_camera_recorded_prefix_development.py`.
Artifacts, including full selected visual results and per-observation native
error comparisons, are in
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_gapped_camera_plane_tracker_prefix_v1_attempt_001`.
