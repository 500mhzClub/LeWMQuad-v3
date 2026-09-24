# Optional plane refinement and continuous route entry

The waypoint-alignment experiment removed the hold trap but alternated route
turns with view turns and failed visual tracking at frame 415. Its 420 saved
camera pairs support two separate diagnoses and replay comparisons.

The image fit at frame 415 had passed the original robust registration gates.
An additional exact floor-plane refinement lost one of its 32 accepted image
matches. `optional_plane_refinement_development.py` now retains that qualified
image estimate only for this failure category. It checks the measured-plane
fit against the original motion, gyro and 2 cm/0.10 rad disagreement limits,
and requires the retained image fit's floor-offset residual to stay within
3 mm. It records refinement as unapplied and does not replace rotation
witnesses with a constrained-fit claim. Other failures and actual plane/image
conflicts still reject. The default predecessor tracker retains its behavior.

Public-sensor replay completed all 420 frames with partial-height registration.
The optional path was selected at frames 79, 94, 183, 320, 357 and 415.
Median/max post-estimation position error was 11.52/15.75 mm, versus
11.01/14.71 mm for the predecessor's 415 admitted poses. This slightly worse
error and longer survival do not constitute navigation success.
Evidence: `go2_waypoint_alignment_round_trip_native_layout00_v1_attempt_001/optional_plane_refinement_replay.json`.

Map reconstruction found all 48 missing-route events started in a conservatively
inflated 5 cm cell, while continuous distance to stored obstacle cell squares
was 0.478–0.500 m. The nominal disk radius is 0.45 m. The new route-entry check
uses the existing segment-to-cell-square distance calculation against every
stored occupied cell. It preserves the footprint, occupied and floor cells,
inflated interior route, and explicit unknown connector cells. The current
depth dispatch check still controls execution. This removed all 48 missing
routes on the same recorded observations and estimator poses.
Evidence: `route_switch_diagnostic.json` and
`continuous_route_switch_diagnostic.json` in that same artifact directory.

Three focused tests passed: optional image retention with conflict rejection,
propagation of other refinement errors, and continuous entry with rejection
at actual disk contact and tangency. Initial test collection failed because
the command omitted the repository's Genesis Python path; the corrected
environment passed.

A fresh 1,800-tick navigation experiment was launched in
`go2_continuous_connector_optional_plane_native_layout00_v1_attempt_001`,
session 67231. It keeps the learned model, action bank, alignment score,
partial-height sensing, current obstacle checks and measured-simulation
timing. Collection completed 1,805 frames, 9,025 policy steps and 91,270 physics
samples over 181.04 simulated seconds. There were no contacts or pipeline
faults, but no arrivals either. Final observed goal distance was 2.547 m.
Only 24 of 450 plans were on time; 426 were late, and 8,522 policy requests
had no on-time plan. The process exited successfully and all 1,805 camera pairs
were persisted with static identity unchanged. Result SHA-256:
`8c285b9c2f9974721802522d1178661f336bfb05c2f81032f31740f970ed3158`.
Host real-time operation is not claimed.

The stage timings localize the regression to planning: its median duration rose
from 58 to 198 ms, whereas tracking changed from 78 to 76 ms and registration
from 68 to 74 ms. Of 450 scored plans, 424 reported additional view required.
The new continuous check tests each nearby entry against occupied cells even
when the start disk itself intersects an occupied cell. A start-disk rejection
can eliminate those impossible searches without changing their result. This
optimization is now implemented and its focused rejection test passed in
1.82 s. Reconstruction of all 450 planning observations preserved all 424
missing-route decisions. In these cases the distance to stored occupied cell
squares was 0.4307–0.4416 m (median 0.4325 m), below the 0.45 m footprint;
the first was frame 108. The other five status differences are the expected
outer translation-veto view override. This confirms the short-circuit changes
no route decision here, but does not yet measure its live timing improvement.

Next, compare the coarse stored obstacle cells against their source depth
points and current fine-grid obstacle observations. A faster rejected route
does not resolve a disk that the stored map places inside an obstacle. The
successful replay on the predecessor trajectory did not establish success
on this changed trajectory. No subsequent native run has been launched.
