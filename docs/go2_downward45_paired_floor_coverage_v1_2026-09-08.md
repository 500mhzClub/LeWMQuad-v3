# Actual 45-degree paired floor-coverage diagnosis V1

Authenticate the completed 45-degree native probe and readout. Replay every
recorded JEPA decision exactly with the unchanged corrected model and explicitly
calibrated 45-degree controller. Fix diagnostic ticks 42 (first infeasible wait)
and 52 (terminal); both are actual six-candidate floor-only FL-foot failures.

For every original failing floor-only foot, reconstruct its exact predicted
nominal centre from the observed pose/joints and model first-step prediction.
Require equality with original primary and auxiliary complete-foot witnesses.
Query original full 44-mm squares and their complete closed 2/4/8 subdivisions
against primary, auxiliary and paired measured histories. Every positive tile
must retain the original single-frame complete-pixel floor witness. Joint history
is read-only and retains clocks, transforms and sensor provenance. For failed
hold squares, explain whether uncovered tiles were ever entirely in a measured
frustum or instead had rejected floor pixels. No unseen tile becomes free.

This diagnosis changes no controller, sensor, tolerance or model and infers no
unexecuted native outcome. It tests evidence aggregation on the actual new
trajectory rather than retuning a camera to predecessor locations.

Freeze complete source and input bindings before exclusive root
go2_downward45_paired_floor_coverage_v1_attempt_001. One CPU process/thread,
8 GiB available RAM and 256 MiB output allowance above the 40-GiB reserve.
Reverify every decision and all source/model/input bindings afterward.
No navigation, independent-maze, real-time or hardware qualification.
