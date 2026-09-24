# Common-floor registered native readout V1

Run `scripts/read_go2_joint_floor_registered_maze_pilot_v1.py` only on the exact
completed native result SHA-256. Output is exclusive
`go2_joint_floor_registered_maze_readout_v1_attempt_001`. Verify all native
artifact/source bindings before and after, and require 8 GiB available RAM plus
128 MiB artifact headroom over the 40 GiB reserve. No model or native execution.

Report actual goal/return verification, physical traversals, strict visibility,
recovery/waypoint intervals, timing, and auxiliary-floor contact intervals under
their original limited scope. The pose-accuracy helper explicitly dispatches
only the reviewed independent-plane or common-plane evidence schemas to their
respective strict validators. No pose is relabeled to bypass admission.

Compare original visual and common-floor registered poses with the same actual
native observation endpoint at sample 749 + 50 * frame. Report both XY/XYZ and
rotation errors. These evaluator-only outcomes are never used for policy,
estimator selection or calibration. Include degradation as well as improvement.
No unexecuted alternative trajectory is inferred. Missing terminal poses are
explicit; missing physical endpoints, nonterminal missing poses, altered
witnesses or chronology fail the readout.

The shared helper's tests exercise both schemas, including synthetic native
endpoints that favor the correction and endpoints that favor the raw pose.
They do not establish physical accuracy, floor identity, calibrated uncertainty,
navigation reliability, realistic timing or deployment qualification.
