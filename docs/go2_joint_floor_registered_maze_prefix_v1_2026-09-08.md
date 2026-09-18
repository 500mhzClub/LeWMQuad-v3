# Common-floor registered maze prefix V1

Execute `scripts/replay_go2_joint_floor_registered_maze_prefix_v1.py` with exact
completed confirmed-floor native and readout result SHA-256 arguments. Output is
exclusive `go2_joint_floor_registered_maze_prefix_v1_attempt_001`. No native
execution, model training, checkpoint selection, retries or output overwrites.

The full completed predecessor audit must report sensor reconstruction, model
and command replay, and model-state preservation. Verify all completed artifact
and source bindings before and after; then reuse those audited predecessor
decisions. Only the new candidate controller is freshly executed in this prefix.
This deliberately avoids another duplicate execution of the already completed
predecessor audit; it is not a claim of two new controller replays.

Also verify the exact launch, failure and saved stream of the failed independent-
plane prefix, and preserve its frozen source bindings. No observer, map or
mission state is resumed from that failure. The current camera inputs come only
from the completed confirmed-floor native trajectory.

At every candidate frame, compare the complete original visual evidence with
the recorded audited predecessor and require identical raw forecasts whenever
both controllers plan. The combined-plane pose, map, contacts, residuals and
mission distances are the declared intervention. The first different command
or terminal ends replay immediately, before reading an outcome from a command
the candidate did not issue. Admission failure is a terminal failed prefix,
not a successful intervention. All original visibility/navigation failures stay
unchanged; this replay provides no native navigation success.

Use one CPU worker, one fresh candidate, at least 16 GiB available RAM, and
512 MiB artifact headroom over the existing 40 GiB reserve. Record current CPU,
GPU, RAM, storage and competing jobs. Keep all launched source bindings frozen.
Native execution requires a completed nonterminal first-command intervention,
review of that result and a separate prospective launcher/resource preflight.
