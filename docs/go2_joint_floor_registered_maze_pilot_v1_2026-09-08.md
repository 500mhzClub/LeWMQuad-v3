# Prospective common-floor registered maze pilot V1

One fresh native CPU scene on reused development maze 0 tests the common-plane
pose estimator with the learned world-model planner. Controller:
`JointFloorRegisteredRoundTripController`. It retains all current measured
camera candidates, checks combined plane observability and every point's
3 mm residual bound, and applies the registered pose consistently to both camera
maps, contact checks, residual scoring and mission tracking. The shared static
floor remains a hypothesis; no calibrated uncertainty or support certificate
is claimed.

Exclusive output: `go2_joint_floor_registered_maze_pilot_v1_attempt_001`.
`scripts/run_go2_joint_floor_registered_maze_pilot_v1.py` requires the exact
completed common-floor prefix result SHA-256. That prefix must preserve the
original visual witness and raw model predictions, reuse a verified completed
predecessor audit, and stop at a changed nonterminal command before its outcome.
The failed independent-plane prefix grants no native execution and remains
preserved separately. This is a new estimator experiment, not its retry.

Before creating output, verify all prefix, predecessor native/readout, model
input and source bindings. Run `--preflight-only` and inspect current CPU
topology/affinity, utilization, RAM, GPU/VRAM, competing jobs and storage.
Require 32 GiB available RAM, plus unchanged 10 GiB collection and 1 GiB
persistence allowances above the 40 GiB artifact reserve. Use one scene worker,
one BLAS/OpenCV/PyTorch thread, and the predecessor's numerical/device setup.
The command sequence depends on the completed prefix; no independent scene is
scheduled concurrently. Resource admission is not an enforced OS quota.

The episode and audit are narrow derivatives of the confirmed-floor collector
and audit: only controller identity and result-status names change. The action
set, learned checkpoint, 3000-tick shared outbound/return mission, ten-command
drain, physical/acquisition stops, collision geometry and strict visibility
gates are retained. Every new observation, complete controller decision and
command is replayed with a fresh controller/model after collection.

Compare actual physics and public packets with the completed predecessor up to
the intervention observation. Every new decision in that prefix must exactly
match the bound prospective candidate stream; retain the original visual
witness separately. Never attach the old trajectory's post-intervention outcome
to a new command. The strict primary/auxiliary visibility audit applies to the
entire new trajectory, even if it changes before predecessor failure frame 909.

Predecessor native result:
`5ee4ef051e1a506f205aae51610deece18f755fb5440c40b1113e22e5ba317ee`;
it failed navigation and strict visibility. This pilot requires its own verified
arrival/return evidence. It cannot alone establish independent-layout success,
matched-model/memory advantage, real-time execution or hardware qualification.
No source edits after launch, output overwrite, checkpoint selection or training.
