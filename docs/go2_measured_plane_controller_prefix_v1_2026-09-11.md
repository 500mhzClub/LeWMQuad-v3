# Measured-plane controller prefix V1

This prospective development replay composes the new measured-plane estimator
with the actual trained no-RGB direct world-model controller. It follows the
completed observer diagnostic and is not a new physical navigation episode.

Before launch, require the complete positive 3,838-frame observer verification
receipt and its exact completion waiter result. The observer and waiter owners
must have ended. Authenticate the completed extended-budget native worker via
`docs/go2_extended_budget_completed_worker_admission_2026-09-11.json`
(SHA-256 `2e748a012979225e1871d79a7240fe9bf1a14da54a2aeca3ea7d6db9b1fbeedf`).
This admits that worker's fixed raw artifacts, not completion of its parent or
any advancement of the existing native queue.

Load two fresh copies of the originally assigned, corrected no-RGB direct model
(`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`).
Run the original ResidualAnchoredContinuationController and the new
MeasuredPlaneResidualController from frame zero, each with fresh geometry,
memory and mission state. Use the original 4,000-tick budget, maze-02 public
mission, model head, sensor packets, floor gates and planner. The only intended
controller change is the measured-plane visual estimator. The assigned model
loader rechecks the original snapshot and correction; full training ancestry
is not executed again in this diagnostic.

Reproduce each complete original decision against its recorded decision and
both visual histories against the authenticated observer evidence. Require the
candidate floor evidence to match that same observer diagnostic. When both
controllers select a forecast, compare all raw predictions and correction
metadata exactly. Mission settling may issue an explicit current zero-command
hold without a new forecast. Retain all selected actions and score changes.

Stop at the first changed requested command or terminal result, any candidate
terminal, or frame 3,837. Never read the next observation after a changed
command. That next observation resulted from the old command and cannot prove
the outcome of the new one. Preserve comparison failures and partial output;
do not restart or overwrite the attempt. A changed command is a comparison
boundary, not a navigation recovery claim.

Execution uses one deterministic CPU replay process with one OpenCV, BLAS and
Torch thread. The two controller histories depend on the same ordered packets;
use the established CPU numerical setup and paired sequential execution.
The existing native job and its queue remain untouched. Require at least
64 GiB available RAM and 43 GiB artifact-volume space, including a 2 GiB output
allowance and 41 GiB reserve. These are capacity checks, not OS limits. Record
CPU, RAM, GPU and volume state before launch; monitor the actual process.

Freeze recursive source/test/protocol bindings at launch, including both input
admission receipts. Rehash bound source and input artifacts before and after
execution. Require both model states and absent gradients to remain unchanged.
The exclusive output is
`go2_measured_plane_controller_prefix_v1_attempt_001` under the existing
navigation development artifact root. No sealed data, training, new native
execution, unseen-maze generalization, timing qualification or hardware
qualification is part of this replay.
