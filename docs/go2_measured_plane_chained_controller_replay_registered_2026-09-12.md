# Chained measured-plane controller replay registered

Recorded 2026-09-12 Europe/London. This goal turn made progress by implementing,
testing and registering the paired controller replay. The full navigation goal
remains incomplete; no new physical command has been executed.

## Implemented and verified

`scripts/replay_go2_measured_plane_chained_controller_prefix_v1.py` compares the
actual measured-plane controller with `MeasuredPlaneChainedAnchorController`
from fresh state on the same completed native public packets. It reproduces
every complete original decision, checks actual model calls and common raw
forecasts, saves both complete decisions, and stops at the first changed command
or terminal. It then independently reconstructs all consumed output rows and
public packets, retaining the boundary and original/candidate negative results.
It does not consume an observation after a changed command.

The original native parent must have completed its raw audit and ended. The
exact earlier full-history timing waiter and child must also have completed and
ended, so this work cannot take their reserved CPU replay slot. All previous
failures remain preserved. No training, source replacement or native scene
launch is part of this replay.

Validation:

- 39 replay/comparison/input-ordering tests passed in 2.34 seconds, session 8835.
  These cover complete original reconstruction, changed commands, candidate
  failures including failures after inference, matched terminal behavior,
  no following packet consumption, input/model mutation, actual model calls,
  corrupted closed outputs, and incomplete or changed predecessor receipts.
- 12 waiter tests passed in 2.26 seconds, session 80666. These cover exact-owner
  polling, transient observation errors, failure preservation, both resource
  thresholds, and complete child input/output reconstruction requirements.
- Child source/resource preflight passed with 2,594 source bindings, about
  72.1 GB available RAM and 583.6 GB artifact free space. No controller ran.
- Final waiter preflight passed with 2,597 source bindings and the exact
  preceding timing owner confirmed live.

The initial extra-row corruption fixture was rejected by the decision writer
before reaching the checker. The fixture now supplies a consecutive extra tick
and confirms the checker rejects any row after the comparison boundary. No
production check was relaxed.

## Actual registered owner

Waiter root under the owned development artifact volume:
`go2_measured_plane_chained_controller_wait_v1_attempt_001`.

- PID: `2930187`.
- Process creation time: `1789170044.92`.
- Boot: `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.
- Tool session: `10547`.
- Command:
  `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/await_go2_measured_plane_chained_controller_prefix_v1.py`.
- Launch SHA-256:
  `4e0ed3e7a14c61b8a31ded1c83f34e15643e40bcd7668840bf163bc485311416`.
- Source bindings: `2597`.
- First event: `2026-09-11T23:40:49.645092+00:00`,
  `EXACT_RESERVED_TIMING_OWNER_LIVE`.

The actual PID, creation time and command were rechecked against the launch.
The child root
`go2_measured_plane_chained_controller_prefix_v1_attempt_001` was absent at that
inspection. This registration is waiting work, not a completed controller replay.

Its predecessor is timing waiter PID `2924370`, creation `1789167167.42`, launch
`c63f9bccb0cdfc273b1d3fd721310da5fac384f1350dcfc4625098fd103184a4`.
That waiter still waits for the original measured-plane native parent and raw
worker audit. The nominal and reactive native waiters remain registered in their
existing order. No completed parent result or failure file was present in any
of those four roots at the registration inspection.

The waiter, replay/comparison implementation, candidate, tests, protocols and
their complete launch source union are now frozen. Preserve them until all
registered work and verification ends. Use separate source identities for any
later correction; do not patch a running or queued experiment.

## What this can establish next

A completed replay can establish the first actual requested-command difference
caused by the combined perception candidate, with the original sensor-to-command
path reproduced and no changed-command outcome consumed. It can also return a
negative candidate failure or no change before the original terminal.

Any promising boundary still requires a separately defined prospective native
run and full physical audit. Independent-maze replication, JEPA/planning/memory
comparisons, realistic timing and sensing, and bounded hardware evidence remain
outstanding. The three earlier never-dispatched diagnostic scenes are still
unexecuted; this registration does not count them as completed.
