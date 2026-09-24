# Prospective single-case hold-reorientation maze 02 experiment

Execute one development case, `full_jepa_hold_reorientation_maze_02`, with the
existing maze 02 scene, public outbound/return mission, sensors, robot/gait,
physics timing and navigation budget. Load the exact full-JEPA expanded model
through AllPhasePlannerModel, unchanged state
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.
Use HoldReorientationController with fresh observed map, contact memory, mission
and executed-residual state. This changes only the declared response to repeated
holds. It does not combine the separately tested reached-frontier policy.

Required prerequisites, each supplied by its exact completed result hash:

1. The original CPU raw-replay waiter, launch
   `969c9153d9bebbf7cc7ba24e56ed76a912edf33f0a13aeaacc243cdcd2862e2c`.
   Its completed child must reproduce every original decision and the declared
   hold-to-left-turn boundary at 405, with all 403 model forecast banks exact.
2. The already scheduled frontier native waiter, launch
   `68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74`.
   Reconstruct its completed native child and original full six-case adapter
   batch admission. Scientific navigation failures remain admissible completed
   results; incomplete raw audits or changed inputs do not.

The launcher independently reexecutes original full input admission for both
chains before execution and again at completion. The first adapter worker used
by raw replay must be exactly bound into that same completed six-case batch.
Verify all predecessor artifacts and inherited/new source bindings. Reconstruct
the raw prefix comparisons and public packet fingerprints. No original run may
be restarted or superseded, and no additional model, seed or layout is selected.

Admit one native scene only after the original scheduled experiments complete
and the native slot is idle. Require 32 GiB available RAM and the existing one-
case 51 GiB artifact envelope (40 GiB reserve, 10 GiB collection, one GiB
persistence headroom). Use deterministic algorithms, one OpenCV/Torch/BLAS
thread, a fresh subprocess, 2-ms physics and 100-ms command intervals. Retain the
3,000-command shared mission budget, original warmup and ten-command zero drain.
Physics remains paused during computation; no real-time claim is made.

The collector and auditor are the separately bound narrow original derivatives
described in `go2_hold_reorientation_native_preparation_2026-09-10.md`. Audit all
raw sensor reconstruction, full model/controller decisions, actual command
completion, geometry, friction, native contact, strict physical visibility,
observed/native arrivals and physical retracing using the original criteria.
Then require exactly matching first 21,000 physics samples and 406 public
observations, identical completed requests before observation 405, and actual
completion of the new left-turn request through sample 21,049. Every complete
candidate decision through the boundary must match the raw prospective replay.
Later physical outcomes are newly measured; do not substitute the old trajectory.

Preserve all failures and report the same strict round-trip conjunction. No
navigation advantage, independent generalization, real-time or hardware claim
follows from component tests or completing the intervention. Record raw audits,
paired prefix evidence, readout, model identity, logs, resource monitor and all
artifact hashes. No new independent comparison layout is consumed.

Exclusive output `go2_hold_reorientation_maze02_pilot_v1_attempt_001`. No retry or
resume. The source-only preflight creates no output and does not admit completed
inputs. Full preflight also creates no output but requires both completed
prerequisite hashes. Runtime arguments are `--raw-prefix-wait-result-sha256` and
`--frontier-wait-result-sha256`. This preparation does not itself queue a new
native launch or bypass the existing frontier waiter's priority.
