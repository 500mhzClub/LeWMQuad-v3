# Full-history measured-plane timing: registered, not yet executed

The new timing waiter was registered and its exact process was confirmed live
on 2026-09-11 at approximately 22:53 UTC. It waits for the original learned
native parent to finish its complete raw audit before starting any replay.

- Waiter root: `go2_measured_plane_full_history_timing_wait_v1_attempt_001`.
- Owner: PID 2924370, creation time 1789167167.42.
- Command: `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/await_go2_measured_plane_full_history_timing_v1.py`.
- Launch SHA-256:
  `c63f9bccb0cdfc273b1d3fd721310da5fac384f1350dcfc4625098fd103184a4`.
- Frozen sources: 2,586.
- First event: `EXACT_LEARNED_NATIVE_OWNER_LIVE`, 2026-09-11 22:53:19 UTC.
- Child root `go2_measured_plane_single_pass_full_history_v1_attempt_001` was
  absent at inspection. No full-history timing result exists yet.

At the same direct inspection, the exact learned parent PID 2916106 was live.
Its latest complete timing row was tick 3,103. Parent result, parent failure
and worker-terminal files were absent. The native episode remains unfinished;
there is no new verified round-trip outcome.

## What this experiment will establish

Replay every observation of that completed episode with both the original
`MeasuredPlaneResidualController` and `MeasuredPlaneSinglePassController`.
The input population comes from the actual revised-perception execution, not
from continuing an older trajectory after the revised controller diverged.
Retain the original ending and all failures. Compare complete original and
normalized candidate decisions, identical public packets, actual model calls,
unchanged model weights, fixed retained-state checkpoints and final state.

Time complete controller calls with alternating execution order. Report totals,
medians, p95 and counts exceeding 100 ms for every observation, every model
forward and fixed frame windows. The approximately 41% reduction from the prior
123-observation prefix remains the only measured combined-controller speedup;
its effectiveness on this full history is unproven.

The stronger state check preserves all original retained map, residual and
history fields, plus motion, floor registration and mission state. Exact OpenCV
keypoint fields are serialized. The existing ten retained-state type paths and
the single known floor-registration implementation tag are explicitly
normalized. Every registration field remains in the comparison; arbitrary
registration subclasses are rejected. This additional type tag was identified
and tested before registration. Tests also verify state equality through a
latched sensor failure and sensitivity to changed observer, registration and
mission state.

## Verification and scheduling

Fifty focused tests passed: timing/state accounting 25, replay orchestration and
closed-output reconstruction 13, exact-owner/resource waiter and completion 12.
The replay source/resource preflight passed with 2,583 sources, about 70.2 GB
available RAM and 584.4 GB artifact space. The final waiter preflight passed with
2,586 sources and the learned native parent live. Tests include synthetic image
inference and synthetic failure paths; they do not constitute the pending full
native-history timing result.

The waiting process creates no model or scene. Actual replay requires at least
64 GiB available RAM and 43 GiB artifact space; the waiter reobserves temporary
shortages. One full CPU replay may overlap one separately owned native scene.
Reserve that CPU slot through the waiter's final completion verification: do
not start another full replay while this registered job may dispatch or run.
The existing nominal and reactive scene waiters remain unchanged and will
execute their native comparisons sequentially.

Do not modify any bound source, relaunch a live owner, infer completion from a
timeout, or consume a growing native decision stream. The new waiter verifies
the ended child, every saved comparison row and original public packet, state
receipts, input and source bindings, and complete timing report. It preserves a
slow or ineffective optimization as well as an improvement. Execution failures
are retained without automatic retry.

The full navigation goal remains active. Neither this preparation nor the
pending replay establishes a new physical outcome, independent-maze reliability,
JEPA advantage, isolated planning/memory effects, realtime control or hardware
readiness.
