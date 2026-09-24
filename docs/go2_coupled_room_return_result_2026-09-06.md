# Coupled feedback result: verified maze-scale local turns, no complete returns

The distinct coupled-feedback experiment and independent raw audit are
complete. **0/3 full room returns**; both nominal runs reach the final home
stage. **12/13 declared local holds pass** native pose, stopping and signed-yaw
checks. Both nominal signed quarter- and half-turns pass. This advances actual
continuous execution, but does not complete the room task or scientific goal.

## Results

| Trial | Completed scripted stages | Native-passing local holds | Terminal cause | Simulated duration |
|---|---:|---:|---|---:|
| Nominal left | 6/7 | 6/6 | Visual registration rejected during final home leg | 170.2 s |
| Nominal right | 6/7 | 6/7 | PLANNER_NO_PROGRESS during clipped home leg | 293.5 s |
| Low friction left | 0/7 | None completed | Visual registration rejected during first leg | 39.1 s |

The right run has an extra clipped corner-return leg, so local holds and
scripted stages have different counts. Its final corner hold has maximum native
planar error **0.0600518798 m**, outside the unchanged0.06 m limit despite visual
acceptance. Keep that failure; do not add a retrospective numerical tolerance.
All13 declared holds pass the separate signed-winding check. No run completes
a home hold, and none triggers native physical-stop supervision.

The raw audit passes all **4,956 exact runtime replays**, all4,986 depth checks
within1 mm, and the raw clocks/sensors, fresh starts, material/gain identity,
command/slew/phase tape, recorded waypoint memory and native guards. Collection
contains4,983 command intervals and251,400 physics samples,502.8 simulated
seconds including setup/drains. No expected artifacts are absent. Audit pass
means faithful evidence, not navigation success.

Maximum available visual position errors are8.55/14.68/2.20 mm for nominal
left/right/low friction. Native path lengths are3.0955/4.2971/0.8172 m. A separate
normalized-Hamilton-quaternion calculation independently reproduces the local
hold and signed-yaw metrics, including the right corner failure; its available
left-audit comparisons agree within1e-10. Mechanical energy remains unavailable.

## What changed and what remains wrong

`CoupledPulseServo` now uses the fixed six-cell empirical pulse/brake table to
jointly plan XY and signed yaw. It executes only the first pulse, waits for
observed quiet braking, measures the response and replans in one uninterrupted
visual frame. The table is fitted solely from the older two nominal pulse
episodes; no online adaptation, current-trial future response, native pose,
friction label or maze topology enters action selection. The nominal-left
half-turn uses forward compensation and passes native scoring: maximum planar
error0.04700 m and signed-yaw error0.00452 rad. The right half-turn likewise
passes with0.01493 m and0.03930 rad, with the requested negative winding.

The new runs use fresh starts and appearances. They are not paired causal
comparisons with the old separable-controller batch, and two nominal room
trials do not establish maze/generalization reliability. The high-level
controller remains hand-engineered model-based control above learned gait
execution, not a learned navigation policy or JEPA-driven controller.

The two visual failures are now localized without changing the estimator.
Left frame1677 against reference1656 retains68/81 inliers and8 reference cells,
but only5 current cells. Low-friction frame366 against317 retains60/75 inliers
and6 reference cells, again only5 current cells. Both fail the fixed six-cell
gate; neither fails the inlier-fraction or reference-translation gate.

A read-only check of the fixed eight most recent accepted references finds one
alternative passing unchanged pair and translation-increment gates in each
case: left reference1654 (67/80 inliers,6/6 cells,7.62 mm native error) and low
reference270 (2.57 mm native error). Native error is attached only AFTER
sensor-only registration/gate decisions. This suggests a bounded reference
buffer; it is not an executed recovery, full-stream replay or counterfactual
successful return. Preserve the original failures. See the
[diagnostic record](go2_coupled_room_return_visual_failure_diagnostic_2026-09-06.json).

The right home failure also exposes a scheduling issue. Its clipped .4 m
subgoal targets(.05550,.00891), not home, but already requires the final home
heading0. This can induce a turn away from the remaining journey too early.
After34 pulses it has one pulse left, position error0.10457 m to that clipped
goal, yaw error0.05445 rad and an empty SEARCH_EXHAUSTED plan. That is a bounded
search/controller failure, not proof of physical infeasibility. Corner-return
subgoals also recompute heading from a tiny residual after clipping, introducing
an extra large turn. Correct these semantics prospectively, without claiming
home from an intermediate target or relaxing final acceptance.

## JEPA interface and verification

A separate untrained `PulseTimedRGBBodyJEPA` now represents partial final action
blocks: the known short-pulse/minimum-brake endpoint is2.2 s, not a fabricated
2.5 s sequence. Direct and recursive prediction heads consume tick-validity and
emit exact target offsets. Thirteen tensor/causality/gradient tests pass. The
model was not trained, saved as a checkpoint, or used in the physical assay;
old full-block training scripts/checkpoints are not interchangeable. See the
[interface and training requirements](go2_pulse_timed_jepa_interface_2026-09-06.md).

Full explicit regression: **2,476 tests passed across195 files in204.87 s**.
This includes20 coupled-controller tests and13 pulse-timed JEPA tests. The
earlier controller-only full suite passed2,463/194 files in203.43 s. Synthetic
model-matched returns test scheduling/control implementation, not physical
generalization. The launched controller/protocol/auditor sources were unchanged.

## Timing, storage and next work

Measured observation/control median times are146.19/143.70/155.21 ms, with95th
percentiles162.71/155.47/162.83 ms. Every decision except each initial one exceeds
100 ms. This segment includes the simulation observation/acquisition pipeline;
it is not an isolated network benchmark. Physics remains paused during compute:
no10 Hz real-time or deployment claim is supported. Hidden-robot ideal RGB-D/
gyro, uncalibrated uncertainty and controlled-floor assumptions remain. Native
collision supervision is not online body-sweep clearance.

Minimum recorded free space is20.34 GiB, above the10 GiB reserve; no resource
stop occurs and no evidence was deleted. Another full batch needs a fresh
storage budget; a different filesystem has space but existing artifact-root
constraints must be explicitly supported, not bypassed.

Follow the [continuous-return completion plan](go2_continuous_return_completion_next_steps_2026-09-06.md):
bounded multi-reference tracking under unchanged gates, correct intermediate
return-heading intent, then fresh closed-loop returns. Continue to actual
branch/marker-driven maze exploration and physical memory evaluation, matched
JEPA predictive-training/online-rollout/memory studies on independent layouts
and seeds, realistic calibrated sensing/timing and bounded hardware. No local
turn, offline diagnostic, source model or green test suite completes that goal.

## Evidence identity

Output: `.generated/go2_coupled_room_return_v1_attempt_001`.
The launch binds653 sources and44,596 inputs.

- Launch: `be4267ab90f122e18ca8ef8f260cacc2150cdb6aed57f568b6170863c41e0fbf`
- Collection result: `b5ac72bf6ad35df56d99f0dc9be00ae19208d091e805aeaea4501e8d0152cf98`
- Raw audit launch: `e031c454cf49154b6be630f7223a92730b4593df64a3908645c116962bbf2a46`

Completed aggregate and per-trial audit hashes are recorded in the autonomous
checkpoint and the audit's own evaluation bindings.
