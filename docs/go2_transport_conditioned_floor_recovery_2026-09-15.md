# Floor candidates conditioned on the normal their consumer uses

The gyro-only candidate change recovered obstacle observations without
recovering registered poses. Registration checks its current candidates
against the transported initial floor-reference normal, which differs from
its gravity-derived pool-selection direction. Its candidate selection must
respect that same normal for a consistent partial-height measurement.

`lewm/transport_conditioned_partial_floor_development.py` computes the existing
transported normal from the retained registered/raw anchor rotations, current
raw visual rotation and accepted reference-plane normal. Pool selection and
plane metadata keep their original gravity direction. Only when the original
plane lacks two-axis extent does the selector prune original candidates against
the transported normal, using camera-weighted mean height. Fully accepted
planes remain unchanged. The 3-mm residual, 100-point, quarter-pool and
eight-step limits remain. Existing raw/anchor, transport, height-correction
and residual-moment validators still validate every accepted pose. No point
is added, no missing ray is filled, and no new full normal is claimed.

Seven focused tests passed (three transported-normal tests and four original
gyro-candidate tests). The new registration uses the original reacquisition
runtime. The obstacle consumer uses the previously tested gyro-conditioned
candidate rule and the existing auxiliary-only turn evidence.

## Sequential replay: usable observations restored

The first 820 frames of the original supervised-seed-1402 layout-1 failure
were replayed through one unchanged BatchedConsensusMotion stream, feeding
old/new registration and obstacle consumers. The run completed in 109.495 s.
Old raw/registered estimates match the original recording. Every new accepted
pose passes the existing evidence reader.

- Accepted registered poses: **581 original, 820 revised**.
- All 581 shared registered poses are exact matches; none were lost.
- Additional obstacle observations: **224**, with none lost.
- Additional frames with both obstacles and the required four-pose streak:
  **243**. Some gains come from restoring the pose streak for observations
  whose obstacle evidence was already available.
- Independent physical evaluation, loaded only after estimation completed:
  maximum position error remains **3.690 mm**; median error is 1.307 mm
  over the old accepted subset and 1.716 mm over all 820 revised poses.

These are fixed-recording results, not executed recovery or goal-reaching.
The original failed root retains both gyro-only and transported-normal results.
New output: `transport_conditioned_partial_floor_consumers_replay_820_v1/`,
including `effective_availability_diagnostic_v1.json`.

## Fixed native experiment

Run two separate 4,800-tick layout-1 follow-ups sequentially on CPUs
8–15,24–31, with no second simulation or estimator replay during execution:

1. Supervised rollout seed 2026091402.
2. Reactive, after evaluation of the first, regardless of its outcome.

The reference for each is its completed auxiliary-only turn recovery attempt.
Keep BatchedConsensusMotion, the original model/correction, sensor geometry,
2-mm depth noise, timing, stopping/disk guards and four-pose reacquisition
rule. The intended change is candidate selection in the two floor consumers.
No cache, new model, threshold relaxation or different maze is included.
Both outcomes remain separate from all earlier comparisons; preserve every
failure and report actual degraded-turn exposure alongside physical outcomes.

Launcher: `scripts/run_go2_transport_conditioned_floor_recovery_development.py`
with `--layout-index 1 --arm <arm>`.
Evaluator: `scripts/evaluate_go2_transport_conditioned_floor_recovery_development.py`
with `--arm <arm>` after the owner finishes archival.
Roots: `go2_transport_conditioned_floor_recovery_{arm}_noise_2mm_native_layout01_4800_v1_attempt_001`.

The supervised follow-up failed with tracking queue overflow after 463
captured frames and 429 published poses. Physical evaluation found no arrivals,
zero contacts and maximum pose error 3.003 mm. Actual model/correction binding
and learned XY/yaw use were verified on 107 plans. There were no auxiliary-only
requests, so this run supplied no physical recovery exposure. Owner exit 1
after 1:26.52, maximum RSS 4,928,388 KiB, no swaps. Its full recording and
failure remain preserved. This is the same class of tracking failure as its
earlier auxiliary-turn-only follow-up, before the close-wall section.

The unchanged fixed reactive follow-up completed its full budget without a
goal: 4,805 accepted poses, zero contacts, maximum pose error 9.860 mm,
1,189/1,200 on-time plans, 480.86 simulated seconds. Owner exit 0 after
11:42.98, maximum RSS 25,343,632 KiB, no swaps. There were no floor-reacquisition
hold requests. Fourteen obstacle frames used the new gyro-conditioned pruning.
It executed 180 auxiliary-only turn requests beginning at 418.38 s, but
requested no translation after that first degraded turn. Later paired-camera
obstacle evidence reappeared at frame 4170 (418.50 s). None of these observations
establish sustained physical recovery or goal-reaching.

The command/plan diagnosis found **88 right-arc veto events, each followed by
a left-turn recovery plan**. The existing recovery target is always 45 degrees
left; the instantaneous controller then steers back toward its rightward route
and tries another arc. There were 450 view-recovery plans, 990 pure-turn plans
and only 500 nonzero translating request intervals. This is a repeated
veto/view/retry pattern, not evidence of a high-frequency left/right steering
instability: only 92 of 893 adjacent pure-turn plan pairs changed sign.

The final failure also contains stored-clearance blocking: 91 plans, first at
frame 4400. At that frame, stored clearance was 446.3 mm versus physical
body-centre wall clearance 475.3 mm; at frame 4800, 424.5 versus 453.5 mm.
The source of this approximately 29-mm discrepancy has not been isolated.
`reactive_veto_view_cycle_diagnostic_v1.json` retains the event list, actual
candidate-pruning frames and example plan cycle. No claim that all missing
poses in the predecessor would have been recovered on this different native
trajectory follows from the absence of holds in this run.

Both fixed attempts are complete, evaluated and fully retained; neither reached
the goal. Further supervised testing requires tracking headroom to reach the
close-wall section. For reactive control, a concrete next hypothesis is to
choose the recovery-view turn direction from the vetoed arc's yaw direction,
while retaining the existing view size, measured completion and all command
guards. No such direction change has been implemented or tested yet. No
deployment, hard real-time, hardware or independent-new-maze claim follows.
