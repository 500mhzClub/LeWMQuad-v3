# Signed view recovery: one reactive development test

The completed transported-floor reactive mission exhausted its 4,800-tick
budget without reaching the goal. Its saved diagnostic records 88 vetoed
right arcs, each followed by a leftward recovery plan. This motivates testing
whether retaining the intended arc direction avoids undoing route alignment.

Run one reactive follow-up on the same exposed layout 1, alone on CPUs
8–15,24–31. On a newly created actual translation-veto recovery, choose the
sign of the existing 45-degree view angle from the vetoed command's yaw.
Straight translations retain the leftward fallback. Primary-depth blindness
retains its existing view behavior. Keep the measured 0.1-radian completion
criterion, mission generation and post-trigger pose checks, original raw
BatchedConsensusMotion tracker, transported-floor registration, gyro-conditioned
obstacle observer, sensor noise, timing, budget and command guards unchanged.

The original view runtime now has a recovery-state construction hook; its
default state and behavior remain the same. The signed mixin records the
vetoed command and its observation time while the parent holds its lock, so
direction selection cannot race with recovery-target creation. Fourteen
focused signed-view and existing auxiliary-turn tests passed.

Launcher: `scripts/run_go2_signed_veto_view_recovery_development.py`.
Evaluator: `scripts/evaluate_go2_signed_veto_view_recovery_development.py`.
Root: `go2_signed_veto_view_recovery_reactive_noise_2mm_native_layout01_4800_v1_attempt_001`.
Reference: `go2_transport_conditioned_floor_recovery_reactive_noise_2mm_native_layout01_4800_v1_attempt_001`.

Evaluate actual executed recovery turns, subsequent translation, physical goal
and return arrivals, contacts and pose error after owner completion. Report
failures without replacing the original comparison. A successful repeated
maze would motivate an independent layout test; it would not establish JEPA,
prediction, generalization or hardware benefits.

Storage before launch: 5.7 GiB free; the completed full-budget reactive reference
used approximately 3.2 GB. Preserve this run in full through evaluation and
retain all failures. No concurrent simulation or estimator replay is planned.

## Completed result

The native owner completed archival and exited 0 in 3:40.07 wall-clock time,
with no swapping. Physical evaluation verified outbound arrival at frame 892
and return arrival at frame 1,412, both with the required one-second quiet
dwell. There were 1,414 accepted poses, zero contacts and maximum position
error 5.695 mm. The mission took 141.52 simulated seconds; 340 of 346 plans
were on time. Actual-treatment evaluation confirmed no predictive outcomes
were used.

The intended intervention was exercised. At 49.40 seconds a right arc was
vetoed; the next recovery plan was a right turn toward a -45-degree target.
The recovery interval contained 98 applied right-turn command intervals and
no applied left-turn intervals. Native evaluation measured net yaw -0.8644
radians and horizontal displacement 45.65 mm over the whole recovery interval
(including all commands in that interval). Translation resumed at 51.80
seconds, followed by 3,958 translation intervals and both arrivals, without
another actual translation-veto recovery.

The reference had 88 such recoveries, all beginning with left-turn plans,
only 500 applied translation intervals overall, and no arrival within its
480.86-second simulation budget. The signed run had 4,358 applied translation
intervals overall and one recovery. This is evidence supporting the direction
hypothesis on the exposed maze, not an isolated estimate of success-rate gain:
asynchronous timing and trajectories also varied before the first veto.

Neither auxiliary-only turn requests nor primary-blind translation vetoes
occurred in the successful run. Degraded-perception recovery remains unproven.
All original failures remain unchanged.

The before/after comparison and PNG/SVG trajectory figures are in
`go2_signed_veto_view_recovery_comparison_v1_attempt_001`. Of 150 common recorded
source paths, 149 have identical hashes. The changed common path is the
view-runtime hook and angle lookup described above; the new mixin and launcher
are recorded separately. Event details and native-turn measurements are saved
in the signed run's `signed_veto_view_recovery_diagnostic_v1.json` and
`signed_veto_view_physical_turn_diagnostic_v1.json`.

Next: test on independent development layouts with the learned and baseline
controllers sharing this local-recovery behavior. This successful reactive
follow-up strengthens the baseline; it does not demonstrate learned-model or
JEPA superiority. Keep the full recording as the first signed-view reference.
