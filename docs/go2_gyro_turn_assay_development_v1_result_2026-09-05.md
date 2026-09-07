# Sensor-only relative turning: completed development result

The fixed 18-trial [assay](go2_gyro_turn_assay_development_v1_2026-09-05.md)
completed; all 18 raw traces passed replay and artifact audit. Gyro feedback
passed the physical endpoint in **9/9**, versus **6/9** for nominal timed turning.
All trials were contact-free and met drift, release-motion and body-stability
limits. The three timed half-turns failed heading accuracy, not execution
integrity. No trial was replaced, resumed or tuned after observation.

| Target | Timed heading error (rad) | Gyro heading error (rad) | Timed / gyro duration including release (s) |
|---|---|---|---|
| +90 degrees | .105–.113 | .091–.092 | 5.3 / 6.0–6.2 |
| −90 degrees | .033–.042 | .090–.093 | 5.3 / 7.1–7.4 |
| 180 degrees | .215–.224 | .092–.106 | 9.8 / 10.6–11.2 |

Gyro feedback trades speed for endpoint control; it is not uniformly more
accurate (the timed right quarter-turn was more accurate). Feedback's maximum
base drift was .0948 m. Maximum integrated-heading disagreement with the
separate physical orientation reference was .0186 rad, below the fixed .04-rad
check. The physical success tolerance was .12 rad, so the observed .09–.106-rad
feedback error is a limited margin, not precision placement.

The audit reconstructs all 8,030 ideal body-sensor samples from 80,300 physics
samples, verifies 1,354 RGB packets/history windows, replays every decision from
its actual policy-only packet, checks command slew and effective gains, and
recomputes native contacts and physical endpoints. Each timed/gyro pair had an
identical physical and body-history settling prefix.

This qualifies a **development arena primitive**, not a narrow-clearance turn,
place association, visual exit detector or real IMU. Only three correlated
initial-heading variants were used per target; do not interpret 9/9 as a
population reliability estimate. Half-turn success concerns endpoint orientation
modulo 2π: the controller does not guarantee a specified swept direction at the
±π branch cut. Before route integration, clearance and direction constraints
must be explicit. The primitive uses ideal gyro only for control; RGB is captured
for future look-around integration. Simulator emergency stops remain a privileged
experimental safeguard. No JEPA checkpoint or old action bank was changed.

## Consequence and next step

Keep this source/result fixed. Use it as a candidate look/reorientation component,
with separate validation of sensing degradation and maze clearance. Do not feed
its realized feedback-command sequence into a predictor as though the sequence
were known before execution. Next, construct causal subtrajectory training
windows from the existing audited development corpus, with remaining-plan masks,
stop censoring and unchanged layout roles. Later-state windows provide measured
outcomes only for their actually executed action, not matched counterfactuals
for all alternatives. Repeated sensor-only decisions and observed place/frontier
memory remain unimplemented at system level.

## Evidence identity

Root: `.generated/go2_gyro_turn_assay_development_v1_attempt_001`.
Collection session4323 and audit7433 both terminated with exit0.

- Launch SHA-256: `7d6199d4b8f507252b0471d6fa73071ff8ba78c4cddc6ff6b262d141e5ed3c7d`.
- Result SHA-256: `a35dfbbcd50758af0d3466e333a40e5d2f71d4cf491365803fa776e55e7ee52c`.
- Raw audit SHA-256: `30e8c2a3256efe08baa59c179954d8c874389b14a205e794481ab10991f55442`.

Final unseen-maze exploration, hidden-beacon discovery/return and physical
transfer remain unachieved. This turn is substantive empirical progress, not
a blocker recurrence or final-goal completion.
