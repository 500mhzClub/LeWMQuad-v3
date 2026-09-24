# Causal gravity feedback: completed development result

All three fixed estimators completed all8 preserved routes and1,498 actual
RGB/body packets. No sensor failures, missing point predictions, physical retries,
parameter fitting or changes to the running JEPA study occurred. The unchanged
gyro-only mode reproduced every previous frame's floor-ray counts and point-error
sum/maximum exactly. Session48362 terminated exit0.

The [fixed protocol](go2_gravity_feedback_ground_development_v1_2026-09-05.md)
and new runtime/runner/test files are now launch-bound.22 estimator/accounting
tests passed before execution. All144 source and143 input bindings were reverified
unchanged after completion (session30747, exit0). Root:
`.generated/go2_gravity_feedback_ground_development_v1_attempt_001`.

- Launch SHA: `f3b26882fbfebdd2730b106b094cf0d40e6df388f19121ea59f63cf9311e46c0`.
- Result SHA: `9c7534cc56f76ea36cd92737ed1dcf93463b9fcf7067e6816748aa6bb1d64bc2`.

## Matched observed improvement, with a remaining error tail

All1,358,287 actual visible-floor rays remain in every method's denominator.
No unavailable ray was deleted to improve the metrics. Normal/height errors
below are frame-weighted; point errors are pixel-weighted:

| Fixed estimator | Mean / maximum normal error | Mean body-height error | Mean / maximum floor-point error | Points unavailable or >0.25 m error |
|---|---:|---:|---:|---:|
| Gyro-only | 0.02405 / 0.08523 rad | 0.005619 m | 0.04941 / 1.18039 m | 26,036 |
| Body-frame mean feedback | 0.01082 / 0.04020 rad | 0.002830 m | 0.02551 / 0.42688 m | 184 |
| Transported-force feedback | 0.01098 / 0.04002 rad | 0.002886 m | 0.02672 / 0.43071 m | 191 |

Both feedback methods improve mean normal and point error in each of the8
routes. Equal-route mean point errors are0.04689,0.02603 and0.02697 m respectively;
equal-route mean normal errors are0.02331,0.01075 and0.01090 rad. The primary
equal-route fractions of points unavailable or above0.25 m error are0.013602,
0.00009650 and0.00010042. These descriptive route-level summaries avoid treating
millions of pixels as independent trials, but the routes themselves share motifs,
width/spawn variants and development history. No significance or final
generalization claim follows.

The common gate accepts a force-history hypothesis in713 frames, including8
initial frames where all modes use the unchanged initialization. Actual feedback
is applied705 times. Recent command changes reject782 frames; high force-history
residual rejects3. No other gate rejects in this population. These counts are
identical between feedback arms, but that does not establish equivalence on other
motions. The0.25-m tail remains in the wide right-left and wide dead-end-return
routes; the maximum error is still approximately43 cm.

The more geometrically consistent force transport does **not** outperform the
simpler body-frame mean overall in these traces. It is slightly worse in the
aggregate normal/point means and has191 rather than184 large-error points.
Report both. A synthetic changing-frame test verifies the transport mathematics;
it does not prove that this correction improves a physical navigation endpoint
when force contamination, gyro integration and temporal filtering interact.

## Interpretation and remaining work

This supports gravity feedback as a candidate way to reduce accumulated ideal
gyro drift before projecting floor evidence. It is not proof of independent
gravity observability: the tests deliberately demonstrate that a constant unknown
acceleration can pass the quiet/near-gravity/steady-command gates. Real IMU bias,
noise, lever arm and timing are absent from these traces. Heading/yaw is not
corrected by the new up-vector filter. The lowest-foot/flat-support assumption
also remains unqualified.

Do not retrofit this estimator into the ongoing18-model comparison or relabel
its old sensor tensors. That would mix a sensor-state intervention into the
fixed action-coverage intervention. Keep all previous positive and negative
model results and the older1.18-m projection-tail result within their scopes.

Next work:

1. Test both unchanged feedback variants and gyro-only on separately specified
   turning/acceleration/deceleration histories, including gate transitions and
   sensor disruptions. Freeze the population before observing its outcomes. Use
   fresh physical excitation before making a deployment claim; do not tune the
   gate or time constant to the worst route recorded here.
2. Carry the improved *candidate* up vector into a new temporal observation
   component, retaining uncertainty and observed-versus-unknown provenance.
   Near-body ground is absent from the current sampled forward views; it cannot
   be declared free without additional evidence or explicit assumptions.
3. Add actual exit observations, provisional place associations and measured
   arrival events to the existing exploration/controller bridge. Neither a
   visible floor patch nor lower projection error is an exit identity, a graph
   edge, a beacon sighting or proof of robot-volume clearance.
4. After the training study and its full audit complete, report all prescribed
   coverage/objective/head contrasts and choose the next separately fixed
   physical-control comparison. Do not choose a favorable interim checkpoint.

Full unseen-maze exploration, beacon discovery, remembered-goal directed return,
independent generalization and bounded real-Go2 evidence remain unfinished.
