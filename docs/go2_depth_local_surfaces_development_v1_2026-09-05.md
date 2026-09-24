# Observed local surfaces: implemented measurement front end, not a navigator

The [single-sample RGBD interface](go2_single_sample_rgbd_observation_development_v1_result_2026-09-05.md)
passed actual rendered visual-surface checks. The next runtime module is
`lewm/depth_local_surfaces_development.py`, with11 focused tests and a read-only
actual-packet checker at `scripts/check_go2_depth_local_surfaces_development_v1.py`.
These new development files are not yet bound into a physical successor and
remain editable. No predecessor source or experimental outcome was changed.

## Runtime measurements and limits

The observer accepts only the existing validated RGB/body packet and separate
calibrated depth packet. It projects sensor-local depth into the fixed body
frame; no maze boxes, labels, simulator pose or evaluator floor identity enter
this path. Four causal observations are retained, with duplicate/gap faults
latched by the existing depth-history contract.

The current model samples every second pixel and uses a body-height slice
within +/-0.06 m of the body origin. Three returns are required per column;
columns mixing depths over0.04 m are unknown. Ordered wall cross-sections are
split at depth discontinuities and corners, with minimum eight columns,
0.05-m support length and0.015-m maximum line-fit residual. These are explicit
development modelling choices, not calibrated sensor uncertainty. All valid
sampled points are retained, including thin obstacles too small for line fits.
Missing rays are never bridged or labelled free.

Widely spaced samples on grazing walls are tested against available neighbouring
tangents before being labelled a depth discontinuity. The first version's
constant-spacing rule failed the analytic corridor test (13 false boundaries);
a two-sided-only continuity check left two false boundaries at the range limit.
The final check uses the available one-/two-sided evidence without extrapolating
into missing rays. Actual depth steps between parallel surfaces remain separated.
These failures preceded final verification and were not physical experiment retries.

A discontinuity is reported as an occlusion or possible surface end, not a
traversable portal with a guessed width. Fitted endpoints are the extent of
observed support, not automatically physical wall endpoints. Surface normals'
two-dimensional span identifies conditional weak translation directions: a
featureless corridor with parallel side walls cannot constrain translation along
the corridor from those planes alone. This assumes fixed attitude and correct
static correspondences; it is not an odometry estimate, information matrix,
covariance, calibrated probability or proof of place identity.

The horizontal slice does not cover the articulated robot's height, feet,
underside or unseen side/rear space. Every output explicitly leaves arrival,
body clearance, turning clearance and free-volume claims false. A controller
must not promote this front end alone into permission to move.

## Evidence obtained

Focused final76933 passes11 tests in1.06 s. Tests cover analytic front/side walls,
corners, corridor translation degeneracy, depth steps, unknown rays, mixed
surfaces, holes, small-obstacle retention, copied histories, causal faults and
rejection of privileged fields. The first test invocation exited before test
collection because ambient ROS plugin loading required unavailable `lark`;
the explicit plugin-disabled project test environment resolved that launch issue.

Final full89033 passes1,079 tests across94 explicitly selected files in75.03 s,
exit0, with no concurrent source edits. The earlier full80433 passed1,078 tests
before explicit thin-obstacle-return retention and its extra regression test.
Final observer SHA-256:
`b83fc6fee89fcfdf32a6021d0649b949010049ef9bb65ba04efb5aadb6abb98d`.
Final read-only checker SHA-256:
`a4f42e26bafff842cbb848bfdd1d4405edd7bbd81be8d264b8932e107582bd55`.
These record the checked development revision, not a frozen physical-run binding.

Post-document guard30231 passes all260 source,192 input, two gait and ten native
bindings plus three terminal-result and three final development-source identities,
exit0. No collection, audit, replay or test process remains running.

Read-only replay26705 completed all ten existing audited sensor packets, checking
260 predecessor source,192 input, two gait and ten native source bindings plus
the exact launch/result/audit and individual artifact identities. It does not
render new frames, rerun physics, change prior metrics or change commands.
Visible-marker views yield two front-wall segments,232 valid slice columns and
88 unknown columns where mixed vertical returns are rejected; four valid points
are retained without a line model. Occluded views yield one front occluder segment
and320 valid columns. All ten frames have one conditional normal direction,
not a full two-dimensional position estimate. Small fit residuals establish only
internal line consistency, not independently measured global localization error.

## Next integration, in order

1. Add this observer and the verified depth acquisition to a separately named
   continuous whole-task development runner. Retain RGB-only results. Record
   current-depth geometry at actual decision times and preserve complete raw
   physics, sensor, command, contact and memory evidence. Use existing failed
   whole-task layouts as development regressions, not new independent tests.
2. Estimate relative motion from successive observations and causal gyro/body
   sensing. Carry weak directions and correspondence ambiguity; do not fill
   them with command integration and call it measured displacement. Validate
   against evaluator-only motion on the moving mission, including braking.
3. Track observed opening boundaries across approach and loss of visibility.
   Replace the global image-change arrival predicate with the measured boundary
   relative to the full articulated body plus uncertainty and post-stop motion.
   Depth discontinuities alone are not sufficient boundary identities.
4. Use the full depth rays and accumulated local observations to check the
   motion envelope, not only this body-height slice. Keep unknown side/rear
   volumes unknown. Reposition or refuse a turn when the required sweep cannot
   be observed with bounded state uncertainty; retain the narrow unsafe-turn
   fixtures rather than changing their geometry to obtain success.
5. Exercise beacon discovery and actual return before attributing benefits to
   memory or JEPA. Then run matched supervised/JEPA and genuine multi-step
   planning comparisons on independent layouts and calibrated sensor variants.

No new stationary population is needed. No navigation improvement, memory
benefit, JEPA benefit, hardware deployment or final-goal completion is claimed.
