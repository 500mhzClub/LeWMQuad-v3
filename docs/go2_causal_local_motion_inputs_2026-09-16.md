# Causal local motion as an explicit model input

The residual models still lose to the fitted pose-based predictor on recorded
maze execution. The neural model has images, body sensors and command history,
but no explicit measured translation history. Before another neural fit, this
experiment derives that information from retained public RGB-D/gyro packets
and compares simple controls with identical training data and draw weights.

The first global training-observer probe failed at frame 40: its newest
reference was frame 37, which failed consensus/grid support, and older retained
references also failed. The intervening frame 39 was not a retained reference.
The failure is preserved. It does not establish that adjacent-image motion
was unavailable.

`CausalLocalRGBDMotion` estimates each adjacent 100-ms interval using the
existing RGB-D correspondence/rigid-consensus gates and gyro integration.
Three consecutive accepted intervals produce the past 300/200/100-ms XY and
sine/cosine yaw differences in the current body frame: 12 numbers. No global
pose is exposed. A failed pair makes every affected four-frame history missing;
three subsequent accepted pairs can yield a new local history without asserting
global relocalization. Missing features are null, not measured zero motion.

On the same clean `switch_episode_000` recording, all 63 adjacent pairs passed,
giving 61/61 histories including frame 40. Two focused composition/missingness
tests passed. This is feasibility on one exposed training recording, not robust
sensing or recovery of the original global observer.

The full derivation uses the fixed 4,514 training and 924 development transfer
contexts. Primary depth receives synthetic independent Gaussian 2-mm noise,
with the same clipping and seed recipe as the live navigation studies. Noise
keys are geometry/appearance/frame/camera, so matched scene histories share
noise and train/transfer geometry keys differ. Gyro remains ideal. No new
simulation or depth files are generated. Native poses and future packets are
not feature inputs. The source registration gates are unchanged.

Every context remains in the following comparison. Two ridge controls predict
residual XY/yaw around command integration, using the same original 7,200 draw
weights and fixed ridge penalty one. Both include the complete past-command
histories from the four causal observations, including validity and age,
alongside prospective commands; this prevents crediting visual motion simply
for revealing preceding commands. One has these command features alone; the other
also has the 12 local-motion values and an explicit availability flag. Missing
values are encoded as zeros only alongside flag zero; the observation source
still records null. Fits are fixed before transfer outcomes are materialized.
Report all contexts, available/missing subsets, braking, and both geometry
clusters at 300/700/800 ms. No hyperparameter search or feature selection from
native accuracy is part of this experiment.

Separately, after features are frozen, compare their past displacements with
native truth. This characterizes available estimates; it cannot calibrate
uncertainty or turn missing histories into successful estimates. Native truth
is never copied into a model input.

Runtime integration must use this same component and causal window definition,
and measure its added latency. The component does not replace the existing
navigation tracker or map. Better prediction alone does not establish online
benefit, visual-world dependence, JEPA advantage or reliable new-maze navigation.

Entry points:

- `scripts/derive_go2_causal_local_motion_inputs_development.py`
- `scripts/fit_go2_local_motion_controls_development.py`
- `scripts/evaluate_go2_local_motion_input_accuracy_development.py`

Artifact roots: `go2_causal_local_motion_inputs_v1_attempt_001` and
`go2_local_motion_matched_controls_v1_attempt_001`.

Status: feature derivation, both matched control fits, transfer evaluation and
posthoc feature accuracy evaluation are complete. The derivation covered 240
recordings and 7,790 frames. Valid histories were available for 3,033/4,514
training contexts and 541/924 transfer contexts; missing histories remain
explicit and no contexts were dropped from the prediction comparison.

At 700 ms, across 870 valid transfer targets, XY RMSE was 16.192 mm for nominal
command integration, 5.890 mm for the fitted command-history control, and
5.817 mm with observed motion added. Corresponding yaw RMSE was 2.286, 0.838
and 0.830 degrees. The small added-motion XY improvement appeared in both
development geometry clusters. It is not evidence of an online advantage.

For available transfer histories, the observed past 300-ms displacement had
0.509-mm XY RMSE against posthoc native truth (95th percentile 1.032 mm).
This accuracy applies only to the 541 available histories, with synthetic
2-mm depth noise and ideal gyro. The component's median feature-computation
time was 31.6 ms, excluding disk loading and the rest of the navigation loop.

The immediate finding is that a simple learned command-history predictor is
a strong comparator. Adding a second motion-estimation pipeline has not yet
shown enough benefit to justify runtime integration. The next useful check
is to evaluate the frozen command-history predictor on recorded closed-loop
execution before deciding on a matched prospective navigation comparison.
No new neural input or navigation controller has been promoted by this study.
The command-history executed check and subsequent targeted pulse-data learning
experiment are now complete; see `go2_short_pulse_learning_2026-09-16.md` for
the measured improvements, regressions and prospective navigation next step.
