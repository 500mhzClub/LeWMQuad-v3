# Training-only translation intercept correction V1

The frozen short-horizon models have substantial signed translation bias on
their own training data. Fit a distinct, closed-form XY intercept correction
for every trained head of all eighteen final models: six direct-only models
and twelve models with direct and rollout heads, thirty heads total. Preserve
the original snapshots. Do not select a model, seed, checkpoint, horizon or
correction magnitude from new transfer or native results.

For each head and each of its eight 100–800-ms horizons, fit both body-X and
body-Y offsets. Use all available training contexts and the exact model seed's
original 1,200-by-six draw schedule. A valid target from context i receives
weight equal to its draw count divided by its number of valid motion horizons.
The offset is the weighted mean of prediction minus target. This is the
closed-form weighted squared-error intercept estimator; it does not change
or claim to refit the original smooth-L1/JEPA objective. Missing and contact-
censored motion contributes no target. Every horizon requires training support.

Compute means in float64 and freeze the applied coefficients as float32, matching
the model outputs. At inference subtract the same horizon's two coefficients
from each valid predicted XY value. Preserve exact zero unknown padding, all
clocks/masks, yaw components, latent predictions and contact logits. Fit no yaw,
hazard or probability calibration and infer no error bound. This adds sixteen
scalar coefficients per head, 480 total; no optimizer or neural-weight update.

Existing all-model admission may reverify already-recorded uncorrected transfer
scores. The coefficient estimator receives only training indices, predictions,
labels and draw counts. It receives no native trajectories, target-layout
residuals, new corrected transfer scores, future RGB tensors or checkpoint
selection signal. The phase-coverage gap remains disclosed and unchanged.

Bind original fit result
`45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418`
and target windows
`c293e454ac391da377a282c61dc30dd6abbdd5274d253c963e4837a78e8f7811`.
Freeze this protocol, fitting/verification/wrapper code, focused tests and
diagnosis before the exclusive `go2_training_translation_bias_v1_attempt_001`
root. Save all coefficients and their completion receipt before any corrected
transfer evaluation. Reverify every bound source/input afterward; preserve any
failure. Use one CPU process and one numerical thread, at least 8 GiB available
RAM and 256 MiB output allowance above the unchanged 40-GiB reserve.

A separate readout must reconstruct all coefficients and evaluate all prescribed
models, roles and strata before native use. Keep seed-2026091001 full-JEPA and
full-direct as the next native pair, with the eight-step planner and original
sensor, physical, clearance, timing and arrival rules. Correction failure or
negative results do not authorize silent fallback, refitting or a retry.

This is development fitting of output intercepts. It establishes neither
navigation success, independent-maze generalization, JEPA/RGB benefit,
probability calibration, real-time performance nor hardware validity.
