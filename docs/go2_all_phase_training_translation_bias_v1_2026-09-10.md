# Prospective expanded-context training-only translation correction

After complete admission of all18 expanded-data fits, fit the same per-horizon
XY residual intercept estimator used by the original navigation models. Use
only the4010 training prediction rows and their original native training targets,
with each fit's exact1200x6 schedule. Geometry-transfer labels and predictions
must not affect the coefficients. Fit all30 trained heads and480 scalars before
any corrected transfer evaluation. Preserve raw fitted predictions and models.

For each training context with m>0 valid motion horizons, each valid horizon
receives weight (schedule draw count)/m. The intercept is the weighted mean of
(predicted XY minus target XY); inference subtracts it. The36 available contexts
with no valid motion target stay in all4010-context and7200-draw accounting and
have zero XY correction contribution. Record their exact indices and draw
counts. Do not drop or reassign them, invent motion labels, or modify their
existing neural/contact training. Require positive training motion support at
every corrected horizon. Preserve yaw, contact outputs, known horizon clocks
and zero padding exactly. Native training targets are used; runtime native
state and transfer targets are not. This does not calibrate contact probability.

Use the exact complete full-fit result supplied by SHA-256, all18 ledger/raw-score
and evaluation-only snapshot admission, and all three exact expanded schedules.
No checkpoint selection or parameter updates. The primary model and matched
controls remain those fixed before fitting in the all-phase matched-fit protocol.
The correction runner opens only training prediction files while reconstructing
coefficients. Its prerequisite all-model admission rechecks the already saved
raw transfer scores, but those values are never estimator inputs.

Output go2_all_phase_training_translation_bias_v1_attempt_001 is exclusive.
Preserve any partial output/failure; no implicit retry/resume. Use explicit
current-stage metadata rather than inheriting unrelated model-training or
future-image-materialization flags. Require8GiB current available RAM and256MiB
output allowance above the40GiB artifact reserve. Record hardware. Bind all
source, fitted-model, schedule, correction and target identities before and
after work. Runtime correction is evaluation-only and uses only original causal
inputs; no native launch is performed here. Prospective navigation still needs
its own complete admission and execution protocol.
