# Frozen predictive features with a separately fitted motion readout

The preceding joint fits degraded the command-history reference even on their
own training inputs. Reducing learning rate helped but did not remove that
degradation. This experiment tests whether useful motion information is present
in the frozen features but poorly extracted by the joint motion head.

Use three fixed feature sources: the lower-rate JEPA checkpoint, its matched
supervised checkpoint, and their common untrained initialization. Preserve the
encoder, history encoder, action-conditioned latent transitions, hidden motion
layer, command reference and contact output. Fit only the four final motion
outputs, using the same 4694 original training contexts and original 7200 draw
weights. Each context's weight is shared across its valid motion horizons, as
in the original per-context supervision. Contexts without a motion label remain
accounted for but contribute no motion-fit rows.

The fixed readout is ridge regression with penalty 1, an unpenalized intercept
and weighted training-only feature standardization. It replaces joint Adam
optimization of the final motion layer with a direct squared-error fit. This
also changes the motion-head fitting loss from smooth L1 to squared error;
improvement would not by itself separate optimizer effects from loss effects.
The untrained-feature control tests whether training the representation adds
value beyond a generic nonlinear feature map. Neither control is a no-RGB
ablation. No navigation observations or labels enter any fit.

Feature standardization remains explicit in inference using float64 arithmetic;
motion outputs are cast to float32 before the existing composition. This avoids
cancellation from folding low-variance feature normalization into large affine
coefficients. No residual is damped or selected using navigation errors.

Plan: `docs/go2_frozen_motion_readout_plan_2026-09-17.json`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_motion_readout_v1_attempt_001/`.
Implementation: `lewm/frozen_motion_readout_development.py`;
fit: `scripts/fit_go2_frozen_motion_readout_development.py`;
evaluation: `scripts/evaluate_go2_frozen_motion_readout_development.py`.

Two focused checks passed in 2.15 s: weighted least-squares parity, and installed
head composition with unchanged contact/features and causal action prefixes.
All three feature streams and fits run sequentially in one process on core 8,
session 19629. Memory available before launch was 64 GiB and recording-volume
headroom 34 GiB. No simulation runs alongside this work. Reproducible feature
arrays remain transient; retain fit coefficients, identities and full scores.

Evaluate all three readouts, their two jointly trained parents and the reference
alone on identical inputs from the same 3726 executed windows in six exposed
recordings. Report per-run/action prefix, action-increment and whole-window XY
errors plus yaw error. Reproduce the preceding joint-head scores. Retain every
result; the training results are diagnostic and cannot establish generalization.
Saved alternative turn forecasts remain unexecuted. Reliable navigation still
requires subsequent prospective experiments on independent layouts.

## Training result

Feature extraction and all three fits completed in 50.30 s; owner 19629 exited
zero. Each fit used 33,904 valid motion rows. The 36 contexts with no valid
motion horizon account for 85 scheduled draws and remain recorded; the total
motion-fit weight is therefore 7115. All 4694 contexts remain in the score ledger.
No feature weight changed. The largest difference between the numerical ridge
output and installed float32 motion output was below 5.97e-8, and all three
installed models reloaded with exact inference agreement on the saved input batch.

| Feature source | Training 300-ms XY RMSE (mm) | Training 700-ms XY RMSE (mm) | Training 700-ms yaw RMSE (degrees) |
| --- | ---: | ---: | ---: |
| Command reference only | 2.85 | 5.68 | 0.582 |
| Frozen JEPA + fitted readout | 1.98 | 3.93 | 0.484 |
| Frozen supervised + fitted readout | 2.24 | 4.30 | 0.524 |
| Untrained features + fitted readout | 2.21 | 4.17 | 0.486 |

The refit extracts training-set motion information the joint output heads failed
to use. The untrained-feature result shows that training improvement alone does
not establish a JEPA contribution. These are fitting diagnostics, not transfer
or navigation evidence. Navigation-recording evaluation launched in session 30830
after fitting finished, using the fixed six-recording population and all arms.

## Complete navigation-recording prediction result

Session 30830 completed and exited zero, evaluating all 3726 fixed windows.
The parent-model scores were reproduced, and all reference comparisons used
the same causal inputs. No native mission was executed by this readout.

| Model | Prefix XY RMSE (mm) | Whole 700-ms XY RMSE (mm) | Action-increment XY RMSE (mm) | Yaw RMSE (degrees) |
| --- | ---: | ---: | ---: | ---: |
| Command reference only | 3.51 | 7.14 | 4.89 | 0.641 |
| Frozen JEPA + fitted readout | 4.21 | 8.37 | 5.45 | 0.788 |
| Frozen supervised + fitted readout | 3.98 | 7.62 | 5.19 | 0.739 |
| Untrained features + fitted readout | 3.48 | 6.90 | 4.67 | 0.716 |
| Lower-rate joint JEPA | 6.11 | 8.55 | 4.88 | 0.813 |
| Lower-rate joint supervised | 4.27 | 7.88 | 5.01 | 0.741 |

The training advantage does not transfer as a JEPA advantage. The untrained
features with a fitted head have the lowest pooled XY errors, while the command
reference has the lowest yaw error. The JEPA readout improves prefix error over
its joint head but worsens the action-increment error. Against the original
1200-update JEPA currently used in navigation, its whole-window error improves
from 9.76 to 8.37 mm and yaw from 2.77 to 0.79 degrees. This warrants testing as
an experimental forecasting treatment, not adopting it as an established
improvement over the reference or declaring a benefit of JEPA training.

An additional saved-map diagnostic (`prediction_evaluation/saved_turn_clearance.json`)
uses only the existing observed-map snapshots at the seven previously diagnosed
visual-recovery conflict states. Preferred right turn passes the full 0.48-m
reserve predicate at 6/7 states for the JEPA readout, 2/7 for supervised readout,
4/7 for untrained readout, and 7/7 for the command reference. The full selector,
dispatch and unexecuted turns were not replayed or physically evaluated. These
counts diagnose forecast-dependent eligibility, not safe or successful escapes.

Next run four fixed exposed-layout missions with the same current controller:
command reference, JEPA readout, supervised readout and untrained-feature readout.
The command-reference arm still computes the JEPA readout for workload matching.
Do not alter models or controller between assignments, and retain every outcome.
The purpose is a small prospective navigation diagnostic of forecast treatment;
one execution per arm cannot establish repeatability, a causal JEPA advantage,
fresh-layout generalization or deployment readiness. This is separate from the
completed prediction study and does not promote any model to the default.
