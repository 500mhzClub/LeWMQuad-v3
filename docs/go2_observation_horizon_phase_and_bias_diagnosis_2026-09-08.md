# Observation-horizon phase coverage and translation bias

Read-only checks of the authenticated saved targets, predictions and completed
native trajectories identify two separate limitations. No checkpoint, command,
label, clearance threshold or completed result was changed by this analysis.

All **408 available training contexts** have departure-camera index congruent
to 3 modulo 5. Family contexts use offsets in five-tick increments and branch
contexts depart at frame 13. Continuous control forecasts at every camera
frame. This is a real context-coverage gap, but the observed errors do not
establish phase aliasing as their cause.

On the short-horizon native probe, each model executed five selected commands
in each residue class. JEPA's mean 100-ms XY errors by phases 0–4 were
33.551, 27.622, 28.325, 33.254 and 27.581 mm. Direct's were 20.975, 18.693,
18.213, 21.426 and 18.768 mm. The trained phase 3 was not consistently better.
The eight-step planning trajectories likewise retained high error at phase 3.
These are correlated observations from reused trajectories, not independent
replicates or estimates of an unseen-maze distribution.

There is substantial signed lateral prediction bias. In the short-horizon
probe, phase-specific mean predicted body-Y motion was 20.766–26.182 mm for
JEPA and 18.602–21.259 mm for direct, while actual means were approximately
-0.443 to 1.103 mm. The eight-step probes show the same broad discrepancy.
Native poses here are evaluator-only, used to measure completed 100-ms motion;
they were not provided to either controller or used to fit a correction.

This mismatch is already present in the training population. First-horizon
training labels have mean XY [5.514, 0.187] mm for the 336 family contexts and
[8.581, 0.449] mm for the 72 branch contexts. Body-Y standard deviations are
0.949 and 0.514 mm respectively.

For every final model, compute the mean signed XY residual over all 408 valid
first-horizon training targets and decompose mean squared error into squared
mean residual plus centered residual error. For the two fixed native models:

| Training diagnostic | Full direct, seed 2026091001 | Full JEPA, seed 2026091001 |
| --- | ---: | ---: |
| Mean signed XY prediction-minus-target residual, mm | [4.767, 15.047] | [-17.966, 23.615] |
| Raw XY RMSE, mm | 18.963 | 31.194 |
| Centered XY RMSE, mm | 10.510 | 9.626 |
| Fraction of MSE attributable to mean residual | 0.6928 | 0.9048 |

Across all eighteen final models this fraction ranged from 0.2313 to 0.9299.
The direction and magnitude varied across optimization seeds. Training errors
are therefore not explained solely by untrained frame phases or a native-input
distribution shift. The final constant-learning-rate updates had nonzero clipped
gradients; this alone does not prove which optimization mechanism caused the
bias. No favorable intermediate checkpoint was loaded or selected.

The centered errors are an algebraic diagnostic using training means, not a
validated corrected model or a new transfer/native result. Before adopting a
correction, define its training-only estimator and weighting, fit it for all
eighteen models without looking at new transfer/native outcomes, bind the
resulting identities, and evaluate every prescribed case. Preserve the current
two model assignments and original physical/sensor/arrival gates. Do not use
native residuals or target-layout outcomes to set correction coefficients.

The tool-recorded calculations checked these input bindings and each consumed
prediction/decision/physics artifact against its parent result. Frame intervals
were checked as 100 ms; the frozen native rotation function and geometry helpers
were hash checked. An initial inspection used a family-only metadata key and
stopped with `KeyError`; the completed coverage calculation uses each row's
common `observation_horizon_receipt.departure_tick`.

| Input | SHA-256 |
| --- | --- |
| Target windows | `c293e454ac391da377a282c61dc30dd6abbdd5274d253c963e4837a78e8f7811` |
| Eighteen-fit result | `45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418` |
| Short-horizon native result | `f484a0c3b95cb2f3d0868dca1f1f40d11ba0d29c42e12cc9604061870a0d5608` |
| Eight-step native result | `a42774f8a7ebd98a46dcf8c6a4c90c49e8551a6bcb0ccd3031810abd039b53c5` |

Neither phase coverage nor a training-bias diagnostic establishes navigation,
JEPA advantage, RGB usefulness, calibrated uncertainty or real-time operation.
