# Frozen split-calibration transfer to recorded navigation trajectories

Coefficients fitted on the branch assay were applied unchanged to executed-action
forecasts on all 2,404 matched navigation windows. Nothing was refitted, no
coefficient was selected from these errors, and no outcome for an unexecuted
alternative was used. Completed in 57.0 s, 3.8 MB of output, exit 0.

## Contract

| Role | Definition |
|---|---|
| Shared grouping context | `(cluster, prefix_action)` in the assay |
| Committed prefix | each window's recorded prefix, unchanged across candidates |
| Branch/candidate dimension | `pulse_action` |
| Reference bank | `forward`, `left_arc`, `right_arc` |
| Canonical bank indices | 1, 2, 3, asserted through the `ACTIONS` mapping |
| Mean weighting | equal over the three pulse candidates |

The reference mean is `p̄ᵢ = (p_forward + p_left_arc + p_right_arc)/3` and the
calibrated forecast is `ẑᵢ = zᵢ + α_common·p̄ᵢ + α_branch·(p_execᵢ − p̄ᵢ)`.

Frozen coefficients, loaded at full precision from `frozen_coefficients.json`:
action `α_common` 0.0805548706108622, `α_branch` 1.5103565474839344;
`no_future_action` `α_common` 0.05231042723646807, `α_branch` 0.0. The zero is a
structural convention for an unidentifiable coefficient — that arm emits identical
forecasts per shared history, so its branch deviations are exactly zero. It is not a
fitted gain estimated to be zero.

Coefficients were fitted pooled over 100–800 ms on the assay and evaluated at
100–700 ms here. This is **population and horizon transfer, not identical-horizon
replication**.

## Validation

All five checks passed before the run.

| Check | Result |
|---|---|
| Identity: coefficients (1, 1) reproduce the executed-action forecast | max abs 1.11e-16 |
| Persistence: coefficients (0, 0) reproduce the current visual state | max abs 0.0 |
| Candidate construction: shared history and prefix identical, only suffix differs | true |
| No-future-action invariance: candidate forecasts coincide | spread 0.0 |
| Baseline reproduction on checked windows | max abs 1.2e-9 |

At population scale the successor independently reproduces all three published
baselines: persistence 0.027609553 against 0.027610, action 0.030776615 against
0.030777, no-action 0.031301298 against 0.031301.

## Result

Pooled over 2,404 windows at 700 ms:

| Condition | MSE | vs persistence |
|---|---:|---:|
| Persistence | 0.027610 | — |
| Action, uncalibrated | 0.030777 | +11.47% |
| No-action, uncalibrated | 0.031301 | +13.37% |
| **Action, calibrated** | **0.026779** | **−3.01%** |
| **No-action, calibrated** | **0.026861** | **−2.71%** |

**The calibration transfers.** Frozen branch-assay coefficients move the action model
from 11.47% worse than persistence to 3.01% better, on a different population at a
different horizon, with nothing refitted. It improves in all four recordings
individually (0.022436/0.023174, 0.020962/0.021569, 0.044390/0.046437,
0.031554/0.031841), so the pooled gain is not one trajectory.

**The action-specific advantage does not transfer.** The calibrated no-action control
also beats persistence, by 2.71%. Action over no-action is **+0.31%** calibrated,
down from +1.68% uncalibrated. On the branch assay the same treatment produced a 15%
action gain while the no-action control failed to beat persistence at all. Here both
arms benefit almost equally, so on recorded navigation the benefit is the
common-innovation correction, not action conditioning.

## The calibration is condition-dependent

| Stratum | n | Persistence | Action raw | Action calibrated |
|---|---:|---:|---:|---:|
| In-bank (executed a pulse candidate) | 1,051 | 0.012049 | 0.023732 | 0.011555 |
| Out-of-bank | 1,353 | 0.039697 | **0.036249** | 0.038604 |

Calibration helps in-bank windows substantially and **hurts out-of-bank windows
relative to the raw model**. On out-of-bank windows the uncalibrated action model
already beats persistence (0.036249 against 0.039697) and calibration gives most of
that back.

By executed action the split is sharper still:

| Action | n | Persistence | Action raw | Action calibrated |
|---|---:|---:|---:|---:|
| hold | 558 | 0.001166 | 0.003572 | 0.001777 |
| forward | 469 | 0.009111 | 0.015019 | 0.007502 |
| left_arc | 324 | 0.014672 | 0.038197 | 0.015128 |
| right_arc | 258 | 0.014096 | 0.021405 | 0.014437 |
| left_turn | 501 | 0.043262 | 0.044729 | 0.042599 |
| right_turn | 294 | 0.106751 | **0.083817** | 0.101693 |

`right_turn` is the clearest case: the raw model is 21.5% better than persistence and
calibration destroys most of that advantage. A single common coefficient of 0.08,
fitted where the model over-predicts by roughly 5.4×, over-shrinks the windows where
it was predicting well. This is the condition-dependent-calibration outcome, not a
general repair.

Note also that on in-bank windows the calibrated no-action arm (0.011518) is marginally
better than the calibrated action arm (0.011555) — the action arm loses on its own bank.

## What this does and does not establish

Established: the frozen correction transfers as a forecast-quality improvement to a
different population and horizon, consistently across four recordings.

Not established, and not testable from these windows:

- correct ranking of unexecuted actions;
- counterfactual branch capture;
- centred error against a true three-branch target mean;
- improved action selection, navigation or any decision outcome.

Each window supplies **one** observed future. The three candidate forecasts supply the
reference mean the calibration needs; they do not create ground truth for the
alternatives. For an out-of-bank executed action the "branch deviation" is a deviation
from the **predicted reference-bank mean**, not a measured deviation from an observed
counterfactual target mean, and must not be reported as one.

The 1,353 out-of-bank windows are an explicitly labelled extrapolation: the executed
suffix was not a bank member, the three-pulse reference mean was retained unchanged,
and the executed sequence was forecast as an additional query rather than added to the
mean. No window was dropped, substituted or reweighted.

Windows overlap and are not independent. One seed, one assay, four recordings, two of
which are failed returns.

## Position

Split calibration is a real and transferable correction to this checkpoint's common
visual innovation, but on recorded navigation it is **not** an action-conditioned
benefit, and it is not uniformly beneficial — it degrades exactly those windows where
the uncalibrated model was already good. A single global common coefficient is too
blunt. The next question is whether a condition-aware or training-time correction of
common-innovation direction and magnitude retains the in-bank gain without the
out-of-bank cost, scored against this frozen split-calibrated baseline.

Recovering the declared broad training targets remains a separately identified
successor. Artifacts: `go2_anchored_navigation_transfer_v1_attempt_001`, result in
`go2_anchored_navigation_transfer_result_2026-09-18.json`.
