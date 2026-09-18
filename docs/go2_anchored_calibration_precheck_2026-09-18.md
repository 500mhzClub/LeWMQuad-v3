# Calibration and action-signal pre-checks on the anchored visual predictor

Two post-hoc diagnostics run before committing to a seven-arm recent-visual-change
sweep. No training, no GPU, no new navigation. Both reuse the retained frozen
targets and the two fixed 1,200-update checkpoints. Every number below is
reproducible from `scripts/precheck_anchored_*_development.py`; console output is
retained in `go2_anchored_calibration_precheck_2026-09-18.json`.

The pipeline was validated by reproducing the published evaluator's
`persistence_mse` and `action_independent_effect_mse` to 1e-6 relative at every
horizon and role. The residual is float32 round-trip in the retained npz.

## Deviation from the registered fit population

The pre-registered intent was to fit the calibration scalars on the retained
training population. The broad 4,694-context fitting distribution's encoded
targets are **not retained** — `encoded_targets.npz` holds only the 36 branch-assay
rows. Re-creating them requires a full re-encode, which the current disk state
does not support.

All scalars below are therefore fitted on the branch-assay **train role**
(clusters 00/01, 18 cases in 6 groups) and frozen before evaluation on
**geometry transfer** (clusters 02/03). This remains a genuine unseen-geometry
holdout and no scalar was selected on transfer outcomes, but it is not the
population originally declared, and the broad-distribution behaviour is untested.

## Pre-check 2: the action-signal budget

Within each identical-history group, `S_branch` is the across-branch target
variance and `S_common` is the group-mean change from the current state.
`S_branch` is identically the evaluator's `action_independent_effect_mse`.

| Horizon | S_common (transfer) | S_branch (transfer) | branch / common | branch / persistence |
|---:|---:|---:|---:|---:|
| 100–300 ms | 0.00095–0.00116 | **0.000000** | 0.00 | 0.000 |
| 400 ms | 0.000570 | 0.001741 | 3.05 | 0.753 |
| 500 ms | 0.000441 | 0.003112 | 7.06 | 0.876 |
| 600 ms | 0.000690 | 0.002813 | 4.08 | 0.803 |
| 700 ms | 0.000737 | 0.002646 | 3.59 | 0.782 |
| 800 ms | 0.000752 | 0.002498 | 3.32 | 0.769 |

Two structural facts follow.

**Branches do not diverge before 400 ms.** They share a committed prefix, so
`S_branch` is exactly zero at 100, 200 and 300 ms. No predictor of any
architecture can show action-dependence at or below 300 ms on this assay. The
300 ms end of the 300–700 ms planning interval used elsewhere carries no action
information at all.

**Where branches do diverge, the action signal dominates true visual change.**
From 400 ms onward `S_branch` is 1.5–7× `S_common` and is 60–88% of persistence's
total error. The action-dependent component is the majority of what there is to
predict. It is small only relative to the current predictor's error, not
relative to the signal.

Per group at 800 ms, the budget is severely heterogeneous:

| Group (transfer) | S_branch | S_common | ratio |
|---|---:|---:|---:|
| cluster_02 / hold | 0.006106 | 0.000460 | 13.29 |
| cluster_02 / left_turn | **0.000107** | 0.000542 | 0.20 |
| cluster_02 / right_turn | 0.004469 | 0.000886 | 5.04 |
| cluster_03 / hold | 0.001322 | 0.000053 | 24.92 |
| cluster_03 / left_turn | **0.000010** | 0.000183 | 0.05 |
| cluster_03 / right_turn | 0.002974 | 0.002390 | 1.24 |

Both `left_turn`-prefix groups are effectively degenerate: their three branches
lead to near-identical futures. **The effective sample for any action question is
four groups, not six**, and not 18 cases or 2,404 windows. The train role shows
the same pattern (left_turn `S_branch` 0.000018 and 0.000006).

This is the empirical branch-information budget of this assay. With one recorded
outcome per action it does not separate action effects from execution
variability, and it is not a noise-separated causal quantity.

## Why branch retrieval reads 6/18 for everything

Every arm scores exactly **one win in every group**, in both roles — persistence
and the original visual JEPA included. That is not statistical chance; it is a
structural floor. An arm whose predictions barely vary across branches ranks the
three targets identically for all three candidates, so exactly one case per group
can win by construction.

The action arm's within-group prediction spread is 0.000017–0.000075 against a
common error of 0.0014–0.141 — a ratio of **0.0005 to 0.013**. Its spread is also
far below the true `S_branch` it should be matching (0.000039 predicted against
0.006106 actual for cluster_02/hold, 157× too small).

Removing the common bias lifts the count to 10/18 (train) and 8/18 (transfer),
but **every one of those gains is in a degenerate `left_turn` group** with
`S_branch` ≤ 0.000107. The four groups carrying real signal stay at the floor
whether centred or not. The retrieval metric as constructed cannot register
improvement in this regime and should not be used as a gate until the common
bias is removed.

## Pre-check 1: scalar innovation shrinkage

With innovation `p = ẑ − z_t` and true innovation `d = z_future − z_t`, the
training-only least-squares coefficient is `α* = Σ⟨p,d⟩ / Σ‖p‖²`.

A **single** scalar gives α* = 0.0959 (action) and 0.0523 (no-action), both inside
the declared 0 ≤ α ≤ 1 range. Horizon-averaged on held-out transfer, the action
arm falls from 0.022055 to 0.002297 against persistence 0.002386 — the gap to
persistence closes and is marginally crossed. But the centred component moves the
wrong way, 0.001378 → 0.001577, because uniform shrinkage destroys the branch
signal along with the bias. Under a single scalar the no-action control lands at
0.002304, statistically indistinguishable from the action arm's 0.002297.

A **split** calibration — separate scalars for the within-candidate-set common
innovation and the branch deviation — separates the two failures. Both scalars
are fitted on the train role only and frozen.

α_common = **0.0806**, α_branch = **1.5104**.

The common innovation is over-predicted about 12×. The branch deviation is
*under*-predicted by about a third. These are opposite-signed errors, which is
why one scalar cannot fix both.

Held-out geometry transfer at 800 ms:

| Variant | Total | Common | Centred (branch) |
|---|---:|---:|---:|
| Persistence | 0.003251 | 0.000752 | 0.002498 |
| Action, α = 1 | 0.036537 | 0.034443 | 0.002094 |
| **Action, split** | **0.002762** | 0.000847 | **0.001915** |
| No-action, α = 1 | 0.047465 | 0.044967 | 0.002498 |
| No-action, split | 0.003288 | 0.000790 | 0.002498 |

Under identical calibration treatment the action arm beats persistence by 15.0%
on total error; the matched no-action control does **not** (1.1% worse). The
no-action arm's centred error is pinned at `S_branch` by construction, since it
emits one forecast per shared history.

The action arm's held-out centred error captures **23.3%** of the branch variance,
up from 16.2% uncalibrated. That gain is concentrated in the real-signal groups,
not the degenerate ones:

| Group (transfer) | S_branch | Split centred | Captured |
|---|---:|---:|---:|
| cluster_02 / hold | 0.006106 | 0.004935 | 19.2% |
| cluster_02 / right_turn | 0.004469 | 0.003370 | 24.6% |
| cluster_03 / hold | 0.001322 | 0.001079 | 18.4% |
| cluster_03 / right_turn | 0.002974 | 0.002026 | 31.9% |
| cluster_02 / left_turn | 0.000107 | 0.000054 | 49.1% |
| cluster_03 / left_turn | 0.000010 | 0.000029 | −192.6% |

All four real groups improve, in a tight 18–32% band. The two wild values are the
degenerate groups and carry no weight.

Leave-one-group-out on the train role gives α_common 0.0717–0.1102 and α_branch
1.1799–1.6588. Both scalars are stable in sign and magnitude and α_branch exceeds
1 in all six folds; no single group drives the fit.

**Branch retrieval stays at one win per group in every variant, including the
split calibration.** Even correctly scaled, the branch deviation remains roughly
10× smaller than the residual common error, so no argmin flips.

The split calibration is deployable in principle: it centres *predictions* across
a candidate set, which is available at planning time. It does not use future
targets. The centred *metric* used for scoring here does, and remains diagnostic
only, consistent with the original evaluator's `centered_action_metric_not_deployable`.

## Against the declared promotion gate

| Requirement | Status |
|---|---|
| Beat persistence | **Met** with split calibration (15.0% held out); failed at α = 1 |
| Beat visual constant velocity | Not tested — requires Δz, not available |
| Beat recent-change no-action control | Not applicable yet |
| Degrade when actions shuffled | Weak: 1.8% mismatched-action penalty, retrieval unchanged |
| Identify correct branch above chance | **Failed** — structural floor in every variant |
| Rank true future above alternative futures | **Failed** — same floor |
| Retain improvement held out | **Met** — 23.3% centred, all four real groups, stable scalars |

Partial. This is not a promotion and no model is promoted here.

## What this changes

The binding constraint is not a missing motion input. It is a ~12× over-prediction
of common visual change that compounds with horizon (the action arm's common error
grows 0.0013 → 0.0344 from 100 to 800 ms while the true common change stays near
0.0008). That bias swamps a real, held-out, under-scaled action signal and
saturates the discrimination metric.

Accordingly the seven-arm recent-visual-change sweep is **not** the next step. It
tests a hypothesis — missing visual velocity — that this evidence does not support
as the dominant failure, and it would be scored on a retrieval metric that cannot
move and a total MSE dominated by a bias that calibration already removes.

The cheaper and better-aimed next steps, in order:

1. Fix the common over-prediction at training time rather than post hoc: the
   innovation normalisation, the cumulative-increment parameterisation and the
   loss weighting are all candidates, and the post-hoc α_common ≈ 0.08 gives a
   direct target to check against.
2. Re-specify the discrimination metric on candidate-centred predictions and
   restrict it to the four non-degenerate groups, or it will keep returning the
   floor regardless of model quality.
3. Recover the broad training population's targets, or accept that calibration
   claims rest on 18 cases in 6 groups.
4. Only then revisit Δz, with the action signal no longer masked.

## Not done

The navigation population was **not** evaluated. Its retained forecast rows hold
per-window `mse` only, not the latent vectors, so α cannot be applied post hoc.
Doing it needs an instrumented re-run of the navigation evaluator (~62 s plus
model load, about 2 MB of new output) which was not launched because both
candidate volumes are at 100% capacity.

One seed, two geometries, six groups of which four are informative, one assay.
Nothing here is a navigation outcome or a JEPA representation-training result.
