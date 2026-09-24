# Autoregressive rollout to H = 1–4: frozen rollout vs one-step control

Date: 2026-08-09
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** Read-only: no model was trained,
no checkpoint written, no frozen model modified. Both checkpoints were verified
against their recorded SHA-256 before use.

Artifacts: `horizons/horizon_manifest.json`, `horizons/evaluation/result.json`

---

# CONCLUSION

> The result supports a **bounded improvement in short-horizon future-state
> fidelity** from two-step autoregressive supervision, extending **one step beyond
> the trained horizon** — but **not** sustained four-step stability, and **not** an
> unqualified improvement in action-conditioned planning utility.

## Setup

479 valid sequences spanning all eight evaluation families. Context
`t−480, t−240, t`; at each horizon the predictor consumes its **own** previous
prediction, never the true intermediate token, under the same fixed sliding
three-frame context and per-token normalisation used in training. Actions `a1…a4`
are directly recorded command blocks from `frames.jsonl`, with `a1`/`a2`
cross-checked against the two-step manifest.

| model | epoch | SHA-256 | objective |
|---|---:|---|---|
| rollout bundle | 22 | `270aabb910fe01f3…` | `1.5·e1 + 0.5·e2`, fixed sliding context |
| one-step control | 28 | `24de7c0089d0a397…` | `e1` |

Rows per family: `local_composite_motifs` 64, `large_enclosed_maze` 63,
`rough_local_dynamics` 63, `medium_enclosed_maze` 62, `open_obstacle_field` 61,
`visual_sensor_stress` 61, `loop_alias_stress` 60, `small_enclosed_maze` 45.

**Scenes and families coincide 1:1 in this selection split** (one scene per
family), so per-scene and per-family counts below are the same quantity.

## Rollout advantage by horizon

| horizon | rollout | control | Δ (roll − ctrl) | families favouring rollout |
|---:|---:|---:|---:|---:|
| H=1 | 0.7457 | 0.7441 | **+0.0016** | 5/8 |
| **H=2** *(directly supervised)* | **0.7172** | 0.7103 | **+0.0069** | **7/8** |
| **H=3** *(not in the objective)* | **0.6959** | 0.6901 | **+0.0059** | **7/8** |
| H=4 | 0.6782 | 0.6787 | **−0.0005** | 5/8 |

One-step performance is effectively matched. The advantage is largest at the
directly supervised H=2 and **retained at H=3**, a horizon absent from the
training objective — so the gain is not confined to the trained horizon. It is
gone by H=4.

## Degradation and baselines

| | H=1 | H=2 | H=3 | H=4 |
|---|---:|---:|---:|---:|
| persistence baseline cosine | 0.4802 | 0.4270 | 0.4224 | 0.4216 |
| rollout advantage over persistence | 0.2655 | 0.2902 | 0.2735 | 0.2566 |
| control advantage over persistence | 0.2639 | 0.2833 | 0.2677 | 0.2571 |
| rollout normalised error | 0.4892 | 0.4935 | 0.5264 | 0.5564 |
| control normalised error | 0.4923 | 0.5056 | 0.5366 | 0.5555 |
| rollout degradation from H=1 | — | 0.0285 | 0.0498 | 0.0676 |
| control degradation from H=1 | — | 0.0338 | 0.0541 | 0.0654 |

Both models remain substantially above persistence at every horizon, with
normalised error rising smoothly from ~0.49 to ~0.56. **Graceful, not
catastrophic, rollout degradation.** Rollout degrades slightly less through H=3
and slightly more at H=4.

## Action-sequence margin — the qualifying observation

| | H=1 | H=2 | H=3 | H=4 |
|---|---:|---:|---:|---:|
| rollout, correct − shuffled sequence | 0.0665 | 0.0777 | 0.0685 | 0.0592 |
| **control, correct − shuffled sequence** | **0.0706** | **0.0846** | **0.0722** | **0.0659** |

Both retain a positive margin through H=4. But **the one-step control has the
larger shuffled-sequence margin at every horizon.**

These are different properties and should not be conflated: the rollout model
predicts the *correct* future more accurately, while the control *discriminates
wrong action sequences* more sharply. The converged-model selection was made on
the former. This result qualifies that selection rather than overturning it, and
it is why the conclusion above stops short of claiming an unqualified improvement
in action-conditioned planning utility.

## Method notes

**Masks.** H=1 and H=2 use the existing frozen thresholds; H=3 and H=4 reuse the
H=2 threshold. No threshold is fitted on the selection rows. Changed tokens per
horizon: 92,046 / 90,815 / 106,308 / 114,942 of 367,872.

**Defect found and fixed.** The first sequence build advanced `max_horizon`
whenever the next *action* existed, without requiring the next *frame* to exist,
so eight rows claimed H=4 with only four frames and the evaluation raised an
`IndexError`. The horizon is now bounded by the frames actually collected: 480 →
479 rows at H=4, with one row correctly demoted to H=3.

## Standing conclusions, unchanged

- **A. Matched-duration causal comparison through epoch 23:** ROLLOUT TEST
  INCONCLUSIVE.
- **B. Converged-model selection (unequal training duration, not a
  compute-matched causal estimate):** rollout bundle, epoch 22.
- **Attribution:** no practically detectable effect from `1.5·e1` versus `e1`
  under the tested Adam configuration (mean differences ≤ 5e−4 across 30 matched
  epochs), so the rollout advantage is **not** explained by heavier first-step
  weighting. The attribution arm did not qualify a checkpoint and terminated.

This horizon result adds the bound: that advantage holds to H=3 and disappears by
H=4, and it does not extend to shuffled-sequence discrimination.
