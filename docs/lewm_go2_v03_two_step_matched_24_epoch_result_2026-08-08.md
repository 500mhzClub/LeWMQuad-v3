# Two-step rollout supervision vs matched one-step control — epochs 0–23

Date: 2026-08-08
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** No manifest or authorization
status is inherited. `probability_calibration`, `evaluation`, `untouched` and
sealed data were never opened. The encoder never moved and was never executed in
training.

Artifacts (preserved, not to be overwritten by any continuation):
`two_step/evaluation/MATCHED_24_EPOCH_result_epochs_0_23.json`,
`two_step/evaluation/MATCHED_24_EPOCH_decision_epochs_0_23.json`,
`two_step/rollout_frozen/frozen_receipt.json`

---

# DECISION

> ## ROLLOUT TEST INCONCLUSIVE AT THE MATCHED 24-EPOCH BUDGET

Recorded precisely:

- **The rollout arm converged** under the prospective two-sided rule.
- **The control remained slightly but materially improving** — it missed on both
  criteria, narrowly.
- **Both arms passed the full one-step operational gate.**
- **`step2_superiority` was deliberately left `null`** because both arms had not
  converged. Comparing a converged arm against a still-drifting one is the exact
  confound the matched design exists to prevent, so no step-two verdict was
  computed.
- **Rollout selected epoch 22** (now permanently frozen).
- **Control's provisional selected epoch is 23**, and it is provisional only
  because that arm is being continued.

## Convergence, prospective two-sided rule

Fixed before any resumed epoch was read: middle window 18–20, late window 21–23;
converged only if `|late_best_IoU − middle_best_IoU| ≤ 0.005` **and**
`|mean margin(21–23) − mean margin(18–20)| ≤ 0.003`, for **both** arms. A decline
beyond 0.005 is classified as late-window deterioration, never as convergence.

| arm | middle best | late best | Δ IoU | \|Δ margin\| | classification |
|---|---:|---:|---:|---:|---|
| **rollout** | 0.35911 | 0.35600 | **−0.00311** | **0.00272** | **CONVERGED** |
| control | 0.35314 | 0.36055 | **+0.00741** | 0.00364 | **still improving** |

The two-sided correction mattered here in the direction it was made for: the
rollout arm's Δ is *negative*. Under the earlier one-sided test (`Δ ≤ 0.005`) any
decline, however large, would have passed automatically. Under the corrected test
it passes on merit — the decline is smaller than the threshold, and a decline
beyond it would have been named deterioration instead.

## One-step operational gate — both arms PASS

Selection rule, fixed before the resumed epochs: within 21–23, the highest
step-one occupied IoU that **also** beats matched persistence, holds margin
≥ +0.0586, beats `open_obstacle_field` persistence, and passes occupied-volume
calibration.

| arm | epoch 21 | epoch 22 | epoch 23 | selected | gate |
|---|:--:|:--:|:--:|---:|---|
| control | ✓ all four | ✓ all four | ✓ all four | **23** (0.3605) | PASS |
| rollout | ✓ all four | ✓ all four | ✓ all four | **22** (0.3560) | PASS |

**Every candidate epoch in the window satisfied all four conditions for both
arms.** This is the first time in this line that any predictor has done so — the
deficit reported in every earlier experiment was schedule length, not objective,
capacity, masking, or encoder freezing.

## Both arms at each selected epoch

Reported at both selected epochs so a rollout advantage cannot be an artefact of
the arms being read at different training durations.

### Epoch 22 (rollout's selection)

| | control | **rollout** | Δ (roll − ctrl) |
|---|---:|---:|---:|
| step-1 occupied IoU | 0.3586 | 0.3560 | −0.0026 |
| step-1 precision | 0.5934 | 0.5767 | — |
| step-1 recall | 0.4754 | 0.4819 | — |
| step-1 occupied fraction | 0.01414 | 0.01451 | — |
| step-1 margin | +0.0684 | +0.0665 | −0.0020 |
| `open_obstacle_field` IoU / P | 0.1586 / 0.3191 | 0.1575 / 0.2922 | −0.0011 |
| **step-2 cosine** | 0.7110 | **0.7177** | **+0.0067** |
| **step-2 normalised error** | 0.5041 | **0.4924** | **−0.0117** |
| **step-1→2 degradation** | 0.0347 | **0.0280** | **−0.0067** |

### Epoch 23 (control's provisional selection)

| | control | rollout | Δ (roll − ctrl) |
|---|---:|---:|---:|
| step-1 occupied IoU | 0.3605 | 0.3410 | −0.0195 |
| step-1 margin | +0.0696 | +0.0695 | −0.0001 |
| `open_obstacle_field` IoU | 0.1353 | 0.1375 | +0.0022 |
| **step-2 cosine** | 0.7087 | **0.7264** | **+0.0178** |
| **step-2 normalised error** | 0.5082 | **0.4772** | **−0.0310** |
| **step-1→2 degradation** | 0.0356 | **0.0234** | **−0.0121**

At both epochs the pattern is identical: the arms are indistinguishable on
step-one geometry and action margin, while the rollout bundle is consistently
better at step two on all three measures. That is suggestive, **not** a verdict —
the encoded `step2_superiority` test was not executed.

## Reference points

| | occupied IoU | precision | recall | occupied fraction |
|---|---:|---:|---:|---:|
| true future | 0.4971 | — | — | — |
| **persistence (gate)** | **0.3128** | 0.5067 | 0.4498 | 0.01645 |
| `open_obstacle_field` persistence | 0.1346 | 0.1954 | 0.3019 | — |
| two-step persistence latent cosine | 0.4267 | | | |

## Action-sequence conditions, rollout at epoch 22

| condition | step-1 cosine | step-2 cosine | step-2 normalised error |
|---|---:|---:|---:|
| **correct a0 / correct a1** | **0.7457** | **0.7177** | **0.4924** |
| shuffled a0 / correct a1 | 0.6776 | 0.6732 | 0.5701 |
| correct a0 / shuffled a1 | 0.7457 | 0.6621 | 0.5894 |
| shuffled a0 / shuffled a1 | 0.6776 | 0.6362 | 0.6347 |

The ordering is clean and interpretable. Corrupting **either** action degrades
step two, corrupting both degrades it most, and `correct a0 / shuffled a1` leaves
step one untouched (0.7457, identical by construction) while costing step two
0.0556 — confirming the second action is genuinely consumed at the second step
rather than ignored.

## Per-family step-one occupied IoU, epoch 22

| family | rollout | control |
|---|---:|---:|
| `small_enclosed_maze` | 0.6204 | 0.6036 |
| `rough_local_dynamics` | 0.4263 | 0.4216 |
| `local_composite_motifs` | 0.4237 | 0.4182 |
| `large_enclosed_maze` | 0.3889 | 0.4007 |
| `visual_sensor_stress` | 0.3790 | 0.3800 |
| `medium_enclosed_maze` | 0.3463 | 0.3603 |
| `loop_alias_stress` | 0.2339 | 0.2329 |
| **`open_obstacle_field`** | **0.1575** | **0.1586** |

Rollout is higher in four families and lower in four — a wash, consistent with
the aggregate.

## Training curves, both arms, 24 epochs

Identical 4,031/488 subset, identical initial weights (`830e2f05…`), fresh
optimiser, same fp16 caches, same runner, seed `2026080651`, batch 4, lr 3e-4.
Arms verified perfectly paired before resumption (data-order generator state
byte-identical, `3061747d…`; first-batch row IDs identical; pre-update `e1`
1.04152).

| epoch | ctrl `e1` | roll `e1` | roll `e2` | ctrl IoU | roll IoU | ctrl margin | roll margin |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.46056 | 0.46222 | 0.46861 | 0.0647 | 0.0609 | +0.0201 | +0.0162 |
| 5 | 0.35679 | 0.35877 | 0.38096 | 0.2626 | 0.2550 | +0.0483 | +0.0487 |
| 11 | 0.34137 | 0.34354 | 0.36490 | 0.3219 | 0.3230 | +0.0546 | +0.0549 |
| 18 | 0.32799 | 0.33130 | — | 0.3531 | 0.3496 | +0.0649 | +0.0651 |
| 20 | 0.32443 | 0.32823 | — | 0.3468 | 0.3349 | +0.0640 | +0.0639 |
| 22 | 0.32111 | 0.32521 | — | 0.3586 | 0.3560 | +0.0684 | +0.0665 |
| 23 | 0.31956 | 0.32398 | 0.34481 | 0.3605 | 0.3410 | +0.0696 | +0.0695 |

## Scope

The rollout arm optimises `1.5·e1 + 0.5·e2`; the control optimises `e1`. The
rollout arm therefore also carries **1.5× the weight on the one-step term**, so
this is an **official-inspired rollout-supervision bundle with fixed sliding
context**, not a pure rollout ablation. Total losses were never compared across
arms. The step-two context is a **sliding-three-frame adaptation**
(`[t−240, t, p1]`), not the official growing-context architecture.

## Rollout arm frozen

Permanently frozen at its selected converged checkpoint; **it will not be
resumed.**

| | |
|---|---|
| selected epoch | **22** |
| checkpoint SHA-256 | `270aabb910fe01f36b341a66232616f90d4bd48bedfd0d01334b292226d8b9d1` |
| bytes | 206,485,807 |
| converged | true (Δ IoU −0.00311, \|Δ margin\| 0.00272) |
| one-step gate | PASS |
| receipt | `two_step/rollout_frozen/frozen_receipt.json` |

The receipt additionally records the selection receipt with per-epoch condition
breakdowns, the convergence windows and rule, the training record including the
rollout-gradient assertion, the complete one-step and two-step battery at epoch
22, the reference points, the masks, SHA-256 for all 24 epoch checkpoints, and
SHA-256 for all seven runner and evaluator scripts.

## Continuation

Only the **one-step control** is continued, from its epoch-23 full-state
checkpoint, preserving predictor and optimiser state, fixed learning rate, all
four RNG streams, the data-order generator, the `e1` objective, the frozen
encoder, and every cached feature, row, ordering, mask, derangement and probe.
Epochs 24–29, then at most one further block 30–35. **No automatic continuation
beyond epoch 35.**

When the control converges, two comparisons follow, and the second must be
labelled honestly:

1. **This matched-duration result through epoch 23** — compute-matched, preserved
   above and not to be overwritten.
2. **Rollout's frozen converged checkpoint vs the control's converged
   checkpoint** — **converged-model selection with unequal training duration, not
   a compute-matched causal comparison.**

The encoded `step2_superiority` test is then executed exactly as already written,
with no change to its thresholds or action-sequence definitions. Rollout is
selected only if it passes that test *while* preserving the complete one-step
geometry, action-margin, occupied-volume and `open_obstacle_field` gates.
Otherwise the simpler one-step control is selected.

Nothing else — proprioception, action tokens, longer context, capacity, encoder
movement, or the `1.5·e1` attribution arm — is launched before a winner is
selected.
