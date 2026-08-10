# Proprioception × rollout factorial: final unblinded analysis

Date: 2026-08-10
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.**
Report digest: **`60b0bb2d0b13ba47eac5e306c33d97dcfdce31102870edfc50b01f7f9b247161`**

Eight complete, technically valid seed quadruplets. The frozen final analysis was
run **once**. Confirmatory, secondary and diagnostic findings are separated below
and must stay separated. No combined H=2–3 endpoint was formed, no checkpoint was
selected, and no success threshold was introduced after unblinding.

---

# 1. Confirmatory result

**Estimand** `I_s = (PropRoll_s − PropOne_s) − (RGBRoll_s − RGBOne_s)`
**Estimator** frozen equal-family H=2: valid tokens within a row → rows within an
episode cluster → episodes within a family → unweighted mean of eight families.
**Replication unit** the training seed quadruplet. **N = 8.**

## Individual interactions

| seed | `I_s` |
|---|---:|
| 2026080901 | −0.002559 |
| 2026080902 | −0.003082 |
| 2026080903 | −0.000567 |
| 2026080904 | −0.001338 |
| 2026080905 | +0.002260 |
| 2026080906 | +0.004649 |
| 2026080907 | +0.001385 |
| 2026080908 | +0.004155 |

| quantity | value |
|---|---:|
| mean interaction | **+0.000613** |
| sample standard deviation | 0.002953 |
| two-sided 95 % *t*-interval | **[−0.001856, +0.003082]** |
| interval excludes zero | **no** |
| final seed count | 8 |

> **The interaction is not distinguishable from zero.** The interval is well
> inside the ±0.005 minimally relevant effect declared in advance, so this is an
> informative null at the resolution the study was powered for, not an
> uninformative one. Four seeds are negative and four positive.

## Δ_RGB and Δ_prop, reported separately

| | mean | sd | 95 % *t*-interval |
|---|---:|---:|---|
| **Δ_RGB** = RGBRoll − RGBOne | **+0.008077** | 0.001972 | [+0.006428, +0.009725] |
| **Δ_prop** = PropRoll − PropOne | **+0.008690** | 0.002052 | [+0.006974, +0.010405] |

Both rollout advantages are clearly separated from zero and closely similar. The
rollout objective helps; proprioception does not measurably change how much it
helps.

## The four cell means, exact frozen estimator

| cell | H=2 equal-family cosine | sd |
|---|---:|---:|
| rgb_one_step | 0.720395 | 0.004378 |
| rgb_rollout | 0.728472 | 0.003715 |
| proprio_one_step | 0.720302 | 0.002980 |
| proprio_rollout | 0.728992 | 0.004231 |

The two one-step cells differ by −0.000093 and the two rollout cells by +0.000520:
adding proprioception moves the H=2 score far less than the rollout objective does.

## Variance re-estimation record

| | |
|---|---:|
| `s_I` at interim (N=5) | 0.0021021 |
| `σ_U`, 90 % one-sided | 0.0040765 |
| minimally relevant interaction | 0.005 |
| α (two-sided) / target power | 0.05 / 0.80 |
| frozen total N | **8** |
| calculated power at N=8 | 0.8435 |
| precision-limited | no |
| recalculated after freezing | **no** |

The realised sd over eight seeds (0.002953) came in above the interim estimate
(0.002102) but below its 90 % upper bound (0.004077), which is what the bound was
for. The sample size was not revisited.

---

# 2. Secondary results

Secondary to the confirmatory equal-family result; never mixed with it.

## Correct-future cosine, H=1–4 (equal-family cell means)

| H | rgb_one | rgb_roll | prop_one | prop_roll | interaction |
|---|---:|---:|---:|---:|---:|
| 1 | 0.7512 | 0.7525 | 0.7511 | 0.7527 | +0.00038 |
| 2 | 0.7204 | 0.7285 | 0.7203 | 0.7290 | +0.00061 |
| 3 | 0.7006 | 0.7112 | 0.7004 | 0.7114 | +0.00039 |
| 4 | 0.6850 | 0.6967 | 0.6849 | 0.6973 | +0.00063 |

The rollout advantage is present at every horizon and **grows** with horizon
(+0.0013 at H=1 to +0.0117 at H=4). The interaction stays near zero throughout.

**H=3 is beyond-trained-horizon transfer; H=4 is a longer-horizon diagnostic.** No
combined H=2–3 endpoint was formed.

## Corpus-weighted (token-pooled), H=2 — secondary to equal-family

| cell | token-pooled | equal-family |
|---|---:|---:|
| rgb_one_step | 0.7080 | 0.7204 |
| rgb_rollout | 0.7190 | 0.7285 |
| proprio_one_step | 0.7073 | 0.7203 |
| proprio_rollout | 0.7189 | 0.7290 |

Same ordering, same conclusion; the two weightings are reported separately and are
never combined into one number.

## Correct-versus-shuffled action margin (co-outcome)

| H | rgb_one | rgb_roll | prop_one | prop_roll |
|---|---:|---:|---:|---:|
| 1 | 0.0598 | 0.0597 | 0.0599 | 0.0597 |
| 2 | 0.0691 | 0.0678 | 0.0691 | 0.0681 |
| 3 | 0.0566 | 0.0549 | 0.0563 | 0.0548 |
| 4 | 0.0495 | 0.0481 | 0.0493 | 0.0484 |

The one-step cells hold a slightly **larger** margin at H=2–4 (≈0.0013–0.0017), so
the rollout objective's fidelity gain comes with a small loss of action-sequence
discrimination — the same qualitative trade-off the RGB-only study reported.
Proprioception does not change it.

**No formal non-regression claim is made.** The frozen configuration declares no
numerical non-inferiority margins, so the harness sets
`formal_non_regression_claimable: false` and this is reported as a mandatory
co-outcome only.

## Occupied spatial co-outcome — reported as UNUSABLE

Every cell returns occupied IoU **exactly 0.5000**. That is a defect in my metric
definition, not a result: `occupied_metrics` thresholds at the **median** of the
valid-token cosines, so by construction half the tokens are "occupied" and the IoU
is identically 0.5 for any model. The co-outcome is therefore **uninformative as
implemented** and carries no evidential weight here. It needs an absolute,
model-independent threshold before it can discriminate anything. I am reporting
this rather than presenting 0.5000 as a finding.

---

# 3. Diagnostic results

## Terminal-window stability (never used for selection or exclusion)

| cell | terminal-window mean | sd | slope (epochs 14–23) |
|---|---:|---:|---:|
| rgb_one_step | 0.3227 | 0.00240 | −0.001802 |
| rgb_rollout | 0.6642 | 0.00417 | −0.003171 |
| proprio_one_step | 0.3223 | 0.00245 | −0.001815 |
| proprio_rollout | 0.6637 | 0.00422 | −0.003181 |

All four still improving slowly at the fixed budget. No run was excluded for
trend, and the fixed epoch-21 checkpoint was used regardless.

## Per-family H=2 interaction

| family | interaction | rgb_one | rgb_roll | prop_one | prop_roll |
|---|---:|---:|---:|---:|---:|
| local_composite_motifs | +0.00198 | 0.7383 | 0.7419 | 0.7378 | 0.7434 |
| small_enclosed_maze | +0.00153 | 0.6811 | 0.6978 | 0.6820 | 0.7002 |
| medium_enclosed_maze | +0.00123 | 0.6860 | 0.6974 | 0.6845 | 0.6971 |
| rough_local_dynamics | +0.00111 | 0.7422 | 0.7443 | 0.7413 | 0.7445 |
| visual_sensor_stress | +0.00074 | 0.7193 | 0.7283 | 0.7200 | 0.7298 |
| open_obstacle_field | +0.00017 | 0.7428 | 0.7489 | 0.7420 | 0.7483 |
| large_enclosed_maze | −0.00014 | 0.7521 | 0.7578 | 0.7522 | 0.7578 |
| loop_alias_stress | −0.00171 | 0.7015 | 0.7114 | 0.7026 | 0.7109 |

## `local_composite_motifs` — prospectively declared diagnostic

H=2 interaction **+0.00198**, the largest of the eight families but an order of
magnitude below the 0.005 minimally relevant effect and unsupported by any
interval. Nothing was tuned to this family, and equal-family reporting is
preserved throughout so it cannot be hidden by corpus weighting.

Its rollout advantage here (+0.0036 RGB, +0.0056 proprio) is the **smallest** of
any family, which is the same family the RGB-only study flagged for a
horizon-dependent control advantage. That earlier post hoc observation is
therefore **not contradicted, and not confirmed either** — it remains a
diagnostic, not a result.

---

# 4. Attempt lineage

| seed | attempts | restarts | cells | epochs | checkpoint |
|---|---:|---:|---:|---:|---|
| 2026080901–2026080904 | 1 each | 0 | 4 | 24 | epoch 21 |
| **2026080905** | **2** | 0 | 4 | 24 | epoch 21 |
| 2026080906–2026080908 | 1 each | 0 | 4 | 24 | epoch 21 |

**32 cells, every one trained for exactly 24 epochs, every one retaining the fixed
epoch-21 checkpoint, all technically valid.** No run was rerun, extended, omitted
or replaced because of its performance.

## Seed 4 refused-evaluation incident

> Seed index 4 (seed 2026080905): the first evaluation attempt was REFUSED by the
> launch guard because the source tree was dirty — a new, unimported module had
> been created in the repository while the stage was running. **No bound
> scientific artefact and no executed scientific source changed, and the first
> attempt produced no evaluation result.** The pinned launch state was restored
> and the read-only evaluation was re-run from the preserved epoch-21 checkpoints
> using the byte-identical evaluator that scored every other seed.

## Bindings, constant across all eight quadruplets

| artefact | digest |
|---|---|
| run package | `cf0456be…` |
| factorial manifest | `6ff05303…` |
| canonical cache map | `a45bcc7d…` |
| horizon masks | `ce32489f…` |
| normalisation contract | `f5ea58b2…` |
| selection rows | 475 |
| initial launch receipt (seeds 1–5) | `abe036ad…` |
| continuation receipt (seeds 6–8) | `5f337895…` |
| variance-only interim | `71d3dded…` |

Registry indices 8 and 9 were never launched and remain locked.

---

# 5. What this study establishes

**Confirmatory.** Under the frozen equal-family H=2 estimator, with the seed
quadruplet as the replication unit and N=8, **proprioceptive conditioning does not
make rollout supervision measurably more effective**: mean interaction +0.0006,
95 % interval [−0.0019, +0.0031], comfortably inside the ±0.005 minimally relevant
effect. This is an informative null.

**Secondary.** The two-step rollout objective produces a consistent
correct-future gain (Δ_RGB +0.0081, Δ_prop +0.0087, both intervals excluding
zero) that grows with horizon, accompanied by a small loss of action-sequence
discrimination. Proprioception changes neither.

**Limits worth stating.** The occupied co-outcome is unusable as implemented and
was excluded from interpretation. The proprioceptive contract deliberately
withheld foot contacts, joint torques and IMU linear acceleration because the
corpus logs them degenerately, and excluded body linear velocity as privileged; a
null for *this* deployment-valid subset is not a null for proprioception in
general. Observed-proprioception slots fall 3→2→1→0 across H=1–4 by design, which
predicted a shrinking Δ_prop advantage that the data neither shows nor needs.

---

## Stopping condition

Report written and hashed. Nothing is running. No navigation evaluation,
counterfactual data generation, architecture change or further training stage has
been started.
