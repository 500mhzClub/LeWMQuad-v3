# Frozen H = 1–4: cached reproduction, episode-cluster bootstrap, matched-epoch sensitivity

Date: 2026-08-09
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** Read-only throughout. No model was
trained, no checkpoint was written or created, no checkpoint selection was made or
revised, no convergence rule was altered, and the sealed benchmark was not touched.

Scope: the three analyses authorised after the candidate-ranking blocker was
accepted. Nothing marked FINAL is amended by this document.

---

## 1. Cached reproduction of the frozen H = 1–4 result

The H=1–4 evaluator was re-run on the corrected 479-row manifest with per-row
prediction caching enabled, so that the bootstrap could operate on row-level
scores rather than the stored per-scene aggregates.

**Result: identical, not merely within round-off.**

| | |
|---|---|
| leaf values compared | 349 |
| key sets identical | yes |
| values bit-identical | **348 / 349** |
| the one difference | `wall_seconds` 861.8 → 458.0 (target-encoding cache warm) |

Every score, difference, margin, per-scene value and changed-token count
reproduced exactly. `FINAL_horizon_result.json` therefore stands unchanged, and
no investigation or amendment was triggered.

Both outputs are retained: `horizons/FINAL/FINAL_horizon_result.json` (frozen) and
`horizons/evaluation/result.json` (reproduction). Per-row predictions:
`horizons/predictions/{model}_h{1..4}_{correct,shuffled}.f16`, 16 blobs, 11.8 GB,
cache only — not Git.

---

## 2. Paired episode-cluster bootstrap, stratified by family

### What was resampled, and what the intervals do and do not cover

**Resampling unit: the episode cluster** `(scene, env_index, episode_id,
reset_count)`. Rows drawn from one episode share overlapping frames and are not
independent, so row-level resampling would overstate the sample size. Maze-seed
level was considered and rejected *as the resampling unit*: the selection split
contains only eight scenes — too few clusters to bootstrap. Episode level is the
finer of the two permitted units.

**228 episode clusters** over 479 rows and 8 families:

| family | clusters | rows |
|---|---:|---:|
| local_composite_motifs | 39 | 64 |
| open_obstacle_field | 37 | 61 |
| rough_local_dynamics | 34 | 63 |
| large_enclosed_maze | 29 | 63 |
| visual_sensor_stress | 29 | 61 |
| loop_alias_stress | 26 | 60 |
| medium_enclosed_maze | 25 | 62 |
| small_enclosed_maze | 9 | 45 |

Clusters are drawn with replacement **within** each family, so every resample
preserves the observed family composition. Both models are scored on the *same*
resampled clusters at each draw, so differences are formed within a resample
(paired). 10,000 resamples, seed 2,026,080,901, percentile intervals at 95%.

> **Interval scope.** These intervals quantify **variation across episodes within
> the present eight families**. They are **not** intervals for generalisation
> across independently sampled maze populations: the eight families and the eight
> scenes are fixed and are never resampled. No interval here speaks to a ninth
> family or to an unseen maze population.

### Weightings reported side by side

The frozen FINAL point estimates pool masked cosines over all tokens
(corpus-weighted). That estimator is reported **unchanged** as primary; the
equal-family-weighted numbers are a **separate robustness analysis** and do not
replace the frozen point estimates.

### 2a. Corpus-weighted — the frozen estimator, unchanged (228 clusters)

Observed values reproduce `FINAL_horizon_result.json` exactly; the bootstrap adds
only uncertainty around them.

| H | rollout ep22 | control ep28 | Δ correct [95% CI] | Δ shuffled [95% CI] | Δ margin [95% CI] |
|---:|---:|---:|---|---|---|
| 1 | 0.7457 | 0.7441 | +0.0016 [−0.0012, +0.0045] | +0.0057 [+0.0031, +0.0085] \* | −0.0041 [−0.0072, −0.0011] \* |
| 2 | 0.7172 | 0.7103 | **+0.0069 [+0.0018, +0.0122] \*** | +0.0137 [+0.0090, +0.0189] \* | −0.0068 [−0.0120, −0.0018] \* |
| 3 | 0.6959 | 0.6901 | +0.0059 [−0.0004, +0.0120] | +0.0096 [+0.0032, +0.0161] \* | −0.0037 [−0.0115, +0.0038] |
| 4 | 0.6782 | 0.6787 | −0.0005 [−0.0066, +0.0057] | +0.0061 [−0.0007, +0.0131] | −0.0066 [−0.0141, +0.0008] |

\* interval excludes zero. Δ = rollout − control. "Margin" = correct − shuffled.

### 2b. Equal-family-weighted — robustness only (228 clusters)

| H | rollout ep22 | control ep28 | Δ correct [95% CI] | Δ shuffled [95% CI] | Δ margin [95% CI] |
|---:|---:|---:|---|---|---|
| 1 | 0.7500 | 0.7480 | +0.0020 [−0.0007, +0.0047] | +0.0062 [+0.0036, +0.0089] \* | −0.0042 [−0.0075, −0.0011] \* |
| 2 | 0.7221 | 0.7155 | **+0.0066 [+0.0020, +0.0117] \*** | +0.0146 [+0.0103, +0.0196] \* | −0.0080 [−0.0130, −0.0032] \* |
| 3 | 0.7029 | 0.6974 | +0.0055 [+0.0000, +0.0114] † | +0.0113 [+0.0056, +0.0172] \* | −0.0058 [−0.0128, +0.0012] |
| 4 | 0.6859 | 0.6859 | −0.0000 [−0.0055, +0.0057] | +0.0072 [+0.0007, +0.0139] \* | −0.0072 [−0.0143, −0.0001] \* |

† The H=3 lower bound is +0.00003 — nominally excluding zero, but at the
resolution of the resampling this should be read as **borderline, not
established**. Under the primary corpus weighting the same interval covers zero.

### What the intervals change about the frozen reading

- **H=2 is the only horizon at which the correct-future advantage is separated
  from zero under both weightings.** It is the directly supervised horizon.
- **H=3 is not separated from zero** under the primary corpus weighting
  (−0.0004 to +0.0120), and only borderline under equal-family weighting. The
  frozen document's "retained at H=3" is a point-estimate and family-count
  observation; the episode-clustered interval does **not** establish it.
- **H=1 and H=4 are indistinguishable from zero** under both weightings, matching
  the frozen reading.
- **The margin deficit is real and is the more robustly separated effect.** The
  control's larger correct-minus-shuffled margin excludes zero at H=1 and H=2
  under both weightings (and at H=4 under equal-family weighting). Note the
  mechanism: the rollout model scores *higher on shuffled sequences too*
  (Δ shuffled is positive and excludes zero at H=1–3), so its smaller margin
  comes from being less sensitive to which action sequence was applied, not from
  predicting the correct future worse.

### Per-family directional consistency (episode-clustered)

Families favouring rollout on correct-future score: **H=1 5/8, H=2 7/8, H=3 7/8,
H=4 5/8** — reproducing the frozen per-scene counts. Family intervals are wide
(9–39 clusters each) and mostly cover zero:

| H | families with Δcorrect CI excluding 0 |
|---:|---|
| 1 | 1/8 — open_obstacle_field +0.0100 |
| 2 | 3/8 — loop_alias_stress +0.0155, small_enclosed_maze +0.0123, rough_local_dynamics +0.0100 |
| 3 | 1/8 — small_enclosed_maze +0.0211 |
| 4 | 2/8 — small_enclosed_maze +0.0200, local_composite_motifs **−0.0313** |

`local_composite_motifs` is the one family where the control is separated from
rollout, and it worsens with horizon (+0.0041 → −0.0078 → −0.0170 → −0.0313). It
also has the most clusters (39). `small_enclosed_maze` favours rollout at H=2–4
but has the fewest clusters (9) and its interval should be treated as the least
reliable of the eight.

Full record: `horizons/bootstrap/result.json`.

---

## 3. Matched-epoch sensitivity: rollout ep22 vs control ep22 — EXPLORATORY

> **This result was not used for checkpoint selection and must not be.** It is an
> exploratory equal-duration comparison only. The selected checkpoint remains the
> rollout bundle at epoch 22, unchanged.

The FINAL comparison pairs rollout epoch 22 against control epoch **28** —
unequal training duration, because the rollout run terminated at epoch 23. This
section holds duration fixed at epoch 22 for both arms.

### Binding and verification

A fresh manifest was hashed before either model was scored, and the evaluator
re-verified both digests at load time and refused-on-mismatch:

| model | epoch | SHA-256 | verified | vs FINAL selection |
|---|---:|---|---|---|
| rollout bundle | 22 | `270aabb910fe01f3…` | yes | **same checkpoint** |
| one-step control | 22 | `b8530a7597267ba1…` | yes | different (FINAL used ep28) |

Inputs are the exact corrected 479-row manifest,
SHA-256 `644a257803b5d49dc05a8e5b90b057b1558e2b4c22208f64070d2cc218fce0cd`, with
the same frozen mask thresholds and the same derangement seed. Changed-token
counts are identical to FINAL (92,046 / 90,815 / 106,308 / 114,942), and the
rollout arm's scores reproduce FINAL exactly to all printed digits — confirming
that only the control checkpoint changed.

Manifest: `horizons/matched_epoch22/models_epoch22.json`.
Result: `horizons/matched_epoch22/result.json`.

### 3a. Correct-future score and paired difference

| H | rollout ep22 | control ep22 | Δ (roll − ctrl) | Δ at FINAL (ctrl ep28) |
|---:|---:|---:|---:|---:|
| 1 | 0.7457 | 0.7461 | **−0.0003** | +0.0016 |
| 2 | 0.7172 | 0.7115 | **+0.0057** | +0.0069 |
| 3 | 0.6959 | 0.6903 | **+0.0056** | +0.0059 |
| 4 | 0.6782 | 0.6754 | **+0.0028** | −0.0005 |

Training the control six epochs longer helps it most at H=1 and H=4 and barely at
all at H=2–3. Holding duration equal, the rollout arm's advantage is **similar at
H=2–3 and now positive rather than negative at H=4**, while H=1 flips to a
negligible deficit.

### 3b. Family directional consistency

Families favouring rollout: **H=1 3/8, H=2 7/8, H=3 6/8, H=4 6/8** (per-scene
counts from the evaluator agree: control better in 5, 1, 2, 2 of 8).

Family intervals excluding zero: H=1 1/8 (open_obstacle_field +0.0066); H=2 2/8
(medium_enclosed_maze +0.0125, open_obstacle_field +0.0120); H=3 2/8
(medium_enclosed_maze +0.0152, small_enclosed_maze +0.0114); H=4 3/8
(medium_enclosed_maze +0.0201, rough_local_dynamics +0.0057,
local_composite_motifs **−0.0249**).

`local_composite_motifs` again favours the control and again worsens with
horizon — the one family-level effect that persists across both the FINAL and the
matched-epoch pairing.

### 3c. Shuffled-action score and correct-minus-shuffled margin

| | H=1 | H=2 | H=3 | H=4 |
|---|---:|---:|---:|---:|
| rollout ep22, shuffled | 0.6793 | 0.6395 | 0.6274 | 0.6190 |
| control ep22, shuffled | 0.6775 | 0.6303 | 0.6190 | 0.6120 |
| **rollout margin** | 0.0665 | 0.0777 | 0.0685 | 0.0592 |
| **control margin** | **0.0686** | **0.0812** | **0.0713** | **0.0634** |
| rollout normalised error | 0.4892 | 0.4935 | 0.5264 | 0.5564 |
| control normalised error | 0.4886 | 0.5034 | 0.5362 | 0.5612 |

**The control retains the larger correct-minus-shuffled margin at every horizon
even at matched duration.** So the discrimination deficit reported in FINAL is
not an artefact of the control's six extra epochs — it survives equal training
duration. It is, however, roughly half the size: the margin gap narrows from
0.0041/0.0069/0.0037/0.0066 (FINAL) to 0.0021/0.0034/0.0028/0.0042 (matched).

### 3d. Paired episode-cluster uncertainty (228 clusters, same procedure as §2)

Corpus-weighted (primary):

| H | Δ correct [95% CI] | Δ shuffled [95% CI] | Δ margin [95% CI] |
|---:|---|---|---|
| 1 | −0.0003 [−0.0023, +0.0016] | +0.0018 [+0.0001, +0.0035] \* | −0.0021 [−0.0041, −0.0001] \* |
| 2 | **+0.0057 [+0.0016, +0.0099] \*** | +0.0091 [+0.0057, +0.0128] \* | −0.0034 [−0.0078, +0.0007] |
| 3 | **+0.0056 [+0.0002, +0.0109] \*** | +0.0084 [+0.0033, +0.0136] \* | −0.0028 [−0.0088, +0.0030] |
| 4 | +0.0028 [−0.0034, +0.0090] | +0.0070 [+0.0017, +0.0122] \* | −0.0042 [−0.0104, +0.0018] |

Equal-family-weighted (robustness):

| H | Δ correct [95% CI] | Δ shuffled [95% CI] | Δ margin [95% CI] |
|---:|---|---|---|
| 1 | +0.0001 [−0.0022, +0.0021] | +0.0017 [+0.0001, +0.0032] \* | −0.0017 [−0.0039, +0.0004] |
| 2 | **+0.0056 [+0.0019, +0.0095] \*** | +0.0093 [+0.0062, +0.0128] \* | −0.0037 [−0.0082, +0.0004] |
| 3 | +0.0042 [−0.0005, +0.0091] | +0.0088 [+0.0046, +0.0134] \* | −0.0046 [−0.0102, +0.0008] |
| 4 | +0.0019 [−0.0032, +0.0072] | +0.0073 [+0.0027, +0.0120] \* | −0.0054 [−0.0107, −0.0000] \* |

### 3e. Exploratory reading

At matched duration the correct-future advantage separates from zero at **H=2
under both weightings and at H=3 under the primary weighting** — slightly
*stronger* evidence for extension beyond the trained horizon than the FINAL
pairing gave, where H=3 covered zero. The margin deficit persists in sign at
every horizon and under both weightings, but at matched duration it separates
from zero only at H=1 (primary) and H=4 (equal-family), i.e. it is weaker than
FINAL suggested.

This is a sensitivity check, not a re-selection: it says the FINAL qualitative
picture is not created by the control's extra six epochs, while indicating that
the FINAL pairing somewhat understates the H=3–4 advantage and somewhat
overstates the discrimination deficit. **No checkpoint decision follows from it.**

---

## 4. Recorded limitations

### 4.1 Candidate ranking and rank regret are not estimable from textured_v03

**0 of 479 states contain multiple realised candidate continuations.** The corpus
records one factual continuation per state, so it supplies no counterfactual
oracle ordering: there is no set of alternative actions from the same state whose
realised outcomes could define a ground-truth utility ranking, and therefore no
rank-regret quantity to estimate. This is a property of the corpus, not a tuning
or implementation gap, and no candidate-ranking or rank-regret endpoint is
reported anywhere in this document.

No new counterfactual corpus was generated, no V3 branch data was used or
adapted, the render contract was not modified, and the metric-validity study was
not repurposed.

### 4.2 Rollout epoch 28 versus control epoch 28 is unavailable

The rollout run terminated at **epoch 23**, so no rollout epoch-28 checkpoint
exists. The missing checkpoint was **not created** and neither model was
retrained. The matched-epoch sensitivity in §3 is therefore run at epoch 22, the
only epoch at which both arms have a checkpoint that is also the rollout arm's
selected checkpoint.

### 4.3 The shuffled-sequence assay is a diagnostic, not ranking

The correct-minus-shuffled quantity throughout this document measures whether a
model's predictions are **conditioned on, and discriminate between, action
sequences**. It compares one true action sequence against one deranged sequence
on the same state. It is **not** candidate ranking, **not** planning regret, and
**not** evidence about planning utility. It is reported as an action-conditioning
and discrimination diagnostic only.

---

## 5. Standing conclusions, unchanged

Nothing in this document changes checkpoint selection, the convergence rule, or
any conclusion recorded in
`lewm_go2_v03_horizon_rollout_result_2026-08-09.md`. The bootstrap adds
uncertainty to the frozen point estimates and the matched-epoch run is
exploratory; neither was used for selection.

## 6. Artifacts

| artifact | path |
|---|---|
| frozen result (unchanged) | `horizons/FINAL/FINAL_horizon_result.json` |
| reproduction | `horizons/evaluation/result.json` |
| per-row predictions | `horizons/predictions/` (16 blobs, 11.8 GB, cache only) |
| bootstrap record | `horizons/bootstrap/result.json` |
| epoch-22 checkpoint manifest (hashed) | `horizons/matched_epoch22/models_epoch22.json` |
| epoch-22 result | `horizons/matched_epoch22/result.json` |
| epoch-22 bootstrap | `horizons/matched_epoch22/bootstrap/result.json` |
| bootstrap script | `scripts/bootstrap_dev_v03_horizon_intervals_v1.py` |
| regression test | `lewm/tests/test_horizon_sequence_frame_action_mismatch.py` (8/8) |
