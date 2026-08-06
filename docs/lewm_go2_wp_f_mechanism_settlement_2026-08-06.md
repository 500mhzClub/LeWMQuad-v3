# WP-F mechanism settlement: horizon, frozen-Q, split, loss

Date: 2026-08-06
Status: **SETTLEMENT ONLY. No training launched.** `DEVELOPMENT_ONLY_NOT_CLAIM_BEARING`.
WP-E untouched. Supersedes §6 of `lewm_go2_wp_f_counterfactual_coverage_audit_2026-08-06.md`.

The three requested corrections are adopted: **Q is frozen-reference**, **ordinary
JEPA prediction is retained on every V3 branch**, and **horizon compatibility is
verified rather than assumed**.

---

## 1. Horizon compatibility — verified, and one real defect found

Both corpora were traced to their source timing records.

| | WP-E (`development_raw_supervision_v1`) | V3 matched-branch |
|---|---|---|
| action unit | command block, `block_size: 5` | command block, 5 command ticks |
| command rate | `command_dt_s = 0.10000000149` (10 Hz) | 5 policy steps/tick at 50 Hz = 0.1 s/tick |
| **action duration** | **5 × 0.1 s = 0.500 s** | **25 policy steps × 0.02 s = 0.500 s** |
| command encoding | `vx_body_mps[5]`, `vy`, `wz` per block | `requested_block` = 5 × `[vx, vy, wz]` |
| observation input | single current RGB | single current RGB (`context:2`, at tick 10) |
| successor timing | frame at current + 1 block | frame at branch end, tick 10 → 15 |
| physical step | median translation `0.0799` m | median displacement `0.060` m |

Evidence for the timing: V3 `timestamp_ns` deltas are uniformly `20000000` ns
(50 Hz) across exactly 25 `policy_step_index` values 0–24. WP-E `frame_index`
is a **global counter interleaved across 48 envs** — consecutive same-env rows
differ by 48, and the recorded `timestamp_ns` delta between them is `1e8` ns.
The stored h=1 delta of 240 global indices is therefore `240 / 48 = 5` env
control steps at 0.1 s = **0.5 s**, not 24 s. Same-env displacement over 5 steps
has median `0.0915` m on the sampled scene, consistent with the corpus-level
`0.0799` m.

**The two horizons are identical: one 5-command block at 10 Hz, 0.5 s.** Both
also share the same `[vx, vy, wz] × 5` command encoding, so one predictor
conditioning vector serves both.

### Defect found: the two corpora do not share a camera geometry

| | WP-E | V3 |
|---|---|---|
| stored resolution | `224 × 168` | `224 × 224` |
| horizontal FOV | `78.323°` | `78.323°` |
| **vertical FOV** | **`62.837°`** | **`78.323°`** |
| camera mount (body) | `[0.326, 0.0, 0.043]` | `[0.326, 0.0, 0.043]` (nominal) |

`native_preprocess` resizes everything to `112 × 112` **without preserving
aspect ratio**. WP-E frames are stretched vertically by 4/3; V3 frames are not.
Passed through one encoder unmodified, identical world content would land on
different tokens — a silent domain shift that would have confounded the whole
experiment.

**Correction, exact rather than approximate:**

```
tan(62.837°/2) / tan(78.323°/2) = 0.750000
224 × 0.750000 = 168.000 px
```

A **centre vertical crop of V3 from 224 → 168 px, before the 112×112 resize**,
reproduces WP-E's vertical FOV to four decimals (`62.8370°` vs `62.8370°`).
Horizontal FOV and camera mount already match. This crop is mandatory on every
V3 frame and is applied throughout the rest of this document.

Encoder token summary statistics (mean, std, per-token norm) are **not** evidence
either way here — the encoder's final LayerNorm pins them by construction
(std `0.885`, norm `12.2` for all three of WP-E, V3-raw, V3-cropped). The
justification for the crop is the analytic FOV identity, not a statistical fit.

**Audit figures restated under the crop** (V3 train role). The §2 item-7/8
numbers in the coverage audit were computed on uncropped frames; corrected:

| | uncropped (as published) | FOV-corrected |
|---|---:|---:|
| within-group branch cosine, mean | `0.8057` | `0.7996` |
| cross-state same-action cosine, mean | `0.5563` | `0.5446` |
| fraction of pairs below 0.95 | `0.901` | `0.9184` |

The coverage verdict is unchanged: within-group separation still sits well above
the cross-state reference, and ~92% of pairs remain separated at 0.95.

## 2. Development-selection split — from V3 *train* scenes only

The already-examined V3 eval role is **not** used for model selection.

The V3 train role is exactly balanced: 8 families × 4 scenes × 4 groups = 128
groups. The split takes **one scene per family** into selection by a deterministic
seeded hash, by whole scene and whole branch group.

```
split_seed = "wp_f_v3_selection_split_20260806"
rank scenes within each family by sha256(seed | family | scene_id); take rank 0
```

| | scenes | groups | branches | families |
|---|---:|---:|---:|---:|
| WP-F train | 24 | 96 | 864 | 8 |
| WP-F development-selection | 8 | 32 | 288 | 8 |

Scene overlap `0`. All 8 families present on both sides. No group is split.

Selection scenes: `large_enclosed_maze_8a6599d5327d`,
`local_composite_motifs_6eacf7dd091e`, `loop_alias_stress_f83f6936f345`,
`medium_enclosed_maze_2fb132318693`, `open_obstacle_field_9b54a4580f74`,
`rough_local_dynamics_cb43b1584f45`, `small_enclosed_maze_99c76ab39ad8`,
`visual_sensor_stress_91b7907220c2`.

The WP-E designated `train` / `checkpoint_selection` roles are unchanged and
remain scene-disjoint from V3 (raw overlap 0).

## 3. Frozen-Q entropy calibration — train-only, before temperature selection

`Q` is computed **once**, before training, from the frozen encoder of the accepted
normalised-state predictor baseline (`predictor_normalised_epoch40.pt`, whose
encoder never moved). It is never recomputed from the moving EMA encoder — an
action-invariant encoder would drive `Q` toward uniform and relax its own
supervision.

Measured on the **96 WP-F train groups only**. Max entropy is `log 9 = 2.1972`.
93.5% of rows have at least one physically degenerate partner (endpoint within
5 cm).

| `τ_t` | H mean | H p50 | H p90 | rows effectively one-hot (H<0.2) | mean `Q_ii` | mass on <5 cm partners |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.462 | 0.171 | 1.303 | **0.528** | 0.861 | 0.104 |
| **0.05** | **1.177** | **1.162** | **1.944** | **0.029** | **0.620** | **0.237** |
| 0.08 | 1.635 | 1.693 | 2.097 | 0.005 | 0.449 | 0.321 |
| 0.10 | 1.800 | 1.858 | 2.134 | 0.005 | 0.377 | 0.356 |
| 0.15 | 1.996 | 2.040 | 2.169 | 0.000 | 0.278 | 0.403 |
| 0.20 | 2.076 | 2.109 | 2.182 | 0.000 | 0.230 | 0.426 |
| 0.30 | 2.139 | 2.159 | 2.191 | 0.000 | 0.185 | 0.446 |
| 0.50 | 2.176 | 2.184 | 2.195 | 0.000 | 0.152 | 0.459 |

`τ_t = 0.02` fails the requirement outright — **53% of rows are effectively
one-hot**, and near-identical successors receive only 0.10 of the row mass.
`τ_t ≥ 0.15` fails the opposite way: entropy exceeds 91% of maximum and `Q_ii`
falls toward the uniform value `1/9 = 0.111`, so the label stops carrying
supervision at all.

**Selected and frozen: `τ_t = 0.05`.** Only 2.9% of rows are effectively one-hot,
median entropy is 1.16 (53% of maximum — genuinely soft), physically degenerate
partners receive 23.7% of the row mass, and the true successor retains a clear
preference at `Q_ii = 0.620` (5.6× uniform). This satisfies the stated
requirement — near-identical successors produce soft rows — without washing the
label out.

**`τ_p = τ_t = 0.05`**, frozen, so `L_match` is a proper soft-label cross-entropy
between two comparably scaled distributions rather than a mismatched pair.

## 4. λ calibration — one fixed value, no sweep

Measured on a **fixed train-only batch**: 8 WP-F train groups (72 branches) plus
16 WP-E train pairs, at the initial state (online encoder, EMA target and frozen
reference all equal — the point at which λ is frozen). Partial unfreeze as in
WP-E: `blocks.4`, `blocks.5`, `norm` — 26 tensors, 890,112 parameters. Frozen
blocks held in **eval** mode, fixing the WP-E module-mode defect.

| quantity | value |
|---|---:|
| `L_jepa` (WP-E pairs + V3 branches) | `0.549391` |
| `L_match` (unscaled) | `2.091061` |
| ‖∇ L_jepa‖ on trainable encoder | `5.747362e-01` |
| ‖∇ L_match‖ on trainable encoder, unscaled | `1.371827e+00` |
| raw ratio match / JEPA | `2.3869` |
| **cos(∇ L_jepa, ∇ L_match)** | **`-0.2293`** |

The matching gradient is **2.39× larger than the JEPA gradient unscaled**, so
λ = 1 would let it dominate. The assumed 0.1 was not correct by derivation, only
close by luck.

**Frozen: `λ_match = 0.105`**, placing the matching term at **25.1%** of the JEPA
encoder-gradient norm at the calibration point — material, clearly subordinate.
No sweep will be run.

One thing to carry into interpretation: `cos(∇L_jepa, ∇L_match) = -0.2293` is an
order of magnitude more negative than the WP-E JEPA/BEV pairing (`-0.0209`). The
two objectives genuinely pull the encoder in partly opposing directions. That is
the tension the work package exists to test — predictability versus action
distinguishability — but it means the matching term is not free, and any gain
must be read against a possible JEPA-side cost.

## 5. Final loss definition

For a V3 branch group with shared current frame `x_s`, actions `a_1..a_9`,
successor frames `y_1..y_9`, all V3 frames FOV-corrected by the 224→168 centre
crop:

```
z_s     = LN(f_online(x_s))                       # online encoder, partial unfreeze
t_i     = LN(f_ema(y_i))          .detach()       # EMA target
t̄_i     = LN(f_frozen_ref(y_i))   .detach()       # frozen reference, precomputed once
p_i     = LN(predictor(z_s, a_i, c_i))            # c_i = mean of the 5-command block

Q_ij    = softmax_j( cos(t̄_i, t̄_j) / τ_t )        # τ_t = 0.05, no grad, computed once
S_ij    = cos(p_i, t_j) / τ_p                     # τ_p = 0.05

L_match = -(1/9) Σ_i Σ_j  Q_ij · log softmax_j(S_ij)
```

Total objective:

```
L = L_jepa(WP-E pairs)          # MSE(p, t) in normalised token state, h=1
  + L_jepa(V3 branches)         # every action predicts its OWN EMA successor
  + λ_match · L_match           # λ_match = 0.105, frozen
  + L_bev                       # unchanged auxiliary, frozen parameters, live forward
```

The second term is correction #2: the matching loss **supplements** direct JEPA
prediction on V3 and does not replace it. Every branch still predicts its own
successor by ordinary MSE.

Negatives in `S` are drawn only from within the same group — same current state,
different action. That is what makes this a counterfactual objective rather than
a generic contrastive one.

**Why frozen-reference `Q` rather than EMA `Q`.** `Q` defines what the world
actually did; `S` measures what the predictor claims. If `Q` were recomputed from
the moving encoder, an encoder drifting toward action-invariance would flatten
`Q` toward uniform and thereby lower its own matching loss without ever improving
action discrimination — the objective would supervise itself into the failure
mode it exists to detect. A frozen reference fixes the target relation
independently of the thing under test.

**Why soft labels rather than one-hot.** 48% of branch pairs end within 5 cm.
A hard `p_i ↔ t_i` InfoNCE would demand separation the true successors do not
exhibit and would be minimised by fabricating differences the world does not
contain. Where successors are genuinely distinct, `Q_i` concentrates and the term
reduces to 9-way InfoNCE; where two actions truly lead to the same place, `Q_i`
spreads and the loss stops asking. The floor is set by the corpus.

## 6. Gate and calibration to record before interpreting any run

Primary metric, unchanged from WP-E and the one every encoder-moving recipe has
so far lost on: **correct − shuffled changed-cosine** on the WP-E designated
`checkpoint_selection` role, plus the three raw health diagnostics (token
variance, effective rank, temporal delta). The objective earns its place only if
action discrimination rises **without** the WP-E collapse signature.

Secondary, on the 32 WP-F selection groups: 9-way match accuracy under `S`.
Chance is `1/9 = 0.111`. The ceiling is **not** 1.0 — it is bounded by the
degeneracy in `Q` and must be computed from the successor states themselves and
reported alongside, or the number is uninterpretable.

## 7. Status

Settled: horizon compatibility (with the FOV defect found and corrected), the
frozen-Q entropy calibration and frozen `τ_t = τ_p = 0.05`, the exact 24/8 scene
split, the frozen `λ_match = 0.105`, and the final loss.

**No training run has been started.**
