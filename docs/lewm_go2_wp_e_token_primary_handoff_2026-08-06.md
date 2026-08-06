# WP-E closure: token-primary representation and joint JEPA

Date: 2026-08-06
Status: **WP-E CLOSED.** All results development-tier,
`DEVELOPMENT_ONLY_NOT_CLAIM_BEARING`. No claim-bearing rollout, no promotion, no
evaluation-role or sealed data opened at any point.

---

## 1. Accepted findings

**The learned-query BEV decoder is not the primary latent.** Under capacity-matched
21M-parameter probes at near-identical recall (0.402 vs 0.397), ViT patch tokens
gave occupied IoU `0.3024` and precision `0.5502` against the BEV state's
`0.1977` and `0.2824`. The BEV state was probed three ways — 195-parameter
per-cell linear, 8.7k per-cell shallow, 21M global — landing at `0.164`, `0.188`,
`0.198`, while tokens reached `0.302`.

**A frozen-encoder token predictor learns genuine action-conditioned dynamics.**
On the designated `checkpoint_selection` role (4,262 train / 495 selection pairs,
72 / 8 scenes, zero overlap), in normalised token state at h=1:

| arm | changed cosine | nErr vs identity |
|---|---:|---:|
| correct action | `0.6432` | `0.6653` |
| shuffled (mean of 3 derangements) | `0.5936` | `0.7577` |
| neutral (`hold`) | `0.5688` | `0.8040` |
| persistence (identity) | `0.4637` | `1.0000` |

Correct − shuffled `+0.0496`; correct − persistence `+0.1795`. Per-scene the
advantage held in **8/8 selection scenes**, all eight families represented
including `open_obstacle_field`.

**Target normalisation removes the contraction shortcut.** Comparing the
unnormalised and normalised joint recipes, frozen → partial:

| | unnormalised | normalised |
|---|---|---|
| JEPA loss | `0.1259 → 0.0205` (−84%) | `0.1460 → 0.1321` (−9.5%) |
| raw token variance | `0.5428 → 0.1489` (−73%) | `0.5428 → 0.4782` (−12%) |
| raw effective rank | `15.37 → 5.74` (−63%) | `15.37 → 19.14` (**+25%**) |
| raw temporal delta | `0.2971 → 0.1143` (−62%) | `0.2971 → 0.2807` (−6%) |

**Encoder tokens are not already in canonical normalised space.** Per-token mean
`-0.00019` but std `0.885` (range 0.832–1.031), because the final encoder
LayerNorm applies a learned affine (scale 0.7613–1.0941). `max|tokens −
layer_norm(tokens)| = 1.223`. Normalising only the target would have changed the
task, not the loss — and with a residual predictor whose output feeds back, would
have compounded per rollout step.

**h=1 is a fixed, adequate horizon.** All 5,172 pairs have frame-index delta
exactly 240. Token persistence `0.7979` at h=1, `0.7241` at h=2, `0.6586` at h=4.
Translation median `0.0799` m, yaw median `0.1155` rad. The corpus stores one
pair-level primitive and velocity label per transition; constancy at every
underlying simulator step is not independently documented.

**Both objective paths are live and nearly orthogonal** on the checked batch
(16 pairs, 26 trainable encoder tensors, 890,112 parameters):

| path | loss | ‖grad‖ on trainable encoder |
|---|---:|---:|
| JEPA alone | `0.215039` | `1.351312e-01` |
| BEV alone | `0.018628` | `2.325603e-02` |
| combined | `0.233666` | `1.366390e-01` |

`cos(JEPA, BEV) = -0.020852`; linearity check `‖g_j + g_b − g_c‖ = 2.197e-07`.
Both paths reach all 26 tensors. The BEV scalar loss is **numerically smaller**;
that is not a statement about gradient influence.

## 2. Retractions and corrections

- **Off-distribution diagnostic retracted in full.** A missing ImageNet
  normalisation drove observable accuracy from `0.706` to `0.049` and produced a
  spurious "state head never predicts FREE". Artifact renamed
  `RETRACTED_result_offdistribution_inputs.json`. It also scored against a
  manifest-reconstructed target on the wrong corpus.
- **Occlusion bug.** An angular-bin visibility sweep left 1,194 free cells behind
  a wall; replaced with exact segment-rectangle intersection. Found by fixtures,
  not by inspection.
- **Silent split fallback.** A `--max-pairs` cap applied before role
  partitioning starved `checkpoint_selection` to zero, and a fallback silently
  substituted a random 39/13 scene holdout. Fallback removed, empty roles now
  fatal, roles partitioned before any cap. The earlier result is preserved as
  `SECONDARY_random_scene_holdout_39_13_result.json` — a genuine scene-disjoint
  result, not comparable in absolute terms to designated-role numbers.
- **Silent arm substitution.** A failed string patch left `v4_model_api`
  hardcoded, so arm A would have trained the v4 model. Caught only because the
  run crashed for an unrelated reason; an explicit trained-class assertion was
  added.
- **Overstated claims withdrawn.** That frozen dense features carry no
  scene-transferable geometry; that 57/64 command/displacement sign agreement
  bounds achievable conditioning; that additive row/column queries cannot
  represent isolated obstacles; that the true-future probe score is a "ceiling";
  that a marginal 0.0345 action-group statistic shows dynamics are mostly
  action-independent; that gradient "share" follows from loss magnitude.

## 3. Retrospective validity note — normalised joint run

- The partial-arm runner placed the **full encoder** in training mode rather than
  only blocks 4–5 and the final norm.
- The lower frozen blocks contain **no nonzero dropout, no stochastic depth, no
  mutable normalisation state and no buffers** (13 dropout modules, all `p = 0`;
  zero buffers; no BatchNorm).
- Measured eval-versus-train forward difference: **`8.583e-06`**, against
  bit-identical eval-versus-eval (`0.000e+00`).
- The deviation is therefore **real but non-material** to the paired conclusion.
- **Future joint runners must set lower frozen blocks explicitly to evaluation
  mode.** Parameter freezing alone does not disable dropout or mutable buffers.

A second scoping correction: in the frozen control the BEV loss had **no gradient
path to any trainable parameter**, so its zero BEV drift was guaranteed by
construction rather than demonstrating the freezing mechanism under load. The
paired treatment is properly stated as *predictor-only continuation versus
top-encoder adaptation under JEPA plus the fixed BEV spatial objective*.

## 4. Rejected

**Learned-query BEV decoder as primary latent.** Retained only as an auxiliary
geometric output / planning readout.

**O-field occupied reweighting (28:1 → 5:1).** Arm A reproduced the official
`update_400` checkpoint **bit-for-bit** (`5387c155ae0f5f19…`), so the comparison
was causally credible. Arm B worsened held-out BEV occupied precision and IoU
under both probes, worsened free-space IoU, and narrowed the token→BEV gap only
by degrading the tokens. Its Stage 3 gain was a calibration shift.

**Unnormalised encoder-moving joint recipe.** Collapse: variance −73%, effective
rank −63%, temporal delta −62%, with action-conditioning down 65%.

**Target-normalised encoder-moving joint recipe.** Collapse fixed, but the
adaptation **improved generic predictability while uniformly weakening action
discrimination and adding essentially no spatial recovery over persistence**:

| metric | frozen control | partial |
|---|---:|---:|
| correct changed-cosine | `0.6423` | `0.7084` (+0.0661) |
| **correct − shuffled** | `0.0538` | `0.0456` (**−0.0082**) |
| per-scene advantage | 8/8, mean `+0.0537` | 8/8, mean `+0.0454`, **weaker in all 8** |
| fixed probe, predicted vs persistence occ IoU | `0.1146` vs `0.1192` | `0.1240` vs `0.1240` (**equal**) |
| fresh probe, predicted vs persistence occ IoU | `0.1169` vs `0.1217` | `0.1331` vs `0.1291` |
| raw variance / rank / temporal delta | reference | `0.4782` / `19.14` / `0.2807` |

Two of three raw health gates breached (variance −12%, temporal delta −6%).

## 5. Surviving checkpoints

| artifact | status |
|---|---|
| `predictor_normalised_epoch40.pt` | **non-JEPA predictor-feasibility baseline.** Demonstrates action-conditioned token prediction. **Not the thesis endpoint — its encoder did not move.** |
| `predictor_only_designated_role_epoch40.pt` (`5993e23c…`) | raw-state predecessor; superseded by the normalised-state line |
| `DEVELOPMENT_ONLY_token_joint_normalised_h1/arm_{frozen,partial}` | rejected recipe, retained as evidence |

None is claim-bearing. None passed the reviewed execution path.

## 6. Exact next unresolved research question

> **Can an encoder be moved by a joint objective such that its representation
> becomes *more* action-discriminative rather than merely more predictable?**

Every encoder-moving recipe tested improved generic predictability while
weakening the correct-versus-shuffled margin. The objective rewards futures being
predictable; nothing rewards futures being *different under different actions*.
The measured deficit is action discrimination under encoder adaptation, not
collapse and not spatial-information loss.

The narrowest untested intervention — **not launched, and not authorised by this
closure** — is an action-contrastive term at the same current state, penalising
predicted-state similarity between the correct action and the other eight
branches already present in the corpus, under the same partial unfreeze, LR
ratio, EMA, normalised state, BEV auxiliary, horizon and split.

## 7. Open repository defects, unrelated to WP-E

- **Frozen-module pin drift.** `run_go2_shared_jepa_v5_matched_training_v1.py`
  changed at `477bfb3` (one line, `np.asarray`); 11 tracked contracts still pin
  the superseded hash. Blocks `contract.current_source_bindings` and one test
  collection. Not repinned — it would assert safety on behalf of 11 experiments.
- **v20/v21 launcher module-name collision** under single-process collection.
- **Model-chain circular import**: `direct_egocentric_bev_state_jepa_v3_...`
  imports `lewm.models.direct_egocentric_bev_state_jepa_v1` partway through a
  source-loaded chain and observes a partially initialised module outside the
  official runner's import order.
