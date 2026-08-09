# Transition audit: officialised proprioceptive architecture

Date: 2026-08-09
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** Read-only audit. No training was
started, no corpus was generated, no frozen result was modified, no sealed
benchmark was inspected. Every claim below about stored data was verified by
reading stored values, not configuration files.

Predecessor experiment closed at `30cb2bb`; its frozen horizon result, bootstrap
analysis and matched-epoch sensitivity are treated as final and are not revisited.

---

# 1. Pinned implementations and component delta

## 1.1 Current implementation (pinned)

| item | value |
|---|---|
| repo commit | `30cb2bb`, branch `jepa-spatial-world-model-nav`, clean tree |
| predictor | `scripts/run_dev_v03_temporal_action_jepa_v1.py::Predictor`, width 384 / depth 6 / heads 6, **17,198,080 params** |
| predictor config | `run_dev_v03_two_step_rollout_v1.PRED = {"width":384,"depth":6,"heads":6}` |
| token grid | 24×32 = **768 tokens**, dim **1024** |
| visual encoder | V-JEPA 2.1 ViT-L/16-384 distilled from ViT-G, `~/.cache/vjepa2_1_vitl_dist_vitG_384.pt`, **frozen**, image path (`img_temporal_dim_size=1` → `patch_embed_img`) |
| preprocessing | v03 224×224 → centre-crop rows 28:196 (224×168, recovers the 78.323°×62.837° v04 FOV) → PIL bicubic isotropic 16/7 → 512×384 → ×255 → ImageNet mean/std. Hashed by `preprocessing_hash()` |
| temporal contract | context `t−480, t−240, t`; target `t+240`; 48 frames per timestep ⇒ each offset = **5 sim steps = 0.5 s**; action = the command block executed from `t` to `t+240` |
| corpus | `temporal_rows.jsonl`, 4,566 rows (**4,075 train / 491 checkpoint-selection**), sha256 `c2014ada5ca3f74e…` |
| horizon manifest | `FINAL_horizon_rows_479.jsonl`, 488 lines / 479 usable at H=4, sha256 `644a257803b5d49d…` |
| schedule | AdamW, lr 3e−4 constant (**no scheduler**), wd 0.01, grad-clip 1.0, batch 4, bf16 |
| render corpus | `render_textured_v03`, 1,450 scenes × 48 envs × 1,000 steps, 10 Hz |

## 1.2 Official reference (pinned) — **three relevant paths, not one**

Local checkout: `~/.cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812`
(commit **`204698b45b3712590f06245fbfba32d3be539812`**).

| path | what it is | relation to us |
|---|---|---|
| `app/vjepa/` | V-JEPA 2 pretraining | not used |
| `app/vjepa_2_1/` | V-JEPA 2.1 pretraining (`configs/train_2_1/…`) | **our frozen encoder's lineage** |
| `app/vjepa_droid/` + `src/models/ac_predictor.py` | V-JEPA 2-AC action-conditioned world model, config `configs/train/vitg16/droid-256px-8f.yaml` | **our predictor's lineage** |

> **Reported rather than silently chosen:** the official repo contains **no**
> configuration that combines a 2.1 encoder with the 2-AC action-conditioned
> predictor. `vjepa_droid` post-trains a **V-JEPA 2 ViT-g** encoder
> (`pretrain_checkpoint: vitg.pt`, `context_encoder_key: target_encoder`). Our
> stack — 2.1 ViT-L encoder + AC-style predictor — is a **hybrid that does not
> exist upstream**. Officialising the predictor does not make the combination
> official, and no upstream hyper-parameters are calibrated for it.

Official AC reference specifics (verified in source):

- Predictor: `VisionTransformerPredictorAC`, depth 24, embed 1024, heads 16,
  **frame-causal** attention, **RoPE**, `uniform_power: true`.
- Per frame the sequence is `[action_token, state_token, (extrinsics_token), *image_tokens]`;
  each conditioning signal is **one `nn.Linear` token**, not a modulation.
- Output takes image-token positions only; `predictor_norm` → `predictor_proj`.
- Loss: `loss_exp: 1.0` (L1), `normalize_reps: true` (`F.layer_norm`),
  `jloss = L1(z_tf, h)`, `sloss = L1(z_ar, h)`, `loss = jloss + sloss`, `auto_steps: 2`.
- Optimiser: warmup 15 / anneal 15 / 315 epochs, lr 7.5e−5 → 4.25e−4 → 0, wd 0.04 → 0.04, ipe 300.

### The single most important reference finding

`app/vjepa_droid/droid.py:222` — `actions = self.poses_to_diffs(states)`.
The DROID **action is the state delta**: `state[t+1] = state[t] ⊕ action[t]`
exactly (xyz addition, rotation composition, gripper delta). Therefore when
`train.py:432` feeds `states[:, :n+1]` into autoregressive rollout steps, it is
**not** injecting future information — future state is an invertible function of
the initial state and the action sequence already given to the model.

**This does not transfer to a quadruped.** For the Go2, joint angles, attitude
and body twist are *not* recoverable from the commanded velocity block; they are
a stochastic function of terrain and controller. Copying the official pattern of
feeding `states[:, :n+1]` at rollout would be **genuine future-proprioception
leakage**. This is the load-bearing reason §3 does not follow the reference here.

## 1.3 Component delta table

Classification: **A** intentional embodied-navigation adaptation · **B** likely
material omission or divergence · **C** implementation difference unlikely to
affect the research question · **D** unresolved, needs an explicit decision.

| # | component | official (`vjepa_droid` @ 204698b) | current (`30cb2bb`) | class | mechanism / cost if changed |
|---:|---|---|---|:--:|---|
| 1 | image preprocessing | 256px random-resized crop, aspect 0.75–1.35, scale 1.777 fixed, no flip | deterministic centre-crop → 512×384, no augmentation | **A** | FOV match to the deployment camera is the point of the crop. Adding scale/aspect jitter would break the frozen preprocessing hash and invalidate cached features. Keep. |
| 2 | temporal sampling | 8 frames @ 4 fps (0.25 s), random window per trajectory | 4 frames @ 2 Hz (0.5 s), fixed offsets `−480/−240/0/+240` | **A** | 0.5 s = exactly one command block; a shorter stride would decouple frames from the action unit. Keep. |
| 3 | context construction | **growing context**, 8 temporal slots, causal | **fixed sliding 3-frame window**, non-causal | **B** | Rollout beyond H=2 discards the oldest observed frame; by H=4 the context is entirely self-generated. Fixing this (grow the window to 8 slots + frame-causal mask) is the largest single architectural change and is the most likely cause of the H=4 collapse. Cost: predictor rewrite + ~2.7× sequence length. |
| 4 | target construction | all frames `1..T−1`, dense | single target frame `t+240`, dense | **B** | Official supervises every frame from every prefix in one pass; we supervise one. Adopting it multiplies supervision per sample at no extra encoder cost. |
| 5 | context/target encoder | same weights, `target_encoder` used for **both** (`context_encoder_key: target_encoder`); encoder in the optimiser but at `enc_lr_scale`; no EMA in the droid path | single frozen encoder, no gradient, no EMA (EMA exists in the earlier arm and was **rejected** by the encoder-movement experiment) | **A** | Encoder movement was tested and rejected (margin fell 8/8 scenes). Keep frozen. |
| 6 | stop-gradient | targets under `torch.no_grad()` | targets under `torch.no_grad()`; encoder frozen entirely | **C** | Equivalent. |
| 7 | positional embeddings | **RoPE** (spatial), frame-causal mask carries time; `uniform_power` | learned absolute `spatial (1,768,W)` + learned `temporal (4,1,W)` | **D** | RoPE generalises over positions and is what the pretrained predictor geometry expects; learned absolutes are cheap and already trained. Switching costs a full retrain and is only justified if we also adopt the growing context (#3), which needs position extrapolation. **Decide jointly with #3.** |
| 8 | predictor input norm | `predictor_embed` linear only; reps normalised by `F.layer_norm` after each predictor step | `input` linear; `normalise()` = `F.layer_norm` on encoder output and after each rollout step | **C** | Matches. |
| 9 | predictor output norm | `predictor_norm` (LayerNorm) → `predictor_proj` | `norm` (LayerNorm) → `output` | **C** | Matches. |
| 10 | action conditioning | **one action token per frame**, `nn.Linear(7→1024)`, prepended, attends through causal mask | **AdaLN-Zero over every block**, one global action vector for the whole sequence | **B** | AdaLN applies one action to *all* timesteps simultaneously; the official form gives each frame its own action, which is what a multi-step rollout needs. This is a direct candidate explanation for the modest action-sequence margin. Changing it is a moderate rewrite and is a prerequisite for #3/#4. |
| 11 | action representation | 7-D continuous state delta | 9-way primitive one-hot + nominal `(vx,vy,yaw_rate)` triple = 12-D | **B** | See §2.4: **40.0 % of command blocks are clipped**, so the nominal triple misstates what was executed. Using `executed_command_block` costs a rebuild of the row manifest but no re-render. |
| 12 | loss form | `mean(|z−h|^1)/1` — **L1, dense, all tokens** | `smooth_l1` on a 0.5-masked subset (default) or `l1_dense` | **B/C** | `l1_dense` mode already exists and was exercised in the supervision-only successor. Default should move to `l1_dense`. Low cost. |
| 13 | loss reduction | `jloss + sloss` over the official tensors ⇒ in our reduced 2-step setting `1.5·e1 + 0.5·e2` | implemented as the two elementwise means, not hardcoded | **C** | Already faithful; the attribution control showed `1.5·e1` alone is inert. |
| 14 | rollout unroll | `auto_steps=2`, feedback **not detached**, growing context | `auto_steps=2`, feedback not detached, **sliding** context | **B** | Same as #3. |
| 15 | teacher forcing | `z_tf` computed on true tokens for *all* frames in the same pass | separate first step; no all-frame teacher-forced pass | **B** | Consequence of #4. |
| 16 | recurrent state | none — context *is* the state | none | **C** | Matches. |
| 17 | optimiser | AdamW, warmup 15 ep → 4.25e−4 → anneal to 0, wd 0.04→0.04, `enc_lr_scale` | AdamW, **constant 3e−4, no warmup, no anneal**, wd 0.01 | **B** | No schedule is why the late-window curve oscillates and why "convergence" needed a bespoke rule (§5). Adding warmup+cosine is cheap and directly improves checkpoint determinism. |
| 18 | grad treatment | GradScaler / bf16, no explicit clip in the droid path | bf16, `clip_grad_norm_ = 1.0` | **C** | Clipping is defensive; harmless. |
| 19 | eval checkpoint / encoder variant | `e{epoch}.pt` every 25; eval uses `target_encoder` | epoch checkpoints, hash-verified at load; eval uses the same frozen 2.1 encoder as training | **C** | Ours is stricter (refuse-on-hash-mismatch). Keep. |
| 20 | proprioceptive state token | **present** — `state_encoder`, one token per frame, and fed at every rollout step | **absent** | **D → §3** | The subject of this transition. Must **not** be copied verbatim (see §1.2). |

**Do not copy** items 1, 2, 5 (the adaptations are deliberate and, for 5,
experimentally justified). **Highest-value changes, in expected-effect order:**
#10 (per-frame action tokens) → #3/#4 (growing context + all-frame targets) →
#11 (executed actions) → #17 (LR schedule) → #12 (dense L1) → #7 (RoPE, only
alongside #3).

---

# 2. Proprioceptive data availability (verified from stored values)

Verification basis: full scan of `messages.jsonl` for one episode
(`large_enclosed_maze_0294e4b379e1`, env 00, 356,248 records) plus per-topic
degeneracy checks across three families, and a full scan of the per-frame
manifest `frames.jsonl` (48,000 lines).

## 2.1 The decisive availability finding

**`plan/…/frames.jsonl` already carries per-frame proprioception, exactly
1:1 with every rendered image**, and is the very file the existing temporal
sequence builder already opens for episode-boundary checks. Verified: 48,000
lines = 48 envs × 1,000 steps, **all 17 keys present on every line, no gaps**,
present for **1,450 / 1,450 scenes** (matching the 1,450 rendered scenes).

Proprioception is logged at **exactly 10 Hz** (dt = 100.0 ms, min = max = median,
zero jitter, zero dropped samples, 1,000 samples per 100 s episode). One image per
proprioceptive sample. The predictor stride is 0.5 s, so **each context slot
subsumes exactly 5 proprioceptive samples**.

## 2.2 Signal table

Frame = body frame `base_link` unless stated. "Physical Go2" = available or
estimable onboard under the intended deployment contract.

| signal | source field | shape | units / frame | rate | vs image | missingness | physical Go2 | onboard vs privileged | verdict |
|---|---|---|---|---|---|---|---|---|---|
| joint positions | `frames.jsonl:joint_state.position` | (12,) | rad, joint | 10 Hz | **same timestamp** | 0/48,000 | yes (`LowState.motorState.q`) | **onboard** | **USE** |
| joint velocities | `joint_state.velocity` | (12,) | rad·s⁻¹ | 10 Hz | same timestamp | 0/48,000 | yes (`…motorState.dq`) | **onboard** | **USE** |
| joint effort/torque | `joint_state.effort` | **(0,)** | — | — | — | **empty on 48,000/48,000 and 1,000/1,000 raw** | yes on hardware | not recorded | **UNAVAILABLE** |
| body attitude, roll/pitch | `base_rpy_rad.roll/.pitch` (≡ `imu/data.orientation`) | (2,) | rad | 10 Hz | same timestamp | none | yes (IMU + gravity) | **onboard** | **USE as projected gravity** |
| body attitude, **yaw** | `base_rpy_rad.yaw`, `base_quat_world_xyzw` | (1,) | rad, **world** | 10 Hz | same timestamp | none | **no** — absolute heading is unobservable without magnetometer/SLAM | **privileged** | **EXCLUDE** |
| angular velocity (gyro) | `imu/data.angular_velocity` ≡ `base_state.twist_body.angular` | (3,) | rad·s⁻¹, body | 10 Hz | same timestamp | none | yes (gyro) | **onboard** | **USE** |
| linear acceleration | `imu/data.linear_acceleration` | (3,) | — | 10 Hz | same timestamp | **identically (0,0,0) on 1,000/1,000, all 3 families** | yes on hardware | not populated | **UNAVAILABLE** |
| body linear velocity | `frames.jsonl:twist_body.linear` | (3,) | m·s⁻¹, body | 10 Hz | same timestamp | 0/48,000 (max \|vx\| 0.57) | **estimated**, not measured — Go2 exposes a state-estimator velocity with drift/slip error | **simulator ground truth** | **USE WITH NOISE MODEL** (§3.2) — flagged **D** |
| previous motor commands | `command_context.{vx,vy,yaw_rate}` (requested) | (3,5) | m·s⁻¹, rad·s⁻¹, body | 2 Hz | block covering the step | 0/48,000 | yes | onboard | **USE** |
| **executed** commands | `messages.jsonl:/executed_command_block.executed_*` | (3,5) | as above | 2 Hz | block, stamped at block **end** | present for all blocks | yes | onboard | **USE — preferred (§2.4)** |
| foot contact flags | `/lewm/go2/foot_contacts.{fl,fr,rl,rr}_contact` | (4,) bool | — | 10 Hz | same timestamp | **False on 1,000/1,000 in all 3 families** | yes (foot force sensors) | logged but degenerate | **UNAVAILABLE** |
| foot contact forces | `…_force_n` | (4,) | N | 10 Hz | same timestamp | **0.0 on 1,000/1,000 in all 3 families** | yes | logged but degenerate | **UNAVAILABLE** |
| gait/mode flags | `/lewm/go2/mode` (`fallen`,`moving`,`standing`,`recovering`,`safety_stop`) | (5,) bool | — | 10 Hz | same timestamp | populated | partially (controller state) | onboard-ish | **OPTIONAL** |
| base world pose | `base_pose_world`, `/odom.pose` | (7,) | m, quat, **world** | 10 Hz | same timestamp | none (covariance all zero ⇒ not a real estimate) | no | **privileged** | **EXCLUDE** |
| camera extrinsics | `camera_pose_world` | — | world | 10 Hz | same timestamp | none | no | **privileged** | **EXCLUDE** (mount `camera_mount_body` is a fixed constant and is fine) |

Joint name order is `[FL,FR,RL,RR]_hip, [FL,FR,RL,RR]_thigh, [FL,FR,RL,RR]_calf` —
**grouped by joint type, not by leg**, which is *not* the Unitree SDK order. A
fixed permutation must be applied at the deployment boundary and asserted in a test.

## 2.3 Resets and discontinuities

`reset_event` fires **once per episode** (`reason: "initial_spawn"`,
`reset_count: 1`) in every episode inspected; `episode_step` runs 1..1000
contiguously. The existing sequence builder already refuses to cross an
`(env_index, episode_id, reset_count)` boundary and records
`crossed_resets: 0`. Proprioception therefore inherits a **clean, already-enforced**
reset contract — but §3.6 still requires the reset test, because a future corpus
with mid-episode resets would silently violate it.

## 2.4 Action fidelity — a material finding

Over 9,600 executed blocks in one scene: **40.0 % carry `clipped: true`, and in
every one of those the executed command differs from the requested command.**
Per tick: tick 0 deviates in 40.0 % of blocks, tick 1 in 2.4 %, ticks 2–4 in
**0.0 %**. Worst observed deviation 0.550 rad·s⁻¹ — a **sign reversal**
(requested +0.45, executed −0.10) while the controller decelerated out of the
previous command. `safety_overridden` was never set.

So the divergence is a **first-tick slew-rate limit**, systematic and largely
predictable from the previous command — but it means the current nominal action
triple misstates ~8 % of all commanded ticks, and misstates the *first* tick of
40 % of blocks. The first tick is exactly where the visual change begins.

## 2.5 Sufficiency verdict

> **The existing corpus IS sufficient for aligned proprioceptive training** of the
> recommended contract in §3, with **no regeneration**. Joint positions, joint
> velocities, body angular velocity, projected gravity, body linear velocity and
> both requested and executed commands are all present, frame-aligned, gap-free
> and cover all 1,450 scenes.

Not obtainable without regeneration, and therefore **excluded from the design**:

| would need regeneration | why | what would have to change |
|---|---|---|
| foot contacts / forces | logged but identically degenerate | rollout publisher must read the solver's contact state; requires re-running `rollout` (not render) for every scene |
| joint torques | `effort` never populated | same publisher change |
| IMU linear acceleration | identically zero | same publisher change |
| deployment-realistic velocity estimate | only ground-truth twist is stored | either a noise model at training time (§3.2, recommended) or a simulated state estimator in the rollout stage |

**Do not generate any of this now.** Contacts are the only one I would argue is
worth a future regeneration, because contact timing is the proprioceptive signal
most likely to carry terrain information the camera cannot see.

---

# 3. Proposed proprioceptive model contract

Preserved distinction, unchanged: an **RGB-derived visual representation**
(frozen V-JEPA 2.1 encoder, untouched) and an **action- and
proprioception-conditioned dynamics predictor**. Proprioception enters the
**predictor only**. The official implementation gives no warrant for touching the
visual encoder — `vjepa_droid` conditions the *predictor* on state and leaves the
encoder's input purely visual — and the encoder-movement experiment already
rejected encoder adaptation on this corpus.

## 3.1 Selected signals (26-D per proprioceptive sample)

| block | dims | content |
|---|---:|---|
| projected gravity | 3 | `R(roll,pitch)ᵀ · [0,0,−1]` — yaw-free by construction |
| body angular velocity | 3 | gyro, body frame |
| body linear velocity | 3 | estimator-style velocity (see 3.2) |
| joint positions | 12 | 12 joints, **Unitree-ordered** after permutation |
| joint velocities | 12 | as above |
| *(subtotal)* | *33* | |
| previous executed command | 3 | the `(vx,vy,yaw_rate)` actually executed at that tick |
| **total per sample** | **36** | |

Explicitly excluded: world pose, world yaw, camera extrinsics, contacts,
torques, linear acceleration.

Each 0.5 s context slot carries **5 samples × 36 = 180 raw dims**.

## 3.2 Normalisation and reset handling

- Per-channel z-scoring with mean/std computed **once, on the train split only**,
  frozen to a hashed JSON alongside the preprocessing hash. Never recomputed per run.
- Joint positions are z-scored **relative to the nominal stand pose**, not the
  raw angle, so the zero point is physically meaningful on hardware.
- Velocity realism: body linear velocity is simulator ground truth. Train with
  additive Gaussian noise σ = 0.05 m·s⁻¹ plus a per-episode constant bias
  σ_b = 0.03 m·s⁻¹, chosen to be conservative relative to a legged state
  estimator. **This is an assumption, flagged D** — it is not calibrated against
  a real Go2 log. Recorded as a limitation, not as a validated noise model.
- Resets: a sequence never crosses `(env_index, episode_id, reset_count)`; the
  builder already enforces this. Proprioceptive normalisation statistics are
  computed per-sample, not per-episode, so a reset cannot propagate through them.

## 3.3 Tokenisation and where it enters

Per context slot, the 5 samples × 36 dims are projected by a shared
`nn.Linear(36 → W)` and mean-pooled over the 5 sub-steps into **one
proprioceptive token per frame** (W = 384), matching the official one-token-per-
frame-per-modality convention. Embeddings added:

- **modality embedding** — a learned vector distinguishing proprio tokens from
  image tokens and from the action token;
- **temporal embedding** — the same learned per-slot temporal embedding the image
  tokens already use, so a proprio token is unambiguously bound to its frame;
- **no spatial embedding** — the token is not spatially localised.

Placement: **prepended per frame**, `[action_t, proprio_t, *image_tokens_t]`,
adopting delta item #10 (per-frame action tokens) at the same time. Proprio
tokens are inputs only; the output projection reads image-token positions only.

## 3.4 Target: visual-only

**The prediction target stays visual-only.** Adding a proprioceptive prediction
target is exactly the design that would confound the 2×2 (§3.5), so it is
deliberately excluded from this comparison and registered as the named follow-up.

## 3.5 Autoregressive rollout — the central design decision

At H≥2 the ground-truth future proprioception for a hypothetical action does not
exist and **must never be read**. Two candidates:

**Option A — observed history only, learned absence token.**
Proprio tokens are attached only to slots holding genuinely observed frames. When
a slot holds a *predicted* frame, a learned `proprio_unobserved` token takes its
place. With the current 3-slot sliding window the observed-proprio count is
3 at H=1, 2 at H=2, 1 at H=3, **0 at H=4**.

**Option B — predict a proprioceptive latent and feed it back.**
Add a proprio head; at rollout the predicted latent fills the slot, so the channel
survives to any horizon.

| | Option A | Option B |
|---|---|---|
| leakage risk | **structurally impossible** — no future proprio tensor is indexed | low but real: the head must be trained on future proprio, so the training loader touches it |
| interventions vs the RGB cell | **exactly one** (proprio conditioning) | **two** (conditioning **+** an auxiliary predictive loss) |
| horizon behaviour | signal decays; predicts Δ_prop shrinking with H | signal persists; risks compounding proprio error |
| cost | one linear + one learned token | head, extra loss term, extra weight to tune |

### Recommendation: **Option A** for the 2×2, with B registered as the follow-up.

The decisive argument is experimental, not architectural. Δ_interaction is a
difference of differences; if the proprio cells also carry an auxiliary loss the
RGB cells do not, then a non-zero interaction is attributable to either the
conditioning or the auxiliary objective, and the experiment cannot separate them.
Option A keeps **one** intervention between the RGB and proprio arms, which is
what makes the interaction interpretable.

Option A also yields a **falsifiable prediction**: because observed-proprio slots
fall 3→2→1→0 across H=1..4, Δ_prop should be largest at H=1–2 and vanish by H=4.
If Δ_prop instead persists at H=4, something is wrong — most likely leakage — and
the tests in §3.6 should catch it before that point.

Trade-off accepted: Option A cannot help long-horizon rollout, which is precisely
where the RGB-only stack was weakest. If Δ_prop is positive at H=2 and decays as
predicted, Option B becomes the justified next experiment.

## 3.6 No-leakage contract and the tests that enforce it

Contract, in one line: **the predictor may read proprioception with timestamp
≤ the timestamp of the newest observed image, and nothing else, ever.**

Tests that must **fail** if the corresponding fault is introduced:

| # | test | fault it catches | how it fails |
|---:|---|---|---|
| T1 | `test_no_future_proprio_index` — assert every proprio sample index used by a row is `≤ row.frame_index`, over the whole manifest | future-proprioception leakage | shuffle-independent assertion on indices; fails immediately if a slot reads `t+240` |
| T2 | `test_rollout_proprio_slots_are_absence_tokens` — run H=1..4 and assert the count of non-absence proprio tokens is exactly 3,2,1,0 | leakage introduced through the rollout path specifically | counts the tokens, not the values |
| T3 | `test_future_proprio_ablation_is_inert` — replace all *future* proprio in the loader with NaN and assert H≥2 outputs are bit-identical | any read of future proprio anywhere in the graph | NaN propagates → outputs differ → fail |
| T4 | `test_timestamp_alignment` — assert `frames.jsonl[i].timestamp_ns == image_stamp(i)` and `|Δt| == 100 ms` between consecutive samples for every row | timestamp misalignment / silent resampling | exact equality on stored stamps |
| T5 | `test_no_reset_crossing_proprio` — assert `(env_index, episode_id, reset_count)` is constant across all proprio samples of a row | reset contamination | tuple mismatch |
| T6 | `test_action_frame_offset` — assert the action block for a row starts at the `t` frame's `sequence_id` and its 5 ticks span exactly `t → t+240` | frame/action off-by-one | index arithmetic assertion |
| T7 | `test_executed_action_matches_block_end` — assert the executed block consumed for a row is the one **stamped at the end** of `[t, t+240]`, not the next one | the executed-block stamp offset (blocks are stamped at block end, requests at block start) | stamp comparison |
| T8 | `test_joint_permutation_is_unitree_order` — assert the permutation maps `frames.jsonl` order to SDK order and is an involution-free bijection | silent joint-order corruption at the deployment boundary | explicit expected permutation |
| T9 | `test_proprio_normalisation_is_frozen` — assert the stats hash matches the recorded hash | stats silently recomputed per run | hash mismatch |

T3 is the strongest of these and should be the gate: it detects leakage
regardless of *how* it was introduced.

---

# 4. The 2×2 comparison

## 4.1 Design

|  | one-step objective | two-step rollout objective |
|---|---|---|
| **officialised RGB** | `M_RGB,one-step` | `M_RGB,rollout` |
| **officialised RGB + proprio** | `M_prop,one-step` | `M_prop,rollout` |

- Δ_RGB = `M_RGB,rollout − M_RGB,one-step`
- Δ_prop = `M_prop,rollout − M_prop,one-step`
- **Δ_interaction = Δ_prop − Δ_RGB** — the test of whether proprioception makes
  rollout supervision *specifically* more effective.

"Officialised" means delta items **#10, #11, #12, #17** adopted in **all four**
cells (per-frame action tokens, executed actions, dense L1, warmup+cosine
schedule). Items #3/#4/#7 (growing context, all-frame targets, RoPE) are **not**
in this experiment — they are a larger change that would confound it, and are
registered separately.

## 4.2 Pairing and budget

- **Paired initialisation seeds**: seed *s* initialises the predictor identically
  in all four cells.
- **Paired data ordering**: the data-order generator is seeded from *s* and
  produces the same row order in all four cells; `dev_checkpoint_v1` already
  persists `data_order_generator_state`.
- **Identical compute budget** within a comparison: a fixed 24 epochs for every
  cell and every seed, no extensions (§5).
- One run per cell may be used for engineering validation only and **must not**
  be reported as the scientific comparison.

## 4.3 How many seeds

There is **no seed-variance estimate in the project** — every historical cell was
run once. The only dispersion available is *within-run late-window* oscillation
(epochs 15–23), measured here:

| arm | step-2 changed-cosine sd | occupied-IoU sd | occupied-IoU range |
|---|---:|---:|---:|
| rollout | 0.0037 | 0.0106 | 0.0329 |
| one-step | 0.0012 | 0.0134 | 0.0431 |
| attribution | 0.0023 | 0.0131 | 0.0381 |

This confirms the ~0.02 occupied-IoU oscillation and shows it is **~4× noisier
than step-2 cosine**, which is why the primary endpoint below is cosine, not IoU.
A 5-epoch windowed mean reduces the within-run sd to ≈0.0017 (rollout) / ≈0.0005
(one-step).

Because Δ_RGB and Δ_prop are each formed **within a seed**, shared seed variance
cancels; the relevant quantity is the sd of the paired difference, σ_d. Taking
σ_d ≈ 0.002 (the windowed within-run figure, used as a provisional stand-in),
detecting Δ_interaction = 0.005 at 80 % power, two-sided α = 0.05 needs
`n ≥ (2.8 · √2 · σ_d / 0.005)² ≈ 5`.

**Recommendation: 5 paired seeds**, with a **pre-registered interim re-estimate
after seed 3**: compute the realised σ_d and, if it exceeds 0.003, report the
achievable resolution honestly rather than adding seeds until significance
appears. Adding seeds after seeing the result is the same failure mode as the
extendable checkpoint rule (§5).

## 4.4 Endpoints — fidelity and discrimination kept separate

- **Primary:** correct-future changed-token cosine at **H=2**, corpus-weighted,
  the frozen estimator. One number, declared in advance.
- **Predeclared non-inferiority on discrimination:** the correct-minus-shuffled
  action margin at H=2 must satisfy
  `margin(cell) ≥ margin(RGB one-step) − 0.005`.
  A cell that wins on fidelity while failing this **does not qualify**; report
  both. The 0.005 bound is one within-run sd of the margin (0.0021–0.0037),
  chosen before any run.
- **H=3** is reported as **transfer beyond the trained horizon**.
- **H=4** is reported as **longer-horizon degradation**.
- **No post hoc H=2–3 combined endpoint** is formed, at any stage.
- Uncertainty: the same paired family-stratified episode-cluster bootstrap
  (228 clusters), corpus-weighted primary and equal-family robustness (§6).

## 4.5 Candidate rank regret — requirements for a future corpus

Still unavailable; not built. A future counterfactual branch corpus would need,
under the **final** observation and rendering contract:

1. From each branch state, **k ≥ 4 distinct action sequences actually executed**,
   each from the identical simulator state (checkpoint/restore, not replay).
2. Each branch **rendered** under the same textured contract, so branch frames
   are comparable to the training distribution.
3. A **ground-truth utility per branch** defined *before* collection (e.g.
   BFS-distance-to-goal reduction), fixed and hashed.
4. ≥ 200 branch states spanning all eight families, episode-clustered so the
   existing bootstrap applies unchanged.
5. Branch states drawn from **held-out scenes only**.
6. Realised outcomes generated **once** and shared by all models scored.

Cost driver is (2): rendering k branches per state multiplies the render bill by k.

---

# 5. Replacement checkpoint-qualification rule

The best-of-three ±0.005 rule is **not carried forward**. Its defect is
structural: it could be extended block by block until a window happened to
qualify, which is how the one-step control reached epoch 28 while the rollout arm
stopped at 22 — the unequal duration that then required a separate sensitivity
analysis.

## 5.1 Candidates, applied retrospectively to the completed curves

Fixed budget = 24 epochs (0..23), the only budget all three historical arms
completed. Primary metric = step-2 changed cosine. Window W = 5.

| rule | definition | rollout | one-step | attribution | deterministic? | extendable? |
|---|---|---:|---:|---:|:--:|:--:|
| **A. rolling-window mean** | centre epoch of the highest-mean 5-epoch window | **21** | **20** | **21** | yes | no |
| **B. fixed terminal window** | within the last 5 epochs, the epoch closest to the terminal-window mean | **21** | **21** | **21** | yes | no |
| **C. fixed-budget final** | always epoch 23 | 23 | 23 | 23 | yes | no |
| *(old rule)* | best-of-three ±0.005, extendable | *22* | *28* | *—* | no | **yes** |

Diagnostics: rollout terminal-5 mean 0.7208 (sd 0.0030); one-step 0.7100
(sd 0.0011); attribution 0.7105 (sd 0.0020).

What this exposes: **Rule B would have selected epoch 21 for *both* arms**,
making the historical comparison compute-matched by construction and removing the
entire need for the epoch-22 sensitivity analysis. Rule C would have selected the
rollout arm's single best epoch (23, cosine 0.7264) — attractive but fragile: it
inherits the full per-epoch oscillation (sd 0.0037), so it is the *least* central
of the three. Rule A tracks B closely but can select an early lucky window when a
run is still improving (it picked epoch 20 for one-step, mid-ascent).

## 5.2 Recommendation: **Rule B, fixed terminal window**

```
budget      : 24 epochs, fixed in advance, identical for all cells and seeds
metric      : H=2 correct-future changed cosine on the selection split
central stat: mean of epochs 19..23  (terminal window W=5)
checkpoint  : the epoch in 19..23 whose value is closest to that mean
              (ties -> lowest epoch number; fully deterministic)
variability : report sd over the window, and the OLS slope over epochs 14..23
flag        : if sd > 0.006 (2x the largest historical sd) OR slope < -0.002/epoch,
              the cell is marked UNSTABLE and reported as such -- it is NOT
              re-run, NOT extended, and NOT dropped
```

Why B over A and C: it measures central performance rather than an extremum, it
is computed at a budget fixed before the first run so no window can be searched
for, it produces one deterministic epoch, it applies identically to all four
cells and all paired seeds, and the separate variability check is *reported*
rather than used to trigger more training. The UNSTABLE flag deliberately has no
remedy attached — a rule whose failure branch is "train longer" is the rule we
are replacing.

**This retrospective application is for rule calibration only. It does not revise
the frozen selection, the frozen horizon result, or any previous conclusion.**

---

# 6. `local_composite_motifs` as a prospective diagnostic

Registered **now, before any run**, as a **family-level diagnostic — not a primary
endpoint**:

> In the completed RGB-only experiment, `local_composite_motifs` was the one
> family where the one-step control was separated from the rollout arm, and the
> advantage grew monotonically with horizon (Δcorrect +0.0041 → −0.0078 → −0.0170
> → −0.0313 at H=1..4; the H=4 interval excluded zero). It has the most episode
> clusters (39) of the eight. This was a **post hoc** observation and carries no
> prior weight.

Prospective handling:

- Reported for every cell and every seed at H=1..4, whatever the result.
- **No tuning to this family**: no family-specific weighting, sampling,
  augmentation, threshold or checkpoint choice.
- **Equal-family reporting is preserved alongside corpus weighting**, so a
  family-level regression cannot be hidden by the corpus weighting (this family
  carries 64/479 rows, so corpus weighting alone could mask it).
- Interpretation is declared in advance: if the horizon-dependent control
  advantage **replicates** under the officialised stack, it becomes a real
  phenomenon worth a dedicated experiment; if it does not, it is recorded as
  noise from a single completed run and dropped.

---

# 7. Required changes, in implementation order

| # | change | kind | depends on | risk |
|---:|---|---|---|---|
| 1 | Extend the row builder to attach 5×36 proprio samples per slot from `frames.jsonl`; write a new hashed manifest (no re-render) | data | — | low |
| 2 | Freeze train-split normalisation stats to hashed JSON | data | 1 | low |
| 3 | Switch the action source to `executed_command_block`, stamped at block end | data | 1 | **medium** — off-by-one risk, covered by T6/T7 |
| 4 | Tests T1–T9 | test | 1–3 | low — **must land before any training** |
| 5 | Per-frame action tokens replacing global AdaLN (delta #10) | code | — | medium — changes the predictor's parameter shape; no checkpoint is resumable across it |
| 6 | Proprio token path: `Linear(36→W)`, modality + temporal embedding, `proprio_unobserved` token | code | 5 | low |
| 7 | Dense L1 default (#12) and warmup+cosine schedule (#17) | code | — | low |
| 8 | Rule-B checkpoint selector as a shared library function used by all cells | code | 7 | low |
| 9 | Engineering-validation run, 1 seed × 4 cells, short budget, **not reported as science** | run | 1–8 | — |
| 10 | 5 paired seeds × 4 cells at the fixed 24-epoch budget | run | 9 | — |
| 11 | Bootstrap + equal-family reporting reusing `bootstrap_dev_v03_horizon_intervals_v1.py` | analysis | 10 | low |

Not in this sequence, registered separately: growing context + all-frame targets
+ RoPE (#3/#4/#7), proprioceptive prediction target (Option B), contact
regeneration, counterfactual branch corpus.

## 7.1 Compute estimate

Measured: **161 s/epoch** for the rollout arm (consecutive checkpoint mtimes),
bf16, batch 4, cached frozen features, single R9700.

| item | per run | ×5 seeds |
|---|---:|---:|
| RGB rollout, 24 ep | 64 min | 5.4 h |
| RGB one-step, 24 ep | ~40 min | 3.3 h |
| proprio rollout, 24 ep (+~5 % tokens) | ~67 min | 5.6 h |
| proprio one-step, 24 ep | ~42 min | 3.5 h |
| **training subtotal** | | **~17.8 h** |
| per-epoch selection eval (cached features, ~2 min × 24 ep × 4 cells) | | ~16 h |
| H=1–4 horizon eval, 20 checkpoints | | ~1.5 h |
| bootstrap (CPU) | | ~0.5 h |
| **total** | | **≈ 36 h GPU ≈ 1.5 days** |

Evaluation, not training, is the larger half. Reducing per-epoch evaluation to
every 2nd epoch outside the terminal window would cut ~7 h without affecting
Rule B, which only needs epochs 14–23.

## 7.2 Blockers and open decisions

**Blockers (no workaround inside this design):**

1. **No seed-variance estimate exists.** The n = 5 recommendation rests on a
   within-run proxy. The interim re-estimate after seed 3 is the mitigation; the
   honest statement is that the resolution of Δ_interaction is unknown until then.
2. **Contacts, torques and linear acceleration are unavailable** and cannot be
   recovered without re-running the rollout stage for every scene.
3. **Candidate rank regret remains unavailable** (§4.5).

**Open decisions requiring an explicit call (delta class D):**

4. **Body linear velocity** is simulator ground truth. Recommended: include with
   the §3.2 noise model, and record that the noise model is uncalibrated. The
   alternative — dropping it — removes the single most navigation-relevant
   proprioceptive channel.
5. **RoPE + growing context + all-frame targets (#3/#4/#7).** Recommended: not in
   this experiment. They are the largest expected win but would confound Δ_RGB.
6. **Whether the officialised RGB cell should be re-baselined against the frozen
   30cb2bb result.** Recommended: no. The officialised stack changes four
   components; comparing it to the frozen result would be a five-way confound.
   The 2×2 is internally controlled and needs no external baseline.

---

## Stopping condition

This report is the deliverable. Nothing was trained, no data was generated, no
frozen result was modified, and the sealed benchmark was not inspected.
