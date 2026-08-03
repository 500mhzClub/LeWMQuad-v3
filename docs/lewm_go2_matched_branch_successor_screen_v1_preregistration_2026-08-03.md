# Go2 matched-branch successor engineering screen V1

**Frozen:** 2026-08-03, before any successor-model feature extraction or
training result is opened.

## Purpose and claim boundary

This is a development-only capacity, action-binding, and runtime screen.  It
does not test fresh-scene generalization, multi-step imagination, closed-loop
navigation, or the thesis claim.  Its only decision is whether it is worth
collecting a fresh matched-state/multiple-action training campaign.

The screen separates three questions that earlier experiments mixed together:

1. whether frozen dense visual features are usable;
2. whether an action-conditioned transition can learn the nine successors of
   one state; and
3. whether a dense JEPA-style transition is more promising than conventional
   deterministic or compact RSSM-style dynamics.

A passed screen is not evidence that V-JEPA 2.1, DINOv2, JEPA, or an RSSM can
navigate.  Navigation evidence requires the later direct branch, rollout,
planning-regret, and closed-loop gates.

## Fixed input and custody boundary

Use only the reviewed `train` role of
`.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1`:

- manifest: 11,964 bytes,
  SHA-256 `87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e`;
- terminal: 1,250 bytes,
  SHA-256 `a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56`;
- independent terminal review: 2,844 bytes,
  SHA-256 `bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669`.

The role contains 128 states from 16 scenes, with three context RGB frames,
two historical requested actions, and all nine requested successor actions per
state.  It therefore supplies 1,152 matched branches and 1,536 bound RGB
artifacts.  The reviewed split-root loader and bound RGB reader must be used;
all RGB file and decoded-pixel identities remain verified.

The already-opened `eval` role may be loaded as metadata by the reviewed
bundle loader, but no `eval` RGB leaf or feature may be opened during this
screen.  It is development data, not a fresh confirmation set.

Candidate input is the requested action ID.  Future executed-command tapes,
endpoint state, physical rank, path length, progress, fall/tip fields, and
target RGB are targets or audit fields only.  None may enter a predictor input.
No sealed, held-out, or final-test material is eligible.

The corpus has no falls or tips, so this screen cannot train or evaluate a
nontrivial safety head.  It also has only one-step branch endpoints, so the
RSSM arm is a one-step capacity control, not a Dreamer rollout result.

## Frozen feature encoders

Both encoders remain frozen and run in evaluation/inference mode.

- **V-JEPA 2.1:** official Meta V-JEPA 2 repository commit
  `204698b45b3712590f06245fbfba32d3be539812`, official
  `vjepa2_1_vitb_dist_vitG_384.pt` ViT-B/16 checkpoint, 384-pixel evaluation
  preprocessing, image-mode input, and the final 24 by 24 grid of 768-channel
  tokens.  Area resampling produces the common 16 by 16 screen grid.  The
  execution authority must bind the downloaded checkpoint SHA-256 and exact
  byte count before extraction.
- **DINOv2:** official cached `facebookresearch/dinov2` `dinov2_vits14`
  checkpoint, ImageNet normalization at 224 pixels, and the final 16 by 16
  grid of 384-channel patch tokens.  The execution authority must bind its
  repository and checkpoint identities before extraction.

Per-token L2 normalization is applied after the fixed spatial conversion.
Feature caches contain only the train-role artifact IDs and float16 frozen
features; training promotes inputs and targets to float32.  Cache receipts bind
the source bundle, artifact order, encoder source, checkpoint, preprocessing,
tensor shape, dtype, byte count, and SHA-256.

The current official V-JEPA Hub source points its automatic checkpoint base
URL at localhost.  The runner may construct the exact official encoder from
the bound source commit and load the public README checkpoint directly.  A
small local `drop_path` compatibility shim is allowed because the available
ROCm environment cannot import the unrelated torchvision NMS extension.  The
shim must be unit-tested against the installed timm implementation formula and
does not change inference because drop probability is zero in evaluation.

## Four fixed model arms

All arms consume the same three context feature grids, two historical action
IDs, and candidate requested action.  All predict the frozen encoder feature
grid of the candidate successor.

1. `dense_vjepa2_1`: two-block dense spatial predictor over frozen V-JEPA 2.1
   tokens, with candidate and history actions conditioning every block.
2. `dense_dinov2`: the identical dense predictor mechanism over frozen DINOv2
   tokens.  This is the image-pretraining feature control.
3. `state_space_vjepa2_1`: a compact deterministic recurrent latent
   state-space transition over pooled V-JEPA 2.1 observations, followed by a
   lightweight positional decoder to the common token target.
4. `rssm_vjepa2_1`: a compact stochastic RSSM-style transition over pooled
   V-JEPA 2.1 observations, with a diagonal-Gaussian candidate-action prior,
   target-conditioned training posterior, and the same class of lightweight
   positional decoder.  It is not called full Dreamer: there is no actor,
   critic, reward optimization, or multi-step imagination in this screen.

The deterministic and RSSM controls use the V-JEPA representation so that the
mechanism comparison does not deliberately weaken their visual input.  Frozen
last-frame persistence and deterministic candidate-action derangement are
analytical controls, not additional trained arms.

Default hidden width is 128, action vocabulary is nine, and dense predictors
use two residual blocks.  Parameter counts and peak allocated GPU memory are
reported.  Source tests must establish shapes, finite gradients, action
sensitivity, and deterministic evaluation before any real feature cache is
opened.

## Fixed optimization and measurements

Use seed `2026080301`, AdamW, learning rate `3e-4`, weight decay `1e-4`, global
gradient-norm clipping at `1.0`, eight states per update, all nine actions for
each selected state, and exactly 800 updates.  State ordering is a seeded
permutation without family filtering.  No checkpoint or update selection is
allowed; report update 800, with update 0 and periodic traces diagnostic only.

For every arm, the primary training objective combines:

- mean matched per-token cosine distance to the correct successor; and
- within-state nine-way cross-entropy from the full predicted-action by
  true-successor cosine-distance matrix, with fixed temperature `0.1` and
  coefficient `0.25`.

The RSSM adds posterior reconstruction with coefficient `0.5` and mean
diagonal-Gaussian KL with coefficient `0.01`; inference and all primary screen
metrics use the action-conditioned prior mean, never the target-conditioned
posterior.

Report for every arm:

- matched per-token cosine error;
- the ratio of matched error to frozen last-frame persistence error;
- nine-way branch retrieval accuracy (each requested prediction retrieves its
  correct successor among the nine targets of the same state);
- requested-versus-one-step-cyclically-deranged action intervention margin,
  defined as deranged minus requested matched error, so positive is favorable;
- start/end loss, nonfinite count, parameter count, peak GPU allocation,
  feature extraction frames/second, training updates/second, and wall time.

All measurements are on the 128 training states.  They are training-set
capacity numbers and say nothing about generalization.

## Fixed screen decision

An arm is engineering-eligible only if all of the following hold at update
800:

1. source, input, cache, determinism, finite-value, and role-access checks pass;
2. matched-error / persistence-error is at most `0.80`;
3. nine-way branch retrieval accuracy is at least `0.50`; and
4. action intervention margin is strictly positive.

Fresh collection is justified only if at least one dense arm and both the
deterministic state-space and RSSM arms are engineering-eligible, no `eval` RGB
leaf was opened, and measured extraction/training throughput projects the
fixed later 12-member comparison to at most 24 GPU-hours.  If this conjunction
fails, stop and report the terminal screen result.  Do not tune widths,
coefficients, update count, thresholds, or seeds against the result.

If it passes, the next data campaign is fixed in scale but requires its own
reviewed source and authority: four independently checked 32-scene/256-state
shards, 1,024 states total, 9,216 branches, and 12,288 RGB frames.  Reserve 768
states from 96 scenes for training and 256 states from 32 disjoint scenes for
development evaluation, balanced as 12 train plus four evaluation scenes per
family.  Existing train/eval scenes are excluded.  The four-shard join and
role split must be tested and frozen before generation.

Measured predecessor cost is approximately 2 hours 6 minutes and 1.211 GB for
that campaign under the current receipt-rich storage pattern; use a three-hour
collection wall allowance.  This screen grants no collection, retry, rollout,
closed-loop, held-out, or promotion authority by itself.

