# Go2 fully learned JEPA navigation full handoff

Date: 2026-07-31

Status: **SPATIAL PERCEPTION QUALIFIED; TEMPORAL V1 VALIDLY REJECTED;
NAVIGATION, G2, AND HELD-OUT EVALUATION NOT YET REACHED**

## Executive summary

The repository has crossed one important boundary and hit the next one.

- A jointly trained RGB masked-spatial JEPA now demonstrably encodes rich,
  noncollapsed spatial and place-relevant features on scene-disjoint
  development data.
- The first temporal extension built directly on that checkpoint was a real
  joint JEPA: the online encoder, future predictor, action/time embeddings,
  and patchwise GRU were trained together to predict masked EMA features of
  the next RGB frame.
- Temporal V1 learned something real. Its loss fell, and correct ordered
  history/action began to beat reset, wrong-history, and wrong-action
  controls.
- It nevertheless failed at update 50 because its future prediction became
  too low-rank. Prediction effective rank was `4.0974` against an EMA-target
  rank of `22.5848`, or `18.14%`, below the preregistered `25%` floor.
- The GRU state and EMA target remained healthy. This localizes the current
  problem to preserving a rich future prediction, rather than showing that
  the spatial encoder lacks useful features or that the recurrent state is
  globally dead.
- The persistence baseline was still substantially better than the learned
  future prediction. Temporal V1 is therefore rejected and cannot be resumed
  or retried.
- No navigation, G2, held-out, or sealed evaluation occurred for these
  candidates or elsewhere in the 2026-07-24 through 2026-07-31 campaign.

The recommended next scientific step is one new rank-preserving temporal JEPA
successor, not more data refinement and not a longer run of V1. Retain the
qualified spatial predecessor, H6 schedule, GRU/history interface, and causal
controls; materially change the future prediction objective/decoder so that
it cannot reduce loss by concentrating variance into a few feature
directions. Scale only after an early capped observation preserves rank and
beats persistence as well as the causal controls.

## Repository snapshot

- Branch: `jepa-spatial-world-model-nav`
- HEAD before this handoff document:
  `f9830552eee085e1b15ce9742616b0a9ea6665b8`
- Starting handoff commit:
  `99370af8d895a8de30a7d6a3ea663e080b535af8`
- Governing held-out correction:
  `f3568880ecdda0d3f01ff8f661b19eb0753b58c9`
- Temporal V1 scientific-result commit:
  `b54d96df33529419a76c8ae2beb9518227c6458f`
- Temporal V1 independent-result-review commit:
  `f9830552eee085e1b15ce9742616b0a9ea6665b8`
- Worktree at handoff preparation: `33` modified and `543` untracked
  paths remain outside the committed handoff work. Do not use `git add -A` or
  assume those paths belong to this experiment.
- Active temporal training process: none; the one-shot terminated normally at
  its update-50 scientific gate.

## Critical authority correction

The original `99370af` handoff's statement that the V4 sealed role remained
unopened is superseded.

`docs/lewm_go2_heldout_maze_authority_correction_2026-07-24.md` records an
earlier manifest-access incident. V4 remains usable only as development
evidence and is permanently ineligible for G8. Every legacy sealed role is
inaccessible, and no eligible held-out role currently exists.

A replacement held-out role may be generated only after G3 through G7 pass
and the complete deployment source graph, checkpoint, calibration,
thresholds, evaluator, environment, and output schema are frozen. Its active
manifest must live outside the model-facing checkout under independent
operating-system custody. This handoff does not authorize creating or opening
that role.

## What counts as the repository goal

The goal is not a useful encoder attached to a classical navigator, nor a
predictor used only as a training loss.

The final claimed system must use deployment-available RGB,
odometry/IMU/proprioception, and executed-command history; maintain learned,
persistent, reversible physical and per-color beliefs; use an
action-conditioned JEPA predictor or rollout causally in deployed decisions;
and learn target, frontier/viewpoint, route/subgoal, and ordinary-motion
selection. Deterministic bookkeeping, enumeration, safety vetoes, and a fixed
low-level primitive executor may constrain learned decisions but may not make
the ordinary high-level choices.

A matched development arm with the JEPA objective disabled is required before
claiming that JEPA itself caused a generalization improvement.

## Current qualification state

| Layer | State | What is established | What is not established |
|---|---|---|---|
| Source/custody shell | Passed | Reviewed recursive source closure and one-shot execution machinery | No production/navigation qualification |
| RGB spatial JEPA | Passed on development | Rich masked-spatial tokens, spatial controls, raw health, and place substrate | Temporal prediction, memory, navigation, or held-out generalization |
| Temporal JEPA V1 | Rejected at update 50 | Real history/action learning with exact execution integrity | Rank-preserving future prediction or persistence advantage |
| Learned memory | Not qualified | GRU state stayed noncollapsed inside the rejected probe | Persistent/reversible deployed memory |
| Learned navigation | Not run | Existing reviewed shell may be reused | Learned target/frontier/route/action policy |
| G2 through G7 | Not attempted by this candidate | None | Perception, navigation, robustness, and deployment qualification |
| G8 held-out | Blocked | Custody requirements are defined | No eligible sealed role exists |

## Decisive spatial result

The RGB single-frame multiblock masked-spatial JEPA V1 was chosen to remove
temporal, action, object-space, physical-label, and navigation confounds. The
online encoder saw only `192/256` patch tokens before attention and jointly
trained with the predictor to predict `64` hidden EMA target tokens.

It passed at update 1,000 after exactly 16,000 logical presentations.

- Preregistration:
  `docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_preregistration_2026-07-31.md`
- Scientific result:
  `docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_scientific_result_2026-07-31.json`
- Result commit:
  `6e1ae4496f36a983935aa9f8a377948dffd9a23b`
- Independent audit commit:
  `b4f48110f6bf7422da6f8f7f434b17ceff410033`
- Selected checkpoint:
  `.generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1/attempt_v1/snapshots/update_1000.pt`
- Selected checkpoint SHA-256:
  `f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873`
- Place retrieval chance multiple: `4.383`
- Target place-key effective rank: `57.27`
- Raw online/target ranks: approximately `56/51`
- Maximum registered spatial-control primary ratio: `0.529`
- Wrong-target and wrong-context ratios: approximately `0.277`

This is the first qualified learned spatial-perception interface in the
campaign. It remains the valid predecessor. No later temporal checkpoint
supersedes it.

## Temporal V1 design

Temporal V1 was not a separately trained probe on a frozen encoder. It was one
joint JEPA initialized from the selected spatial checkpoint.

- Online RGB context: `rgb[0]`, `rgb[1]`, `rgb[2]`
- Model-visible actions: `actions[0]`, `actions[1]`, `actions[2]`
- Detached EMA future target: `rgb[3]`
- New memory: learned `9 x 192` action embedding, `3 x 192` time embedding,
  and one shared width-192 GRU, zero-initialized per sequence
- Future decoder: 256 recurrent patch tokens plus 64 masked future queries
- Trainable jointly: online encoder, spatial predictor, action/time
  embeddings, and GRU
- Sole loss: normalized half-squared masked future latent JEPA loss
- Excluded: RGB reconstruction, pose, geometry, physical labels, route,
  success, collision, or policy losses
- Frozen schedule: 4,000 unique H6 rows, ten rows per update, five `B=2`
  microbatches
- Maximum: 400 updates and 16,000 logical frame presentations
- Registered controls: persistence, current-only reset, wrong history, and
  wrong action

Source and authority chain:

- Preregistration: `1ac341cd97ab7a7d1a1b8c46695cf2fd3382ed60`
- Final implementation: `e61031ca1fcb5d149c919ff1c44ce0979410097e`
- Source-and-review freeze: `e61d447428d4e1f4e32749f6a7a3a609510c6d3c`
- Narrow clean-export certification: `af6e5e7699bc6392fbfcf9bdec56f16a1bdde653`
- One-shot authority: `17b162ebd37a15d757b304a52f2943a1e371fd23`

The metadata preflight passed before training. It parsed 18,048 metadata rows,
126,336 RGB path strings, and 108,288 action IDs; reconstructed the exact
4,000-row schedule, sentinel, donor maps, action coverage, and split
disjointness; and opened zero RGB leaves, checkpoints, navigation, held-out,
sealed, or GPU payloads.

## Temporal V1 result

Machine-readable result:
`docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_scientific_result_2026-07-31.json`

Independent audit:
`docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_scientific_result_independent_review_2026-07-31.json`

Terminal status:
`FAIL_SCIENTIFIC_CONTINUATION_GATE_NOT_MET`

The run stopped at update 50 after exactly:

- 500 unique sequence rows, equal to the first 500 rows of the frozen
  schedule;
- 2,000 logical RGB training presentations;
- 1,500 online and 500 EMA target frame encodings;
- 250 microbatch graphs/backward calls; and
- 50 clips, optimizer steps, and EMA steps.

All 50 training receipts passed. Loss moved in the right direction:

- update 1 loss: `0.636797`
- update 50 loss: `0.215415`
- first-five mean: `0.523376`
- last-five mean: `0.237617`
- minimum: `0.204572` at update 36

The causal controls also improved:

| Control ratio, lower is better | Update 0 | Update 50 | Update-50 evidence |
|---|---:|---:|---|
| Current-only reset | 0.98361 | 0.95736 | Correct history better in 8/8 families; bootstrap lower bound `0.00858` |
| Wrong history | 1.00186 | 0.96844 | Correct history better in 8/8 families; lower bound `0.00514` |
| Wrong action | 1.01055 | 0.99214 | Correct action better in 6/8 families; lower bound `0.00095` |
| Persistence | 4.57686 | 1.65884 | Gap shrank substantially, but persistence remained much better |

The mandatory representation-health gate correctly overrode those promising
trends.

- Prediction effective rank: `19.6967 -> 4.0974`
- Matching EMA target rank: `22.9730 -> 22.5848`
- Prediction/target rank at update 50: `18.14%`
- Required prediction/target rank: `>=25%`
- Required prediction rank: `5.6462`
- Shortfall: `1.5488`
- Prediction variance: `0.2290 -> 0.5020`
- Recurrent-state rank: `35.5034 -> 35.4862`

This was not constant-output collapse: variance increased. Instead, prediction
variance concentrated into very few directions. The recurrent memory and EMA
target remained broad, while the future predictor found a low-dimensional
shortcut that reduced loss.

The valid claim is narrow: this exact joint encoder/predictor/GRU future-JEPA
configuration is rejected. The result does not show that the spatial encoder
lacks useful features. It also cannot prove that the online encoder retained
all spatial qualities after 50 updates, because the registered full spatial
retention panel was scheduled for update 200 and the sentinel health gate
stopped the run earlier.

Execution integrity passed independently:

- exactly eight immutable output files, all mode `0444`;
- all seven JSON content hashes and all 50 trace-row hashes recomputed;
- request/open/success/decode counts all exactly `15,824`;
- zero denied or forbidden RGB positions;
- predecessor checkpoint opened/deserialized once and never reopened;
- no checkpoint or success file emitted;
- no retry or resume;
- no G2, navigation, held-out, or sealed access.

## Full experiment chronology since the 2026-07-24 handoff

The repository accumulated many version numbers, but many were source
closure, schema, visibility, immutable-publication, or failure-receipt fixes.
Science-identical integrity replacements below are grouped with their parent
mechanism and are not counted as new hypotheses.

### 1. Custody and reproducible execution shell

Commits `031568a` and `8cce85e` closed and certified the G2/runner source
graph. This made the reviewed execution shell reproducible. It did not turn
the current-frame/classical controller into the claimed fully learned stack
and did not authorize G2 or navigation.

Commit `f356888` corrected the held-out authority and permanently invalidated
V4 for G8.

### 2. Static RGB perception and simple temporal additions

- Multiresolution perception V3 completed 1,000 updates / 16,000
  presentations and improved all 54 tracked metrics, but passed `0` complete
  physical scopes. Rough pixel, ground, and depth-tail gates remained the
  limiting factors.
- Causal temporal fusion and ego-motion alignment each completed the same
  cap and landed almost exactly on static V3: `0` scopes and the same four
  failed gates. Post-encoder residual temporal additions were not sufficient.
- Science-valid overlapping-tokenization V2 lost a previously passing margin
  and worsened pixel/depth tails. Overlapping patches were not the missing
  mechanism.

### 3. Encoder JEPA, action residuals, and correspondence

- Staged encoder JEPA V3 learned current/next discrimination, but raw and
  projected rank collapsed and requested-action predictions were nearly the
  same as zero-action predictions.
- Whitening repaired rank. Action-gain, live-reference, action-energy,
  state-flow, inverse-dynamics, local/all-candidate correspondence, and dense
  cost-volume variants then tested whether action-conditioned structure could
  be extracted.
- State-dependent flow and the live-reference hinge produced genuine partial
  improvements, and dense correspondence showed the representation was not
  globally dead. None reliably ranked the executed action above the hardest
  wrong action or an action-free mean target.
- Valid next-target retrieval retained scene/next-frame identity but had
  below-chance action classification with HOLD dominance.
- Masked tubelet V13 was the justified longer run. Its update-100 metrics
  looked promising, but by update 400 future loss worsened, action NLL
  returned to approximately `log(9)`, macro action accuracy was only `0.1284`,
  and ordered-history/current controls failed. Longer training was a real
  negative result.

### 4. Direct BEV and geometry-anchored models

- Direct RGB-to-BEV models repeatedly demonstrated real perception learning
  while action prediction stayed near chance.
- Timing-corrected Direct-BEV V12 reached balanced accuracy `0.7097`, but
  occupied recall `0.5235` and NLL `0.4731` were still below the gate.
- Geometry-anchored deformable-BEV V3 reached semantic balanced accuracy
  `0.843`; generic succession and same-action retrieval worked, but action
  macro accuracy was `0.2114 < 0.2222` and the hardest wrong action won in all
  eight families.
- Rigid transport made action identity worse. Event-delta training failed a
  `70.9:1` semantic/dynamics gradient-balance gate. Action-query prediction
  again passed perception/target contrast but failed action NLL and
  hardest-wrong ordering.

These runs established a recurring split: RGB perception and generic
successor structure were learnable; fine state-specific action dynamics were
not yet usable.

### 5. Recurrent H4 models and the main-pool audit

- Recurrent H4 V1 produced the first strong all-family action signal, but its
  best H4 prediction was `2.28x` persistence and ordered history was harmful.
- Persistence-residual V2 found a correlated low-rank shortcut.
- Fixed-teacher V3 prevented that collapse and improved H4 to `1.44x`
  persistence, but history remained harmful.
- Dense cross-attention retained the same failure, weakening the idea that
  GRU compression alone was responsible.
- A four-component distributional predictor was the first to beat persistence
  broadly under a proper distribution score, but action dependence was weak
  and ordered history was still negative.
- Local-innovation, dual-domain, factual shared-transition, factorized
  increment, latent-momentum, causal-system-identification, and posterior
  expert variants traded better fit against action/HOLD/history failures.
  None combined prediction, action, ordered history, and HOLD in one passing
  learned state.

The main-pool audit then established approximately `2.896 TB`, `55.2M` frame
rows, and `1.81M` packed reset-safe H6 candidates spanning all families,
scenes, and actions. The capped schedule exposed only about `0.22%` of the
textured RGB allocation. This made data breadth an implausible primary
explanation for the structured failures. Scaling data remains appropriate
only after a mechanism passes a capped falsification.

### 6. Swept-progress survival and physical-evidence models

- Swept-progress survival V1 learned action ordering, RGB dependence,
  persistence advantage, and useful progress selection, but missed occupied
  recall.
- V2/V3 shifted the free/occupied tradeoff without satisfying both.
- A small residual local semantic decoder V4 passed all 24 development
  checks. This proved useful spatial information existed in the latent, but
  it was a decoder result, not evidence that the encoder or JEPA treatment was
  sufficient.
- The completed matched no-persistence treatment nearly passed its absolute
  panel, missing only free recall (`0.839420 < 0.85`). Full V4 minus that
  control had utility delta `-0.002868`, bootstrap lower bound `-0.017125`,
  and was positive in only `3/8` families. This is a narrow negative result
  for that persistence treatment, not evidence that JEPA is generally
  useless. The result is recorded in
  `docs/lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence_integrity_replacement_result_2026-07-28.md`.
- Its physical calibration failed: no threshold simultaneously achieved
  safe-free precision, nearby obstacle detection/exclusion, and useful-free
  recall.
- Hazard, fine-RGB fusion, hierarchical CNN, higher-resolution ViT, dense
  learned lift, projective cell volume, height-role factorization, and neutral
  ternary competition variants improved different sides of the tradeoff.
  None passed the conservative physical gate.
- Camera-evidence V13 through delayed temporal V17 improved physical margins
  and depth progressively, but completed zero full scopes. The justified V15
  extension still missed its feasibility gate; V16/V17 temporal ray
  consistency did not beat matched physical comparators.

### 7. Object-space causal predictors V18 through V28

- Height-volume V18 improved physical evidence but the correct prediction did
  not beat same-action wrong-scene RGB or an action-conditioned mean prior.
- V20 proved the successor objective was live but retained a six-of-twelve
  causal pattern.
- V21 lost simultaneous scene/action specificity; V22 recovered action
  specificity but became scene-independent.
- V23 reached 10/12 causal checks but lacked robust persistence advantage.
- V24 passed all 12 causal checks at update 400, then suffered prediction
  loss/gradient runaway by update 1,000 and failed the physical gate.
- Science-valid V26 stabilized scale, passed inherited and causal checks, and
  reached `102/112` physical margins, but completed zero scopes, retained
  excessive depth error, and never beat copy-current.
- Explicit-plan V27 was `5.32x` worse than persistence. A coherent terminal
  target in V28 improved this to `3.46x`, still failing temporal controls and
  rough depth.

These models repeatedly found action, scene, or mean-state shortcuts instead
of a useful temporal state.

### 8. Place/memory-role objectives

- Valid memory-role V2 passed the physical floor and all causal controls, but
  place R@5 was only `1.37x` chance, target rank was `1.41/64`, and local
  prediction was `17.2x` persistence.
- Spatial-grid V4 exposed strong untrained place structure, but cross-scene
  negatives were easy shortcuts: its objective improved the contrastive gap
  while harming within-scene retrieval and rank.
- Scene-local V5 removed most of that shortcut and retained R@5 `2.13x`
  chance in 7/8 scenes, but still reduced retrieval to `88.86%` of update zero
  and moved matching revisits farther apart.

The important conclusion was that useful single-frame place features already
existed and the attempted memory/place objectives damaged them.

### 9. Spatial-token delay-line memory

After adapter and numerical integrity replacements, the final valid delay-line
V4 observation reached update 250 with finite nonzero state. The state was
effectively about two dimensions out of 64, ordered history did not help, and
H4 prediction was catastrophically worse than persistence.

Falling loss was a low-rank shortcut, not useful memory. The delay-line
mechanism is closed.

### 10. Masked-spatial success and patch-memory temporal failure

The masked-spatial experiment finally isolated and passed the encoder
question. Temporal V1 then used that qualified interface and localized the
next failure: real causal learning occurred, but the future prediction lost
rank and still trailed persistence.

This is the current frontier.

## Operational failures versus scientific evidence

Do not count every versioned attempt as a scientific hypothesis. V13,
V15/V16, object-space V18/V19, V25/V27, memory V1/V3, and delay-line V1-V3
include zero-exposure, import, schema, GPU-visibility, alias, or receipt
failures. Their science-identical successors are plumbing recovery.

This final temporal round also had three non-scientific setup events:

- a file-path metadata-preflight invocation failed at Python import before the
  authority was loaded or the attempt root existed; the same frozen module
  then ran successfully with `python -m`;
- the first empty narrow-export directory was removed after a zsh special
  variable accidentally shadowed command search before any file was copied;
  and
- clean-checkout tests created 44 Python/pytest cache files; those exact cache
  files were removed and the remaining 44 authorized files were rehashed
  before launch.

None consumed the scientific attempt or changed model, data, seed, schedule,
loss, thresholds, initialization, or cap.

## Recommended next scientific step

Preregister one **rank-preserving future JEPA V2**, not a resume of V1.

Keep fixed:

- the selected spatial V1 predecessor;
- the three-context/one-future H6 interface;
- the corrected H6 train/validation roles and deterministic schedule;
- action/time-conditioned recurrent patch memory;
- persistence, current-only, wrong-history, and wrong-action controls;
- scene/family macro evaluation and anti-collapse gates; and
- perception-only development scope.

Materially change one mechanism: make the future prediction objective/decoder
rank-preserving. Plausible alternatives are loss in whitened EMA-target
coordinates, a dimension-balanced prediction loss, or an explicit prediction
variance/covariance floor. Preregister exactly one of these as V2 rather than
bundling them. Retain the joint encoder/predictor/memory JEPA loss and detached
EMA target. The aim is to prevent a few latent directions from carrying all
prediction variance without introducing RGB reconstruction or privileged
navigation labels.

Use an early capped falsification:

1. observe prediction/target rank no later than update 25 and update 50;
2. stop immediately if prediction rank again falls below the registered
   target-relative floor;
3. require continued improvement against current-only, wrong-history, and
   wrong-action controls;
4. require a credible route toward beating persistence, not merely falling
   training loss; and
5. scale beyond the early cap only if both representation health and causal
   controls pass.

If a rank-preserving successor still cannot approach persistence while the
recurrent state remains healthy, change the temporal decoder mechanism rather
than repeating optimizer/seed/schedule variants. Do not reopen V1 or train
from its state; it emitted no checkpoint.

Only after a temporal substrate passes should work proceed to persistent,
reversible learned memory and a learned target/frontier/route/action policy.
The predictor must remain causally present in the deployed decision path.

## Exact custody and continuation boundaries

- Temporal V1 attempt root:
  `.generated/go2_rgb_recurrent_patch_memory_temporal_jepa_v1/attempt_v1`
- Terminal failure file SHA-256:
  `9ff5c9884638721f945ba8e2bc61d9df6c55e74aa687157eaa4959cb81be292d`
- Terminal access file SHA-256:
  `72e36e3d40a4e46bd3d03a42958257cbc6d1650d40f32a7ea4566c4af1d55113`
- Trace SHA-256:
  `51e749cc093c3c8af8b45febae5f4f45b725e1903d4fd7bacecc20cbeeef7f77`
- Attempt consumed: yes
- Retry: unauthorized
- Resume: unauthorized
- Temporal checkpoint: none
- Navigation/G2/held-out/sealed access: none
- V4 sealed role: permanently invalid and inaccessible
- Eligible G8 role: none

The clean temporal source root remains at
`/home/andrewknowles/Workspace/LeWMQuad-v3-rgb-recurrent-patch-memory-temporal-jepa-v1-source`
as historical source evidence. Its authority grants no second run.

Do not open rejected checkpoints, legacy sealed material, or any V4 sealed
manifest. Do not stage the dirty worktree wholesale. A successor requires a
new preregistration, reviewed source freeze, clean-export certification, and
separate one-shot authority.
