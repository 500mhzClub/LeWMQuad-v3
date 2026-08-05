# G2 dataset v2 build and development runs

Date: 2026-07-09

Governing preregistration: `docs/lewm_go2_generalization_execution_contract_2026-07-09.md`,
section "G2 dataset-v2 preregistration". This document records execution of
that preregistration; it changes no threshold.

## Source selection (label-independent)

- source index:
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/source_index/go2_navigation_sources_09991d78f2e2b483a43b7157a0301987308f958b6a9570c99670b1fb60dfd6b9.jsonl`;
- 96 scenes, exactly 12 from each of eight rendered navigation/stress families
  (`large_enclosed_maze`, `local_composite_motifs`, `loop_alias_stress`,
  `medium_enclosed_maze`, `open_obstacle_field`, `rough_local_dynamics`,
  `small_enclosed_maze`, `visual_sensor_stress`);
- v4 development and sealed role commitments
  (`config/go2_generalization_v4/scene_role_commitments.json`) were enforced as
  exclusions both at indexing and again by the dataset builder before any
  scene-owned artifact was opened; zero forbidden overlaps.

## Role split

`deterministic_family_role_split` with split seed
`g2_geometry_v2_dev_v2_roles_20260709`: within each family, SHA-256 hash rank
over (seed, family, scene_id) assigns one checkpoint-selection, one
probability-calibration, and one untouched-G2 scene; the remaining nine train.
Roles are persisted in the dataset manifest (`scene_roles`) with per-role
scene-ID SHA-256 set commitments:

- assignments SHA-256:
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02`;
- role scene counts: 72 train / 8 checkpoint_selection /
  8 probability_calibration / 8 g2_evaluation;
- untouched-G2 set commitment:
  `0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402`.

## Window screening and selection

- 915,141 raw 0.5-second primitive windows screened;
- 202,490 (22.1%) pass geometry-v2 0.47 m configuration-space validity for all
  recorded poses and every adjacent segment (709,961 rejected on pose
  clearance, 2,690 on segment freedom) — consistent with the pilot audit's
  21.9% and a property of the mazes, not the labeler;
- hash-rank selection (seed `g2_geometry_v2_dev_v2_rows_20260709`) capped rows
  at 64 per scene before any label raycast; 16 scenes yielded fewer than 64
  valid windows (minimum 8), which the preregistered "at most 64" contract
  permits and the manifest records per scene
  (`allow_role_transition_shortfall` explicit);
- final dataset: 5,641 rows (4,262 train / 495 selection / 415 calibration /
  469 untouched G2).

## Labels

RGB visibility raycast through zero-inflation physical occupancy; FREE /
OCCUPIED / UNKNOWN targets from the 0.47 m body-inflated configuration space
(geometry v2 semantic SHA-256
`e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca`). This
fixes both v1 pilot defects (body-inflated occupancy misused as camera
occlusion; unvalidated windows).

## Adequacy audit (no G2 contact)

`scripts/audit_go2_paired_navigation_adequacy.py` — the untouched
`g2_evaluation` shards are never opened. All preregistered floors PASS:

| Role | FREE cells | OCCUPIED cells | UNKNOWN cells | nonempty next-observed rows |
| --- | ---: | ---: | ---: | ---: |
| train | 864,034 | 534,042 | 16,059,076 | 98.71% |
| checkpoint_selection | 98,585 | 60,863 | 1,868,072 | 98.18% |
| probability_calibration | 82,388 | 46,080 | 1,571,372 | 96.87% |

Floors: calibration ≥ 10,000 FREE (8.2x over), ≥ 1,000 OCCUPIED (46x over);
combined loaded-role nonempty next-observed fraction 98.51% ≥ 90%. Contrast
with pilot v1: calibration role [131,070 UNKNOWN, 0 FREE, 2 OCCUPIED] and 20%
row coverage.

## Artifacts

- dataset manifest:
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/dataset/dataset_manifest.json`,
  SHA-256 `e474fce5c6ca520728a94fdaada9edc7d86beb69387e14a9cd882e4240530b0c`;
- rows index SHA-256:
  `959c0a9920477931395af9acc77dff69881b5253435e3eb295541ffef048ba0a`;
- adequacy report:
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/adequacy_report.json`,
  SHA-256 `24a0a64aa2a3d69e447289c0de82ea8628c6330841008fdefe855fb66109920a`.

## Training protocol

Balanced hierarchical occupancy loss (equal-capacity UNKNOWN-vs-KNOWN and
FREE-vs-OCCUPIED-given-KNOWN), held-out vector calibration at natural class
priors, checkpoint selection on the selection role only, untouched G2 role
evaluated only by a final non-development run. Promotion calibration forbids
rare-class backfill (`allow_rare_class_backfill=False` outside
`--development-only`).

### Loss-contract audit: old 2/20/60 executions INVALIDATED

The original two-, 20-, and 60-epoch executions used inverse-frequency
three-class cross entropy. The preregistration requires two equal-capacity
binary terms: UNKNOWN-vs-KNOWN and FREE-vs-OCCUPIED given KNOWN. The model
implements these as mutually exclusive objective branches, so the old runs did
not execute the registered loss. Their numbers are retained only as debugging
history and cannot satisfy G2, support a training-vs-data verdict, or trigger
the 128-row rebuild. The old checkpoints lack resolved
`occupancy_training_objective` provenance and are forbidden for promotion.
The untouched G2 model outputs were not evaluated in those runs.

| invalid run | epochs | debug-only outcome |
| --- | ---: | --- |
| `dev_v2_smoke_v1` | 2 | completed data/calibration/serialization wiring |
| `dev_v2_candidate_v1` | 20 | 0/288 calibration threshold candidates passed |
| `dev_v2_candidate_v1_epoch60` | 60 | 0/288 threshold candidates passed; best epoch 23 |

### Strict deterministic attempt: FAILED CLOSED

The corrected two-epoch command was first run with `--deterministic` on the
ROCm GPU. It stopped during the first backward pass with
`RuntimeError: grid_sampler_2d_backward_cuda does not have a deterministic implementation`.
No checkpoint or report was created. This is an execution-reproducibility
result, not a model result; no G2 artifact was opened. Subsequent runs use a
fixed seed and row order, single-process loading, and an explicit report that
strict deterministic execution was not enabled.

### Corrected two-epoch hierarchical smoke: PASSED wiring

`.generated/go2_egomotion_bev_jepa/dev_v2_hierarchical_smoke_v1/` records the
required `hierarchical_equal_capacity_v1` objective with coefficients 0.5 and
0.5 and `three_class_weights: null`.

- train loss fell from 0.95338 to 0.72575;
- natural-prior vector calibration improved NLL from 0.32439 to 0.19119;
- the report verifies only the 88 train/selection/calibration scenes and
  records zero G2 rows, `final_g2_evaluation: null`, and
  `g2_evaluated: false`;
- the runtime refused the development checkpoint with `checkpoint is not
  promoted: g2_passes must be true`;
- checkpoint SHA-256 is
  `5eefe3ec1a75ac75fb37388ca5ad3dc73ed7376c326ade367d9579113771f345`;
  report SHA-256 is
  `ba91c45e94ea2290bb8686034937bdb9acaf8e496754504e202521bea010a4e6`.

An identical fixed-seed replay was not bitwise reproducible, as expected after
the strict-kernel failure: train loss ended at 0.71965 and calibration NLL at
0.19108. Its checkpoint/report SHA-256 values are respectively
`54a5b08b4dce138acc346875b6210b1abb7465413af99f489d0464f67a50a338`
and `824aabc32d1fb979218831de4d8ab5221ebd6af96acc8de5ce74985bc0059338`.
Both replicas agreed that the epoch-two offline gate failed with 8/20 checks
and zero passing threshold candidates. This is best-effort seeded
reproducibility, not a determinism pass.

### Corrected 20-epoch hierarchical candidate: COMPLETED / offline G2 bar NOT met

`.generated/go2_egomotion_bev_jepa/dev_v2_hierarchical_candidate_v1/`
completed with the registered objective and development-role-only provenance.
The run used commit `617d119172a6f49caf31a678e0fa7d05d5a3f4e9` plus dirty-diff SHA-256
`8c338632abd609e7358a95c81d4dce99b1eae8cfa46a5170ba5e48e7eb0ff4c3`;
the checkpoint embeds exact trainer/model source hashes and the resolved
command/configuration. Strict deterministic execution was not enabled.

- best checkpoint was epoch 15; train loss fell from 0.95338 to 0.54503 by
  epoch 20, while the last-five-epoch slope was only -0.00408 per epoch;
- held-out natural-prior vector calibration improved multiclass NLL from
  0.27295 to 0.13953, but **0/288** calibration-role threshold candidates
  passed;
- the selected calibration operating point reached planner-free precision
  0.7127, obstacle detection recall 0.6068, useful-free recall 0.3174,
  route success 0.4367, route-length recall 0.2409, and effective rank 3.355;
- the action-conditioned predictor beat warped persistence and zero-action,
  but did not beat shuffled action by the registered 0.10 margin;
- checkpoint SHA-256 is
  `385a2380fe4170f9fa89b94a493662662f335f790f432f9b3eba76f663cccd58`;
  report SHA-256 is
  `9b036c8dea74226dd9602590810a202a7b163b598711614a9f89dff57af4b97d`.

The leak-safe train-vs-selection diagnostic evaluated only train and
checkpoint-selection artifacts (80 shards and 8,701 RGB files verified) and
recorded zero G2 opens. Both roles failed 11/20 checks under the same frozen
runtime contract and had 0/288 passing role-local calibrated threshold
candidates. Train precision/obstacle recall/useful-free recall were
0.7595/0.6486/0.3555; selection was 0.7705/0.6420/0.2711. This is not a
selection-only generalization gap and does not justify a data-volume verdict.
Checkpoint calibration worsened known-cell free/occupied Brier and ECE on both
roles despite improving calibration-role multiclass NLL. The final four-view
counterfactual shows that raw selection logits have 114/288 safety-feasible
thresholds, but their best useful-free recall is only 0.00550; calibration
removes all of them. Raw train logits have 0/288 safety-feasible candidates;
their high-recall fallback reaches recall 0.8270 but only 0.4662 precision and
0.94991 obstacle recall. Calibration is harmful, but removing it does not make
the head functional. Final diagnostic SHA-256 is
`824d1cdfd597992966ae0e53250288e63e87d575d8c288f43a3196a3438aefff`.

The untouched G2 role remains unread and `final_g2_evaluation` is null. A
128-row rebuild is blocked: the present evidence cannot distinguish finite
data from capacity, objective, calibration transfer, or observability, and it
shows failure on the training role itself. The next bounded experiment is the
matched occupancy-only ceiling, with non-occupancy losses zeroed and checkpoint
selection ranked only by occupancy evidence.

### Matched occupancy-only interference ceiling: FAILED

`.generated/go2_egomotion_bev_jepa/dev_v2_hierarchical_occupancy_ceiling_v1/`
used the identical encoder, BEV decoder, hierarchical occupancy objective,
data, seed, and 20-epoch budget, but set JEPA, equivariance, action-contrast,
and variance loss weights to zero. The new development-only
`occupancy_ceiling_v1` selection contract ranks checkpoints only by occupancy
evidence, so untrained predictor checks cannot select the artifact.

- best epoch was 17, with 120/288 raw selection safety candidates;
- the best raw safety point reached precision 0.9920, obstacle recall 0.9530,
  and exclusion recall 0.9999, but useful-free recall was only **0.01770** and
  route success only 0.00649;
- the epoch-20 occupancy loss was 0.27237, effectively the same as the joint
  model's 0.27130, so auxiliary-loss interference is not the main blocker;
- vector calibration again destroyed safety feasibility: 0/288 candidates,
  precision 0.7101, obstacle recall 0.5864, useful-free recall 0.3059;
- raw train-role evaluation had 0/288 safety candidates. Its high-recall
  fallback reached useful-free recall 0.8368 but only 0.4592 precision and
  0.94790 obstacle recall. The supervised encoder/head therefore fails on its
  own training scenes, not only on selection;
- checkpoint/report/diagnostic SHA-256 values are respectively
  `bbd9b496219792c146571a5628257388a9bb24bcb6c9d1a6fb44c0a5080d9d17`,
  `32b4b45cd977483deac81e664d16c11c94ad31b234c94db51966fe00ba35f9f3`,
  and `c75751c618792a19dcfbef44b452f3f4d34dbd5048f07e1d6c2a179d73886d11`.

This blocks longer training, loss reweighting, and the 128-row rebuild as the
next move. The measured failure is the unconstrained RGB-to-BEV lift/head plus
misaligned post-hoc calibration. The next model ablation adds a fixed-camera
projective attention prior while preserving the JEPA/predictor interfaces; the
next calibration ablation fits monotone UNKNOWN/KNOWN and
FREE/OCCUPIED-given-KNOWN log-odds factors. G2 remains unread.

### V3 projective lift and hierarchical calibration smoke: PASSED wiring

The v3 development path preserves the encoder, BEV tensor, occupancy head,
predictor, warp, and EMA interfaces, but adds a fixed-camera projective prior to
token-to-BEV attention. It adds no trainable parameters. The smoke uses nominal
simulation camera geometry and therefore is not hardware-promotable. It also
replaces independent three-class vector scaling with two positive-affine
hierarchical log-odds calibrators and reconstructs a normalized
UNKNOWN/FREE/OCCUPIED simplex.

`.generated/go2_egomotion_bev_jepa/dev_v3_projective_occupancy_smoke_v1/`
completed two epochs with zero G2 contact:

- epoch-one loss/precision/useful-free recall were 0.84997/0.4130/0.6367,
  versus 0.92461/0.3371/0.3837 for the matched legacy occupancy ceiling;
- epoch-two loss was 0.58853, with precision 0.3736, useful-free recall 0.7698,
  and obstacle recall 0.9201. This is an early optimization gain, not a gate
  pass;
- hierarchical calibration consumed all 1,699,840 valid calibration-role
  cells with natural priors, no weighting, subsampling, balancing, or
  backfill. Joint NLL improved 0.27087 -> 0.13791; UNKNOWN/KNOWN NLL improved
  0.24190 -> 0.11022; conditional FREE/OCCUPIED NLL improved
  0.38337 -> 0.36642;
- the calibrated two-epoch map still had 0/288 threshold candidates and failed
  offline. Occupied-detection threshold selection remains a registered v3
  follow-up rather than being changed post hoc;
- checkpoint/report SHA-256 values are
  `78ab0d333b61bf50686b9bc24925310c02b946cd43b64cedee768fb3735c00ef`
  and `025f97ee98d2535e4e6aaa950190ff34d7472f08a9184ad397a60c1d1cf88af4`.

### Retrospective camera/observability invalidation: DATASET V2 NOT PROMOTABLE

The matched 20-epoch projective occupancy candidate was **not launched**. A
first-principles audit before that run found two upstream contract failures, so
the two-epoch projective result remains wiring/debugging evidence only.

First, v03 generated square 224x224 Genesis RGB and passed the declared
horizontal FOV to Genesis' vertical-FOV API. Those source images have
horizontal=vertical=78.323 degrees, whereas the platform 640x480 pinhole is
horizontal=78.323 and vertical=62.8370386364 degrees. A centered 75% vertical
crop restores the intended view without discarding horizontal information. The
camera audit opened render plans and summaries, not G2 images:

- artifact:
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/source_camera_contract.json`;
- artifact SHA-256:
  `f8f24d43768c2d5ddbccb85b91d26a3de790fed27d5b5d34d803f08824b6c80d`;
- content SHA-256:
  `1d55b8a7ad7169f0790f793d72bc57625b411cf5e1b6fbcf9c4e87e7d46fc6d6`;
- 96 source scenes, `g2_images_opened: false`.

The dataset-bound v1 record cannot be an input to its own replacement without
a circular hash. A source-index-bound v2 contract was therefore emitted before
the corrected build. It has the same 96-scene/render-summary evidence and zero
G2 image contact, file SHA-256
`b31fd8afdf1f4ec05589677d8c39b90521769501cb7f9e1c161fc5ca779a54e4`
and content SHA-256
`06013aaf471e83b8da3ca3806a7072c73050764d9ae46e3a80564ee7c21bc4ea`.

Second, v2 assigns configuration-space labels using global 0.47 m inflated
geometry after raycasting only the physical line of sight to each cell center.
This leaks hidden footprint occupancy into the target: on checkpoint-selection
scenes, 33,665/98,585 FREE cells (34.15%) and 29,880/60,863 OCCUPIED cells
(49.09%) have footprint support outside the horizontal frustum. Two worlds can
therefore produce identical RGB but different labels. The frozen 99% precision
and 90% useful-recall target cannot be reached from the registered input.

Third, the camera-only correction was itself insufficient: v03 did not render
`visual_randomization.distractor_objects`, although geometry-v2 treats them as
real collision objects. It also could not bind full roll/pitch/yaw geometry for
rough-dynamics ramps. Cropping the old RGB was therefore abandoned before a
new dataset or longer model run.

The replacement preserves the exact label-independent 96-scene set, 5,641 row
identities, and 72/8/8/8 family roles, but sparsely rerenders only the 10,311
selected endpoint frames. V04 renders 224x168 RGB at H=78.323 and
V=62.8370386364 with no crop, includes walls/obstacles/landmarks/distractors,
uses full object RPY, and hashes every selected image and source contract:

- render plan file/content SHA-256:
  `d93b17d45dd51f7bad4c442e8d434105997c2be4198f86dc498ded955c56a34c` /
  `1fbedb84ca584e1ffba7cfa1ae22e4e379deb10d5b85ca7f7cb2dda1a369f7e3`;
- corrected source-index SHA-256:
  `11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c`;
- combined render-audit file/content SHA-256:
  `9a045dff82fb82adbbb89d10cb4dc0063297805038b000e5f6cd53816e995a9a` /
  `c9280ed4cab9ff54f7d8684835b8448886209a8cc50eba3588519c34572a6358`;
- audited totals: 96 scenes, 10,311 frames, 4,087 rendered object instances,
  roles 72/8/8/8. The audit hashes G2 image bytes for integrity but never
  decodes/inspects them; it opens no G2 label shard or model output.

The v3 learned target is now `observable_physical_occupancy_v3`, not per-frame
configuration space. Physical evidence is fused across views in
`OnlineBeliefMap`; only then does the deterministic 0.47 m morphology produce
planner FREE/OCCUPIED/UNKNOWN. The next admissible model run is a two-epoch
development-only center-projective physical-occupancy smoke under
`physical_occupancy_ceiling_v1`. Dataset v2 and all earlier candidates are
forbidden for G2 promotion; untouched-G2 model outputs remain unread.

### Observable-physical dataset v3: BUILT / ADEQUACY PASSED

The corrected dataset was built with six bounded scene processes. Workers
computed scene-local labels only; the parent retained sorted scene order,
global-row assignment, shard/index writes, and all manifest aggregation. A
serial-versus-parallel regression test proves identical logical rows, arrays,
shard hashes, and normalized manifests. A representative six-scene corrected-v04
benchmark improved from 33.544 seconds serial to 9.309 seconds with six workers.

- manifest:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json`;
- manifest SHA-256:
  `ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180`;
- rows SHA-256:
  `187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac`;
- image-set SHA-256:
  `49d15b8a68b1bc2724767a19535f4585af80bcba345e876fbe4e6a33515e9227`;
- 96 scenes and 5,641 rows; roles 72/8/8/8 and row counts
  4,262/495/415/469;
- assignment SHA-256:
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02`;
- untouched-G2 scene-set commitment:
  `0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402`;
- transition identity SHA-256 after removing only row schema/label/shard/image
  representation fields:
  `51aca982da3c5a5e86ffaa959d10e6a5354d781a84824bae0e3076097fcf93d5`;
- sorted scene-set SHA-256:
  `2ab65f3511b7b2405ea0c3df062077771582c1c1b045c98ca2477b6226d7aa5d`.

The manifest records `observable_physical_occupancy_v3`, corrected 224x168
H=78.323/V=62.8370386364 projection, full-RPY first-hit evidence, conservative
0.05-to-0.10 m aggregation, collision-geometry FREE veto, and evaluation-only
post-memory morphology with radius 0.47 m. Its embedded build record binds the
resolved command/defaults, six-worker count, source code, v04 source index,
render audit, role exclusions, geometry, environment, git commit, and dirty diff.

The no-G2 adequacy artifact is
`.generated/go2_paired_navigation/geometry_v3_physical_v1/adequacy_report.json`,
SHA-256
`6fa6a667af2729ea1cf717a19997777a4fa227cd8e01551d21f2fb42d2e00e4d`.
All frozen floors pass:

| Role | FREE cells | OCCUPIED cells | UNKNOWN cells | nonempty next-observed rows |
| --- | ---: | ---: | ---: | ---: |
| train | 1,068,108 | 130,458 | 16,258,586 | 97.84% |
| checkpoint_selection | 116,208 | 14,255 | 1,897,057 | 98.18% |
| probability_calibration | 100,403 | 11,219 | 1,588,218 | 96.39% |

The audit opened 88 development-role shards and zero untouched-G2 shards. A
role-scoped integrity pass additionally verified 9,460 development RGB files,
88 scene manifests/plans/frame logs/render summaries/shards, the five build
inputs, and the global geometry/render/exclusion commitments. The remaining
eight G2 shards/images and every G2 model output remain unopened.

### Physical center-projective two-epoch smoke: PASSED WIRING

The registered occupancy-only smoke ran on the corrected v3 target with the
parameter-neutral `projective_column_attention_v1` lift, hierarchical
equal-capacity occupancy loss, natural-prior hierarchical calibration, and
`physical_occupancy_ceiling_v1` selection. JEPA/equivariance/action/variance
weights were zero for this bounded head-ceiling smoke only.

- train loss fell from 0.80392 to 0.64240;
- raw checkpoint-selection FREE recall / FREE precision / <=2 m obstacle recall
  were 0.8485 / 0.4077 / 0.6334 at epoch one and
  0.7621 / 0.4178 / 0.7984 at epoch two;
- epoch one was retained by the preregistered physical selection tuple because
  neither early epoch had a passing safety point and epoch one had higher useful
  FREE recall;
- hierarchical calibration used all 1,699,840 calibration-role cells with no
  weighting, subsampling, duplication, or backfill. Joint NLL improved
  0.28623 -> 0.12893, UNKNOWN/KNOWN NLL 0.26705 -> 0.11563, and conditional
  FREE/OCCUPIED NLL 0.29220 -> 0.20249;
- calibrated threshold selection still had 0/288 passing points. This exposed
  the already-registered implementation gap that obstacle detection remained
  fixed at joint OCCUPIED posterior 0.5 instead of selecting its operating
  threshold on the calibration role. The resulting diagnostic obstacle recall
  was 0.000735; no longer run is licensed until that selector is implemented;
- checkpoint/report SHA-256 values are
  `12179fd35d5c5fdf3ac99f7a43b2c1907ec872cbe75a10234f476617df83edc9` /
  `55ccd652f527188990035430aa729dcd1e49e1cf3ff03c3ec53e369e9d7d5c26`;
- checkpoint/report schemas are v4, target space is
  `observable_physical_occupancy`, `runtime_ready` is false, legacy
  `g2_passes` is false, and the v2 runtime rejects the artifact;
- exact train/selection/calibration row-subset SHA-256 values are
  `b4865e81b6954d674ad9cc4087250802e8f108fb06260f6d41af6c848bb9e97d`,
  `cccc0855d9fba509c36a3ed1d68615334dee2463b61adde02a179944b904466b`,
  and `850b2358abb57456e94c3d1f70b0082fceb77be6a0eeef8957bdf97fad8d0066`;
- the embedded access ledger records all 469 G2 row metadata entries as known,
  but zero G2 shard/image byte opens and zero G2 model-output rows. No
  `final_head_g2_evaluation` exists.

Two preceding launch attempts ran inside a filesystem sandbox without GPU
device nodes and stopped before model construction. They wrote no checkpoint or
report and are execution-environment failures, not model results. The completed
run used the explicitly approved ROCm device scope.

### Physical occupied-detection selector: REPAIRED BEFORE LONG RUN

The smoke confirmed the previously registered selector gap rather than creating
a new post-hoc metric. Physical target-space threshold selection now searches
occupied-detection posterior thresholds
`[0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]` on the probability-calibration
role. Legacy configuration-space evaluation remains fixed at 0.50. Detection
recall is recomputed for each candidate without repeating invariant validation
or calibration metrics; selected admission and detection ranges must be
disjoint, and ties choose the highest equally effective detection threshold.

The exact selector has 2,016 evaluated tuples. An integrated
calibrated-output fixture injects an OCCUPIED posterior of 0.025 and proves that
the evaluator selects 0.02 and reaches recall 1.0 for the physical target while
the legacy 0.50 path remains unchanged at recall 0; calibration fitting itself
is covered separately.
The full `lewm/tests` regression suite passes 581 tests plus 3 subtests.
Post-fix trainer and metrics source SHA-256 values are
`8b37bf69c5f9c262e30200391bc4a1bb59aaa14e9bfe189621af0da2f47f104b` /
`97be0acb1a9cf6e170db90945c908a1a30b2ce0a230a5664024b8c06edd03396`.

The two-epoch checkpoint embeds the pre-fix trainer/metrics hashes and is not
silently reinterpreted. The separate development-only rescore report below
binds the parent checkpoint/report, exact saved row subsets, frozen
model/calibration, and current selector code. It is diagnostic only and cannot
become the one-shot G2 or runtime artifact.

### Frozen two-epoch selector rescore: COMPLETED / UNDERTRAINED

The development-only rescore is
`.generated/go2_egomotion_bev_jepa/dev_v4_physical_projective_smoke_v1/threshold_rescore_v1.json`,
with file SHA-256
`768a3ba2e4fb5a105c71a6b2237df42cfdd99dfeadda6b95d3cf6d961632ae9d`
and canonical content SHA-256
`81959f1c4d7bd74ab6c95c8364e415ab0ce66eeefd06994a588c475c5c6c2932`.
The helper source SHA-256 is
`34e4a93ff1eb2c843b5d4a93f173869684608b0376845646888142e07f9cd090`.

The rescore verified the parent checkpoint/report hashes, the exact saved
selection and calibration subsets, and unchanged model-state and calibration
identities. It fitted no calibration, performed no training, and emitted or
mutated no checkpoint. Its access ledger records all 469 G2 row metadata
entries but zero G2 label-shard/image-byte opens and zero G2 model-output rows.
The frozen model-state SHA-256 is
`ac7f9ad1fe6e05a2839a38f16c79f9e65f61bdbdaf5106a01ff9e3d17985a8b2`;
the calibration canonical/content SHA-256 values are
`2975a32c1d8e7189cd92ff3dd3f92f83617ffd9064526e7709f64ad462bc6687` /
`2a0c2f986ed35cbcf3fa144cca188aa6c22eab92eb9b3356efa825de68782699`.
The bound selection/calibration subsets contain 495/415 rows with identities
`cccc0855d9fba509c36a3ed1d68615334dee2463b61adde02a179944b904466b` /
`850b2358abb57456e94c3d1f70b0082fceb77be6a0eeef8957bdf97fad8d0066`.

The corrected selector evaluated all 2,016 tuples, found zero passing tuples,
and retained the compatible fallback
`free_probability_min=0.50`, `unknown_probability_max=0.35`,
`occupied_probability_max=0.35`, `occupied_detection_min=0.50`.
On calibration, admitted-FREE precision / useful-FREE recall / directly
observable <=2 m OCCUPIED recall were 0.73667 / 0.11034 / 0.000735; on
checkpoint selection they were 0.68800 / 0.16497 / 0.000857. Lower detection
thresholds are available, but with this undertrained posterior they require a
stricter disjoint FREE-admission boundary whose recall/precision tuple loses to
the registered fallback ordering. Therefore the unchanged 0.50 result is
coherent, not evidence that the selector failed to search.

This closes the bounded smoke/selector wiring gate and licenses the already
registered matched 20-epoch development candidate. It does not pass G2 and it
does not license an untouched-G2 evaluation.

### Matched 20-epoch physical candidate: COMPLETED / G2 HEAD NOT MET

The preregistered development-only candidate is
`.generated/go2_egomotion_bev_jepa/dev_v4_physical_projective_candidate_v1/`.
Checkpoint/report SHA-256 values are
`c56fe958a841d8e7f89b4df0cecf63546abeae1a6a40044c4a1499f0486f0256` /
`f40df099087d5e37e14366713b766963573e5638ecc9412ddfedfde90ad65fe6`.
It used the smoke configuration unchanged except for 20 epochs and output
paths. The checkpoint retained epoch 10 by the registered physical selection
tuple; train loss fell 0.80392 -> 0.31906 through epoch 20.

Natural-prior calibration used all 1,699,840 calibration cells and improved
joint NLL 0.23554 -> 0.10231, UNKNOWN/KNOWN NLL 0.22243 -> 0.09273, and
conditional FREE/OCCUPIED NLL 0.19965 -> 0.14588. The selected thresholds were
`free_min=0.50`, `unknown_max=0.05`, `occupied_max=0.01`, and
`occupied_detection_min=0.02`. Thirty-three threshold tuples met the two
safety conditions, but no tuple met the complete 99/95/90 head gate. At the
selected calibration operating point, admitted-FREE precision was 0.99209 and
directly observable <=2 m OCCUPIED recall was 0.97703, while useful-FREE recall
was only 0.01998. The head can be made safe only by admitting almost no free
space.

The report is schema v4, records `head_g2_passes=false`, `runtime_ready=false`,
and has null final head/legacy G2 evaluations. Its access ledger records 469 G2
metadata rows but zero G2 label-shard/image-byte opens and zero G2 model-output
rows. Exact train/selection/calibration subset identities remain
`b4865e81b6954d674ad9cc4087250802e8f108fb06260f6d41af6c848bb9e97d`,
`cccc0855d9fba509c36a3ed1d68615334dee2463b61adde02a179944b904466b`,
and `850b2358abb57456e94c3d1f70b0082fceb77be6a0eeef8957bdf97fad8d0066`.

The frozen train-vs-selection diagnostic is
`.generated/go2_egomotion_bev_jepa/dev_v4_physical_projective_candidate_v1/train_selection_diagnostic_v1.json`,
file SHA-256
`24bfa8a38f1769d5fed2eeb435909acd8bb9ec5e543b9b97d737ed693c671bb1`.
It verified 8,701 train/selection images and 80 role-local shards, opened no G2
image/shard bytes, and evaluated no G2 output. Its bounded read is
`train_role_physical_head_failure_blocks_generalization_attribution`: the
failure is already present on training scenes, so a held-out generalization
claim is premature. On train, frozen calibrated precision / <=2 m OCCUPIED
recall / useful-FREE recall were 0.98944 / 0.97534 / 0.02394. Even a role-local
safe threshold reached only 0.02174 useful-FREE recall; the raw train head had
zero safety-admissible tuples. Selection shows the same failure class.

This rejects a threshold-transfer-only or scene-generalization-only diagnosis.
It does not distinguish incomplete optimization from representation, lift,
capacity, or remaining observability bias. The next bounded action is a frozen
spatial-grounding diagnostic: visible-interior versus FOV-boundary rings,
correct versus shuffled/mean RGB, and alignment perturbations. No further
training or G2 read is licensed until that result selects a falsifiable model
change.

### Frozen spatial-grounding diagnostic: COMPLETED / GROUNDED BUT COARSE

The source-bound development artifact is
`.generated/go2_egomotion_bev_jepa/dev_v4_physical_projective_candidate_v1/spatial_grounding_diagnostic_v1.json`.
Its file/content SHA-256 values are
`f8fc7c529197b3ba08574cba409695f564f54401d487c8eb48c1aa9cfdb4e3da` /
`bff482f9036dc1549eedc676ca9944205c28c0b4581f03cea16f4a26f2dc817e`;
the content hash and all nested support/geometry hashes recompute. The
diagnostic helper/script/test source SHA-256 values are
`c64e12242959b4eee22a29fa52538090997385396f3413974c31102a46ae418d`,
`d3d2de385d9ffb05f0188078f800d8a2a2f8108e9c351e396b3720f2cd6306d2`,
and `0b87177c6bea53d357e0484cd15889ac52be38e9d8fbed4cccfc009ad6f9f758`.
The full repository suite passes 610 tests plus 3 subtests.

The checkpoint/report/manifest hashes were checked before deserialization.
Training-critical sources matched the checkpoint; the legacy unlisted encoder
matched the clean blob at training HEAD. Regenerated center-projective bias /
visibility / combined geometry hashes are
`750b68df2424246682a239bf40c9902c85310fe42bac79c308af638062e4b0a7`,
`026d7654864bea7ae0545bd6448f6def64519a3bedcbc7ea747e7b4b95f82b3a`,
and `0457c09fbaa3b3f9c95f36eb01658fd512937542f592a1c8b45f25d0403d9a48`.
Start/end git state and dirty-diff hashes are identical.

The role-global controls pair every 8,524 train and 990 selection frame with a
different image, scene, and transition. Correct RGB is decisively better than
both controls:

- selection balanced NLL is 0.22805; cross-scene shuffle adds 1.03732 and the
  per-image channel-mean control adds 1.66031;
- train balanced NLL is 0.20091; shuffle adds 0.64510 and mean adds 1.78006;
- shuffle worsens all 8 selection scenes and all 72 train scenes. Mean worsens
  all 8 selection scenes and 71/72 train scenes, with the lone delta only
  -0.00045;
- identity is the best of 51 equal-support spatial transforms in both roles.
  On selection its NLL is 0.22867 versus 0.23453 at the next-best +1 forward
  row, 0.58339 flipped, and 1.12821 transposed.

This proves genuine image grounding and rejects a position-prior collapse or
gross flip/transpose/offset explanation. It also localizes two real limitations:

- directly observable <=2 m OCCUPIED recall on selection is 98.44% in the
  center-visible interior but only 83.46% in exterior ring one. This confirms
  the measured mismatch between finite 0.10 m label cells and center-point
  query support;
- safe selection FREE recall is 8.11% at 1-2 m, 0.76% at 2-3 m, and 0 beyond
  3 m. `rough_local_dynamics` admits no FREE cell; `open_obstacle_field`
  admits 0.018% at 78.95% precision. The current 8x8 token encoder is grounded
  but spatially too coarse for useful long-range FREE evidence.

The access ledger reads only forbidden-role metadata: all 469 G2 rows and 415
calibration rows have zero image/shard-byte opens and zero model outputs. No
training, calibration, threshold selection, or G2 evaluation occurred.

The bounded follow-up sequence is therefore:

1. implement a distinct parameter-neutral
   `projective_cell_square_attention_v1` support using the center plus four
   fixed +/-0.05 m output-cell corners, with no 0.47 m body footprint;
2. score a frozen-weight development counterfactual without claiming
   promotion;
3. run a deterministic train-only micro-fit comparison of 8x8 versus 16x16
   visual tokens under the original center support, so token resolution is the
   sole training intervention;
4. license a full higher-resolution candidate only under the registered
   micro-fit decision rule.

Multi-view memory fusion remains a G3 diagnostic and cannot substitute for the
unchanged single-view 99/95/90 G2 gate.

### Cell-square support and frozen counterfactual: IMPLEMENTED / NEGATIVE

`projective_cell_square_attention_v1` is now bound end to end. Its support
contract SHA-256 is
`904ec5892f789bab55dda93431a0de167333f3887ff6d07f51ccfc79cd0b4107`;
it derives 0.10 m only from the physical-v3 local grid and hashed aggregation
contract
`db288979e7c389df2c4ca846f3309e395bcb6ec7bcf40cb8db6a3107f7e9f717`,
records center plus four +/-0.05 m corners, and explicitly records
`uses_body_footprint=false`. Model state keys, parameter count, and RNG
construction remain unchanged. Checkpoint/report/training/output contracts and
development consumers validate the support by lift type; legacy lifts remain
backward compatible. The full repository suite passes 624 tests plus 3
subtests.

The non-promotable frozen-weight artifact is
`.generated/go2_egomotion_bev_jepa/dev_v4_physical_projective_candidate_v1/frozen_cell_square_counterfactual_v1/spatial_grounding_diagnostic_v1.json`.
Its file/content SHA-256 values are
`c88131efa8ef28b7db30f1105bafad14224d43ced8875f518c82491ee7f92eda` /
`1e71339a66cce24aec5414575317292cf314fa18a3be2a8a8490253bb2ce77ab`.
It strict-loaded the unchanged learned state
`b7928fa3ece2aa093f732f8cae0827d16ff3eafb1a2a63ae2ac45a3aec06eeeb`,
kept calibration and thresholds fixed, trained nothing, emitted no checkpoint,
and evaluated no G2 artifact.

Cell-square support increased visible queries from 1,990 to 2,062, with new
bias/visibility/combined geometry SHA-256 values
`43d54c26354b68505eaecbb74d34165ba03a547759c9779c49edeaa5e1abf0ca`,
`4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b`,
and `bbe1465bc94156a47f19915e73b4c39800b6cb37e6b599b8b842100be4c1a47e`.
However, unchanged weights did not benefit: selection balanced NLL worsened
0.22805 -> 0.23929, admitted-FREE precision fell 0.98591 -> 0.97545, useful
FREE recall changed 0.01974 -> 0.01953, and all-distance OCCUPIED recall was
effectively flat at 0.91620 -> 0.91648. Train shows the same direction. This is
not evidence against training the corrected lift, but it decisively rejects
cell-square geometry alone as the immediate coverage fix.

The next active experiment is therefore the preregistered train-only micro-fit
comparison: center-projective `patch_size=14` (8x8 tokens) versus
`patch_size=7` (16x16), with no calibration, threshold fitting, selection-role,
calibration-role, or G2 access.

### Patch/tokenization-resolution micro-fit: FROZEN BEFORE GPU OUTPUT

The authoritative preregistration is
`docs/lewm_go2_physical_micro_overfit_protocol_2026-07-10.md`. It supersedes
the earlier temporal-quartile, independent-early-stop, token-resolution-only,
and single-seed-license draft. The canonical execution contract contains the
same corrected summary.

The first eight-transition/single-scene pilot was aborted before an
authoritative panel or GPU output because its frozen support gate found zero
medium-maze FREE cells beyond 2 m in fit and cross. The superseding metadata-only
rule hash-partitions all nine train scenes per family into four fit/same-pool
scenes and five cross-pool scenes, splits fit/same by even/odd stream rank, and
selects 32 transitions per family/panel. Each panel has 160 transitions and 320
frames; all panels total 480 globally disjoint rows and 960 endpoint hashes.
Post-selection support must still meet the registered aggregate and per-family
FREE minima or abort without reselection.
The non-authoritative N=32 reproduction passed with minimum aggregate gated-bin
support 20,551 and minimum family/bin support 512; it produced no model output
and is not the authoritative panel.
Both arms consume the same fixed 2,000-update faithful budget; a failure by
either arm automatically runs both arms for the fixed 3,000-update ceiling
budget from their original initial states. Stage pass requires aggregate and
all-family gates at the final three evaluations, including both cross-scene and
same-scene wrong-view controls.

The intervention is the complete patch7/16x16 tokenization and patch-embedding
bundle, not token resolution alone. Holdout comparisons use equal-weight family
macros at a common passing stage and require the registered 5/5 cross-scene and
4/5 same-scene directional checks. A single seed can report only provisional
support. Only `scripts/finalize_go2_physical_micro_overfit.py` may license a
full patch7 candidate after immutable seeds `20260710` and `20260711` agree on
the same favorable branch and provenance. The finalizer requires expected input
file hashes and recomputes decisions from complete stages; bounded smoke runs
use a distinct non-promotable schema and cannot be finalized.

The preparer parses full global row metadata, including non-train path strings,
but emits and dereferences only train-role paths. Source NPZ archives contain
unselected train rows; the optimizer indexes only selected fit rows. No
checkpoint-selection, calibration, or G2 artifact is opened by this diagnostic,
and the diagnostic cannot pass G2 by itself.
