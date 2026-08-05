# Go2 dynamic-Cartesian N32 v1 execution binding

Date: 2026-07-11

Status: frozen before sidecar construction, dynamic-model output, training,
checkpoint selection, probability calibration, untouched G2 access, runtime,
held-out, or sealed output.

## Question And License

The physical task needs an RGB-to-Cartesian evidence head whose spatial prior
matches the camera at the measured robot attitude. The independently finalized
full-label diagnostic established:

- result file/content SHA-256:
  `ace9b39c4be31fad84eb7bc2aa65c584acec04febb638672fbcead0db4b6b4fe` /
  `923b401d062819578ee65130007daffeee658044b9ebceccab3f70c1df830567`;
- 1,310,720 cells and all 129,021 known occurrences scored;
- level-center support missed 373 known labels;
- static cell-square support missed 4 OCCUPIED labels;
- full-quaternion/yaw-aligned cell-square support missed zero and supported
  129,021/129,021 known labels in every family.

This licenses exactly one new question: does replacing the frozen level-camera
prior with the deployment-valid per-frame attitude prior allow the existing
patch-7 Cartesian JEPA occupancy head to fit and generalize the registered
physical evidence task? It does not license label, role, threshold, loss,
budget, or post-hoc family changes.

The older categorical-radial N32 ladder remains valid negative evidence for
its own static polar architecture. Its requirement to pass before a shared
JEPA candidate is superseded only for this independently licensed dynamic
Cartesian path; no categorical-radial weights initialize this candidate.

## Immutable Data And Comparator

- physical-v3 dataset manifest file SHA-256:
  `ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180`;
- dataset rows file SHA-256:
  `187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac`;
- v04 source index file SHA-256:
  `11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c`;
- role assignment SHA-256:
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02`;
- rows by role: train 4,262, checkpoint selection 495, probability
  calibration 415, untouched G2 469;
- N32 panel file/content SHA-256:
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c` /
  `f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f`;
- fit/same-scene/cross-scene row SHA-256:
  `5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d` /
  `d32713086c042d20f94825aa362c27a07bef6fd0e0cce0aa5846bb67bf8dc465` /
  `3565f7f7844f3aeee28b0433aa6dc77d553a9ebb831cf9af20b6d392c5416817`;
- immutable static patch-7 comparator file/content SHA-256:
  `6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c` /
  `32d848d3df68e670ddb4cc24436981f62a1aa5562b89e6d6719ecb113f66b749`;
- comparator final-state SHA-256:
  `fba4e91b333d57a813fb94edb13b215064d03da2830aae9d0ae4b34685cd38c1`.

Dataset labels, images, role assignment, rows, and shards are immutable. The
sidecar is metadata-only and may not rebuild, rewrite, or duplicate a label or
image payload.

## Attitude Sidecar

Create one manifest and four canonical role-separated JSONL files under:

```text
.generated/go2_attitude_sidecar/dynamic_cartesian_v1/
```

The roles are exactly `train`, `checkpoint_selection`,
`probability_calibration`, and `g2_evaluation`. Each sidecar row has exact keys:

```text
schema
global_row
dataset_role
row_identity_sha256
scene_id_sha256
frames_jsonl_sha256
env_index
current_frame_index
next_frame_index
current_timestamp_ns
next_timestamp_ns
current
next
content_sha256
```

`schema` is `lewm_go2_attitude_sidecar_row_v1`. `current` and `next` are
exactly `{base_quat_world_xyzw,stored_base_yaw_rad}`. Quaternion order is XYZW.
Every numeric value is a finite JSON number and not boolean. Each row's content
hash covers every other key using canonical JSON.

Join each immutable dataset row to its committed source `frames_jsonl` using
the bound source hash plus scene, `(frame_index,env_index)`, timestamp, episode
ID, reset count, and episode step for both endpoints. Bind `global_row`, scene,
label-shard row/hash, and current/next image hashes into
`row_identity_sha256`, although those label/image payload bytes remain unopened.
Reject any missing, duplicate, non-injective, reordered, role-mismatched, or
partially joined row.

For each endpoint, require quaternion norm within `1e-5` of one and wrapped
quaternion-yaw disagreement at most `1e-5 rad`, using the exact arithmetic of
`go2_dynamic_cell_square_projection.py`. The source is only
`base_quat_world_xyzw` plus `base_rpy_rad.yaw`; recorded world camera pose,
absolute world position, geometry, depth, and labels are forbidden inputs.

The sidecar manifest binds dataset/rows/source-index/render-audit/dynamic-
geometry hashes, exact role counts, role assignment, every source-frame file
hash, every role file/content/ordered-identity hash, the complete source map,
and an access ledger. Construction may use a bounded six-process CPU pool with
one native numerical thread per worker and canonical scene/global-row merge.

Training may open only train, selection, and calibration sidecar files. The G2
file may be mechanically created and hash-bound without any distribution
summary, but its bytes remain unopened until the role-global one-shot G2
attempt marker has been atomically published.

## Sole Model Intervention

Add the distinct lift literal:

```text
dynamic_projective_cell_square_attention_v1
```

Legacy lifts retain their exact APIs, state keys, outputs, and strict loading.
Attitude arguments are required for the dynamic lift and forbidden for every
legacy lift.

The dynamic model keeps:

- input 112 x 112, patch size 7, ordered 16 x 16 tokens;
- encoder width/depth/heads 192/6/6;
- BEV width 64 and Cartesian grid 64 x 64 at 0.10 m;
- ranges with centers `[-0.95,5.35]` forward and `[-3.15,3.15]` left;
- four attention heads, predictor width 128, and existing refinement;
- camera H/V FOV `78.323/62.8370386364 deg`, near `0.05 m`, mount
  `(0.326,0,0.043)` with zero mount RPY;
- vertical anchors `(-0.333,-0.133,0.067,0.267,0.467) m`;
- cell center plus four `+/-0.05 m` square corners;
- attention sigma 2.0 tokens and floor -6.0;
- UNKNOWN/FREE/OCCUPIED order and hierarchical equal-capacity loss.

For every batch, reconstruct the yaw-aligned camera from raw normalized XYZW
quaternion and stored yaw without renormalizing. Compose the fixed mount,
project all 25 cell witnesses, reduce the minimum image-token distance, and
produce a per-frame attention bias `[B,4096,256]` plus visibility `[B,4096]`.
Repeat the bias over the four attention heads for `MultiheadAttention`.
Geometry calculation uses bounded float32 on GPU0 with an explicitly tested
inclusive boundary tolerance; a float64 CPU reference must reproduce the
stdlib support mask on the complete 320-frame fit panel before model output.

`occupancy_logits` accepts image, quaternion, and yaw. Full JEPA `forward`
accepts current/next quaternion and yaw, concatenates the paired attitudes in
the same order as paired online images, and uses next attitude for the EMA
target decoder. The action-conditioned predictor and SE(2) warp do not consume
attitude and remain unchanged.

At deployment, quaternion and yaw come from one timestamp-aligned base-state
or calibrated IMU/odometry orientation record. Genesis WXYZ is reordered to
XYZW. World position, scene identity, simulator geometry, and target state are
not model inputs. Missing, stale, nonfinite, bad-norm, or yaw-inconsistent
attitude fails closed; level-attitude fallback is forbidden.

## N32 Fit Qualification

N32 means the frozen 160-transition/320-frame fit panel and two independent
320-frame holdouts, not 32 frames. Use only the occupancy head and set JEPA,
equivariance, action-contrast, and variance weights to zero for this bounded
mechanism qualification. No checkpoint, calibration, threshold, G2, or runtime
claim follows directly from N32.

For seeds `20260710` then `20260711`, preserve the existing panel expansion,
wrong-view permutations, FP32 preprocessing, batch size 4 frames, direct
balanced hierarchical loss, gradient clip 1.0, and full-budget rules:

1. production-faithful AdamW: LR `2e-4`, weight decay `1e-4`, 2,000 updates;
2. only if its terminal fit gate fails, restart the identical initial state for
   ceiling AdamW: LR `1e-4`, zero weight decay, 5,000 updates.

Evaluate every 100 updates. Fit passes only when the final three evaluations
all pass aggregate and every family:

- balanced hierarchical NLL `<=0.03`;
- UNKNOWN/KNOWN and conditional FREE/OCCUPIED balanced accuracy `>=0.99`;
- UNKNOWN, FREE, and OCCUPIED recall each `>=0.98`;
- FREE recall `>=0.95` in 1-2 m, 2-3 m, and >=3 m bins;
- role-global and same-scene wrong-view minus correct NLL each `>=0.25`.

Wrong RGB retains the target frame's attitude, so the control changes visual
content rather than target geometry. Holdout image/label/model access is
forbidden unless the invoked branch passes its terminal fit gate. Authorized
holdouts run once. Preserve the prior family-macro comparator arithmetic; the
dynamic candidate must satisfy both the absolute fit gate and the existing
strict static patch-7 holdout-improvement gates. Two favorable registered seeds
license shared-JEPA construction; neither seed may be selected or retried.

## Shared-JEPA Development Sequence

Only after N32 qualification, construct a fresh shared model; N32 learned
weights are forbidden initialization. Freeze a separate implementation
manifest and then:

1. run a two-epoch development-only wiring smoke;
2. run the registered 20-epoch development candidate over all 4,262 train rows;
3. select checkpoints only on 495 checkpoint-selection rows, subject to JEPA
   health and physical-evidence eligibility;
4. fit hierarchical natural-prior calibration and select thresholds only on
   all 415 probability-calibration rows;
5. freeze checkpoint, calibration, thresholds, code, source and sidecar hashes;
6. atomically reserve the role-global G2 generation before opening all 469 G2
   images, labels, sidecar rows, or model outputs exactly once.

The numerical G2 gates remain unchanged: admitted physical-FREE precision
`>=99%`, directly observable physical-obstacle recall within 2 m `>=95%`, and
useful physical-FREE recall `>=90%`, with calibration reported. No family-
specific threshold or optimistic UNKNOWN routing is permitted.

Train a matched development-only no-JEPA-loss ablation from the same initial
state/data order. It is evaluated at the promoted arm's preselected update and
cannot select or replace the promoted checkpoint.

## Resources And Outputs

All neural execution must use only the discrete R9700:

```text
env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES=0
```

The 2 GB Raphael integrated GPU (GPU1) is forbidden for model construction,
training, inference, or evaluation. Use approximately six DataLoader/source
workers with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`; preserve canonical ordering and deterministic seeds.

Authoritative N32 outputs are exclusively:

```text
.generated/go2_dynamic_cartesian_n32/v1/seed_20260710_result.json
.generated/go2_dynamic_cartesian_n32/v1/seed_20260711_result.json
```

A distinct three-update smoke is nonauthoritative and fit-only. Runner and
torch-free finalizer must bind complete source/input/sidecar/state/schedule
hashes, exact access ledgers, conditional holdout non-access, and immutable
no-replace publication. Any source, test, sidecar, threshold, data, or command
change requires a new pre-output amendment.

## Required Source Tests Before Any Model Output

- sidecar exact schema, row join, role isolation, canonical hashes, quaternion
  norm/yaw boundaries, bool/int/float mutations, duplicate and reorder failure;
- stdlib versus Torch support parity at level attitude and diverse tilts,
  including support count/hash `2062` /
  `4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b`;
- full 320-frame float32/float64 support parity and zero known unsupported;
- batch/single parity, current/next sensitivity, attention-head mask ordering,
  finite outputs, gradients, and no attitude gradient requirement;
- dynamic attitude required, every legacy API rejects extra attitude, and
  legacy state/output compatibility is unchanged;
- one encoder call per image and both JEPA/occupancy gradients reach the shared
  encoder in the later shared mode;
- exact N32 initialization, schedule, controls, gates, access order, result
  schema, finalizer recomputation, and tamper rejection;
- runtime WXYZ-to-XYZW/yaw parity and fail-closed missing/stale attitude.

No dataset-backed model output, G2 payload, physical-nontrain payload, runtime
result, held-out, or sealed artifact was opened to freeze this binding.
