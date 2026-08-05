# Go2 categorical radial micro-fit protocol

Date preregistered: 2026-07-10 21:09 BST

Status: active; written before any categorical-radial model output

## Purpose

The authoritative patch14-versus-patch7 diagnostic proved that the existing
projective cell-query decoder is RGB- and view-grounded but cannot reconstruct
precise physical FREE/OCCUPIED/UNKNOWN boundaries. Patch7 changed token count
without changing the effective 14-pixel projective blur. This protocol tests
one new mechanism: preserve bearing and range explicitly, keep projected
vertical anchors distinct, and lift categorical radial evidence into the
existing Cartesian physical map deterministically.

This remains a train-role diagnostic. It cannot pass G2, select thresholds, or
license checkpoint-selection, calibration, development, or G2 access by
itself.

## Frozen inputs and isolation

The experiment reuses, without modification or reselection:

- dataset manifest
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json`,
  file SHA-256
  `ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180`;
- train-only panel
  `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`, file SHA-256
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`;
- its 160/160/160 transitions, 320 frames per panel, 480 disjoint rows, 960
  unique endpoint images, 45 train scene shards, five families, and fixed
  post-selection support thresholds;
- observable physical v3 labels, 64x64 output grid, corrected V04 camera, and
  unchanged hierarchical loss and evaluation metrics.

Only paths emitted by the frozen train-only panel may be opened. Every artifact
must record zero image-byte, label-shard, and model-output contact for
checkpoint-selection, probability-calibration, G2, and every other non-train
role. The sealed benchmark remains unopened.

The immutable negative reference is
`.generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json`,
file SHA-256
`6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c`.

## Geometry factorization gate

The registered categorical radial lattice is:

- 64 radial bins, each 0.10 m, covering `[0.0, 6.4)` m;
- 256 bearing bins spanning the corrected camera horizontal field of view,
  `[-39.1615, +39.1615]` degrees;
- Cartesian cell centers use `r = hypot(forward, left)` and
  `bearing = atan2(left, forward)`;
- each representable Cartesian cell maps to one fixed nearest radial/bearing
  bin; mappings must be injective;
- cells behind the camera or outside the horizontal field of view are fixed to
  UNKNOWN and are not learned.

Before any GPU output, a geometry-only audit must prove on every one of the 960
frozen panel frames that:

1. every supervised FREE or OCCUPIED Cartesian cell is in representable
   support;
2. no two representable Cartesian cells share a radial/bearing bin;
3. scattering integer Cartesian classes to the radial lattice and gathering
   them back reproduces every FREE/OCCUPIED/UNKNOWN target exactly;
4. axis, angle-sign, range-bin, and outside-FOV mutations fail focused tests.

Failure aborts before model training. A changed lattice or a conservative new
target contract requires a dated amendment written before model output.

## Candidate architecture

The candidate name is `projective_categorical_radial_patch7_v1`.

- input remains 112x112 RGB;
- the encoder remains patch7, 16x16 spatial tokens, width 192, depth 6, six
  attention heads;
- the five fixed body-frame vertical anchors remain
  `(-0.333, -0.133, 0.067, 0.267, 0.467)` m;
- each anchor is projected and bilinearly sampled separately from the spatial
  token grid; no minimum-distance union may discard anchor identity;
- sampled anchor features are combined with fixed radial/bearing coordinates;
- a shared radial decoder emits three categorical logits for every registered
  radial/bearing bin, with radial context available to express FREE before a
  visible surface, OCCUPIED surface evidence, and UNKNOWN behind it;
- the fixed injective gather produces the existing 64x64 Cartesian logits;
- outside-support logits are deterministically UNKNOWN;
- no simulator depth, occupancy, pose other than fixed camera calibration,
  scene identity, image identity, analytic RGB mask, or exact target vector is
  available to the model.

The first candidate changes only the decoder/factorization. It does not bundle
native-resolution input, stride-4 skips, a new encoder, memory fusion, JEPA
losses, exploration logic, or threshold changes.

## Overfit ladder

The runner must pass these stages in order and must stop at the first failure:

1. one frame;
2. four frames;
3. sixteen frames;
4. the complete N32 fit panel of 320 frames.

Ladder frames are chosen once by the fixed namespace
`go2_categorical_radial_ladder_v1`. The one-frame anchor is the lowest-hash fit
frame containing all three classes. Remaining frames are hash ranked using
only their frozen metadata, and the lowest-ranked frame from each previously
unused scene is appended until 16 scene-disjoint frames are frozen. The first
4 and all 16 therefore retain all-class cumulative support and admit a
zero-match cross-scene wrong-view permutation. This explicit label-based
anchor is used only for the train-role implementation ladder. No frame may be
substituted after model output.

Every rank is the hexadecimal SHA-256 of canonical compact JSON. Frame ranks
hash `[namespace, "frame", scene_id, global_row, side, image_sha256]`; scene
ranks hash `[namespace, "scene", scene_id]`. The anchor is ranked by the frame
rule after the all-class predicate. For each other scene, its lowest-ranked
frame is selected, then scenes are ordered by `(scene_rank, scene_id)`, with the
anchor scene first. These exact identities and their combined content hash are
written before training.

For one frame, balanced hierarchical NLL must be below `1e-3` and every
supported class recall must equal 1.0. The 4- and 16-frame stages require NLL
below `0.01`, every supported class recall at least 0.99, and correct RGB NLL
at least 0.25 below a fixed wrong-view RGB control. These are implementation
and capacity checks, not promotion results.

Every ladder size restarts from the same seed-specific initialization. N=1
uses AdamW for 1,000 fixed updates with batch size 1; N=4 uses 1,500 updates
with batch size 4; N=16 uses 2,000 updates with batch size 4. All three use
learning rate `2e-4`, weight decay `1e-4`, gradient clipping at 1.0, and an
evaluation every 100 updates. The complete budget is consumed; there is no
early success stop. A failed stage prevents larger stages from running.

The N32 stage uses the unchanged authoritative gate from the patch/token
protocol at the aggregate and each of five family levels:

- balanced hierarchical NLL <= 0.03;
- UNKNOWN/KNOWN and FREE/OCCUPIED balanced accuracy >= 0.99;
- UNKNOWN, FREE, and OCCUPIED recall >= 0.98;
- FREE recall >= 0.95 in 1-2 m, 2-3 m, and >=3 m bins;
- cross-scene and same-scene wrong-view minus correct-RGB NLL >= 0.25.

An evaluation passes only if the aggregate and all five families pass. The
terminal gate requires three consecutive passing evaluations.

## Optimization and decision rule

The faithful N32 stage uses AdamW, batch size 4, learning rate `2e-4`, weight
decay `1e-4`, gradient clipping at 1.0, 2,000 fixed updates, and evaluation
every 100 updates. If it fails, the capacity ceiling restarts from the same
initial state and uses AdamW, batch size 4, learning rate `1e-4`, zero weight
decay, gradient clipping at 1.0, 5,000 fixed updates, and evaluation every 100
updates. There is no metric-selected early stopping.

If and only if the N32 fit gate passes, the same-scene and cross-scene panels
are evaluated. Against the immutable faithful patch7 reference on the same
rows, the candidate must have:

- equal-weight family-macro NLL ratio <= 0.80 on both holdouts;
- equal-weight family-macro >=3 m FREE-recall delta >= +0.10 on both;
- every family-macro class-recall delta >= -0.01;
- no individual family/class recall delta below -0.01;
- all five cross-scene families and at least four same-scene families strictly
  improve both NLL and far-FREE recall.

A single qualifying seed is provisional only. Seeds 20260710 and 20260711
must independently take the same favorable branch before a full
categorical-radial G2 training candidate is licensed. Aggregation consumes only
immutable, prehashed result artifacts and recomputes every decision.

If the one-frame stage fails, the implementation or loss is wrong. If one and
four frames pass but 16/N32 fails, capacity is insufficient. If N32 fit passes
but held-out far-range evidence fails, the next isolated intervention is native
224x168 rectangular input; it must not be retrofitted into this result.

## Predictive JEPA boundary

This diagnostic sets JEPA, equivariance, action-contrast, and variance weights
to zero so it isolates observable physical perception. Passing it licenses a
full perception candidate, not a JEPA claim. Predictive JEPA objectives are
restored later and must separately beat persistence while satisfying effective
rank and variance gates before the learned frontier and online-memory claims
can be made.
