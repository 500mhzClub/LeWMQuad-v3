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

### Two-epoch wiring smoke: PASSED

`.generated/go2_egomotion_bev_jepa/dev_v2_smoke_v1/` (2 epochs,
`--development-only`):

- data: both epochs completed, train loss 1.0419 -> 0.7874;
- calibration: natural-prior stride sample carried all three classes with no
  backfill (224,516 UNKNOWN / 11,729 FREE / 6,590 OCCUPIED); held-out vector
  calibration improved NLL 0.2472 -> 0.1807;
- serialization: checkpoint and report written;
  `EgomotionBevJepaRuntime.load` fully deserializes and validates the
  checkpoint and then refuses it with "checkpoint is not promoted:
  g2_passes must be true" — the correct promotion defense for a
  development-only artifact;
- the G2 role was not evaluated (`g2_evaluated: false`); selection metrics
  after two epochs (free precision 0.4688) are a wiring signal only.

### 20-epoch development-only candidate: COMPLETED / offline G2 bar NOT met

`.generated/go2_egomotion_bev_jepa/dev_v2_candidate_v1/` (20 epochs,
`--development-only`, untouched G2 never read):

- training loss still descending at epoch 20 (0.5674, no plateau);
  selection-role gate-shaped checks improved 6/20 -> 11/20 (peak, epochs
  15-18);
- threshold selection on the calibration role: **0 of 288 candidates pass**.
  Best-effort selected thresholds give planner-admitted free precision
  0.6427 (gate >= 0.99), obstacle detection recall within range 0.5883
  (gate >= 0.95), useful traversable recall 0.3102 (gate >= 0.90),
  free-probability ECE 0.3010;
- failing check families: free precision, obstacle recall/exclusion,
  traversable recall, predicted-route success/length, action margin over
  zero/shuffled on changed cells (0.10 margin), and target effective rank
  (< 4); passing: calibration applied held-out, predictor beats warped
  persistence, target change nontrivial, representation not collapsed,
  wrong-command discrimination on changed cells;
- decision (contract iteration rule): the untouched G2 role is NOT evaluated
  on this candidate — a failed offline gate blocks the one-shot G2 read. The
  measured failure class is map quality plus weak action margin, with
  learning curves still improving at the epoch budget.

### 60-epoch training-limit probe: PENDING (running)

The preregistration increases rows to 128 per scene only if
checkpoint-selection learning curves remain data-limited. A 60-epoch
development-only run on the identical dataset separates training-limited from
data-limited before that decision.
