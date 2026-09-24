# Counterfactual RGB/body maze corpus: completed development result

The V2 composite corpus completed and passed full raw-artifact recomputation:
120 executed branches on 24 procedurally generated layouts, all with a valid
shared physical prefix. Sixteen layouts/80 branches are training and eight
layouts/40 branches are development validation. There is no final-test material.

| Evidence | Result |
|---|---:|
| Physical branches | 120/120 complete; 19 retained originals + 101 untouched pairs |
| Contact stops | 24, retained with native contact evidence |
| Causal RGB/history packets | 8,966 |
| Ideal simulated sensor samples | 53,155 |
| Physics samples | 531,657 |
| Available future motion targets | 891/960 horizons: 595 train, 296 validation |
| Known cumulative-contact targets | 960/960 horizons |

All stop actions were contact-free. Each nonzero action had four contact stops
among sixteen training layouts and two among eight validation layouts. These
are correlated actions within a layout, not 120 independent maze trials.
Unavailable post-contact motion was censored; it was not replaced by terminal
position. Contact labels remain known after the observed contact stop.

The audit rechecked source/gait and artifact hashes, actual 20/0.5 gains, scene
wall identity, native per-contact forces, command timing/slew, prefix crossings,
every measured horizon, causal ideal sensors/histories, proper camera frames and
all image packets. It also repeated the composite prefix-matching reduction.
The learning loader was subsequently exercised on all 120 branches: all five
sibling actions receive exactly equal current model tensors from their layout's
selected real canonical observation. Prospective plans are reconstructed from
the known command and preceding applied command, not future measured control.

## Preserved integrity history

The original V1 attempt remains `INFRASTRUCTURE_FAILURE` after nineteen branches;
it is not retrospectively marked complete. Exact RGB hashes differed despite
identical physics, body/control histories, timing and camera transforms. V2
reverified and retained those branches and executed only previously untouched
pairs under its separate [recovery protocol](go2_counterfactual_maze_dataset_development_v2_recovery_2026-09-05.md).
Across the final corpus, maximum prefix-image RMS difference was 0.049047 raw
8-bit units; maximum changed-pixel fraction was 0.000009766 (three native pixels).
Both fall within the declared limits. Every original image is preserved; model
context canonicalization selects an actual image rather than modifying pixels.

## What this establishes—and does not

There is now audited, action-diverse, scene-separated data with real executed
outcomes and deployment-shaped input channels. The topology hashes are unique
under grid rotation/reflection, but the collection observes **one fixed junction
approach per layout**, with four repeated local exit configurations, randomized
width/initial pose, one material style and one gait/runtime. The unvisited maze
graph does not by itself increase observed visual complexity. This is local
prediction data, not evidence of general maze solving or realistic sensor noise.

The [fixed learning comparison](go2_rgb_body_learning_comparison_development_v1_2026-09-05.md)
has now been launched after audit and loader checks. It compares direct,
supervised-rollout and JEPA conditions with three seeds and common observation
exposure. Its future result must distinguish offline prediction, executed choice
value, and eventual exploration/memory/return. A successful loss reduction cannot
substitute for those later stages.

## Immutable evidence

Output: `.generated/go2_counterfactual_maze_dataset_development_v2_recovery_attempt_001`.
Collection session 11380 and audit session 97632 both terminated with exit 0.

- `result.json`: SHA-256 `bb7307e4a55ee896ad91753f03c309f39d37e7ec497fd95d7b59cf098d387015`.
- `raw_artifact_audit.json`: SHA-256 `fea689e4211b2bff02d76149c60150e9ca65b2c2a6ccc66879dcecfec8834eb6`.
- Original retained roots and each artifact binding are enumerated in launch/result
  evidence; source and audit-helper bindings remain intact.

The current explicit suite across 22 new test files passes 361 tests. No frozen
tracked source or protected benchmark material was changed or accessed.
