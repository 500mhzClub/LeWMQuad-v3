# Storage candidate: `go2_supervised_rollout_mazes_v1_attempt_001`

**Status: NOT deleted.** This directory is a candidate for clearing before E1. It is kept until Andrew gives the go-ahead. This file and the manifest are the record that would remain.

**Location:** `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_supervised_rollout_mazes_v1_attempt_001`

**Size:** 54,372 files, 28,699,375,622 bytes (26.7 GiB).

**Manifest:** [`go2_supervised_rollout_mazes_v1_attempt_001.manifest.tsv.gz`](go2_supervised_rollout_mazes_v1_attempt_001.manifest.tsv.gz), sha256 `d167384f928388ec0874a077b66f882d10254d2459fbc0d79e7b2af5cc5033ee`. It has one row per file: relative path, bytes and sha256.

## What it contains

It is the **"Matched supervised rollout development cohort V1"**, run 9–10 September 2026 under protocol [`docs/go2_supervised_rollout_mazes_v1_2026-09-09.md`](../go2_supervised_rollout_mazes_v1_2026-09-09.md). It belongs to the closed pre-capability programme and is an **evaluation, not a training set** (`model_training: false`).

**What was run:**
- The `MeasuredFloorTransportController` with the assigned model `seed_2026091001_full_supervised_rollout` (model state sha256 `171c8576…`), condition `supervised_rollout`, variant `full`.
- It ran once on each of three reused development layouts, 1–3 (`full_supervised_rollout_novel_maze_01` to `_03`), with 3,000 navigation ticks each.
- **Outcome:** `SUPERVISED_ROLLOUT_MAZES_V1_COMPLETE`, **0/3 measured round trips**; wall time 26,824 s.
- Every qualification flag is false: navigation, hardware, real time, and the JEPA and memory advantages.

**Contents per case (three cases, about 3,014 camera frames each):**

| Kind | Files per case | Size per case |
|---|---:|---:|
| `auxiliary_depth_NNNN.npz` (auxiliary depth camera) | 3,014 | 3.4–3.7 GiB |
| `depth_NNNN.npz` (primary depth) | 3,014 | 2.2–2.5 GiB |
| `native_depth_NNNN.npz` | 3,014 | 2.2–2.5 GiB |
| `rgb_NNNN.png` and `auxiliary_rgb_NNNN.png` | 3,014 each | 80–102 MiB each |
| `context_decisions.jsonl.gz` | 1 | 230 MiB |
| `native_guard_rows.json` | 1 | 25 MiB |
| `physics_trace.npz` | 1 | 22 MiB |
| Per-frame JSON metadata, meshes (`.ply`), audits, identities | — | small |

At the top level: `launch.json`, `result.json`, three `cohort_progress_after_0N.json` files, and per-case admission, audit (about 10.7 MB each), prefix-comparison and worker records, plus `resource_monitor.jsonl`.

**By type:**
- `.npz`: 27,144 files, 27.05 GB
- `.gz`: 3 files, 0.72 GB
- `.png`: 18,084 files, 0.56 GB
- `.json`: 9,128 files, 0.34 GB
- `.ply`: 6 files, 0.02 GB

The depth arrays make up 94% of the size.

## Provenance check: no current training set or the transfer set was derived from it

The check traced **dataset provenance**, not code references (`scripts/trace_go2_storage_candidate_provenance_development.py`). The result is in `e1_storage/provenance_go2_supervised_rollout_mazes_v1_attempt_001.json` under the capability artifact root.

**Roots of the trace.** The records of every current dataset and fit:
- C3/C4 v1: C4-v1 preparation and fit, the C3-v1 readout `go2_maze_view_readout_v1_attempt_003`, its initialisation `go2_full_heading_readout_v1_attempt_001`, and the frozen C3 predictor `go2_horizon_dense_predictor_v1_attempt_001`;
- C3-v2 and C4-v2: matched data, recordings and both fits;
- the C3-v3 round: its registry and the on-policy C1 cohort configuration;
- the transfer set `go2_maze_view_transfer_v1_attempt_001`;
- the pre-registered model bindings.

**Method.**
- Every path into an artifact directory and every sha256 in those records was collected.
- Checkpoint hashes were resolved to their directories by hashing all 121 `.pt` files in both artifact roots.
- Each referenced directory's own records were followed recursively, including first-level subdirectory specifications, launches and tapes for the frame-supplying datasets.
- 1,598 record files were read. The closure has 22 directories, the whole dataset and model lineage.

**Result.**
- **The candidate is not in the closure.** No record of any current dataset or model lineage refers to it by path or checkpoint.
- **Two of its file hashes do appear in the closure's records.** They are `actuator_identity.json` (904 bytes, `733d00d3…`) and `terminal_actuator_gains.json` (259 bytes, `5f726045…`).
  - These are the robot's fixed actuator configuration records, byte-identical in every recording.
  - The same files already exist in `go2_geometry_progress_family_v1_attempt_001`, recorded on 8 September, before this directory existed; the matches are those recordings' own terminal records.
  - This is a shared constant, not derived data.
- **Conclusion:** no current training set (C3/C4 v1, v2, the round) and not the transfer set was derived from this directory. The C3-v3 round's data comes only from its new C1 missions and will be re-checked when prepared.

**Losses if deleted.**
- The raw evidence for the closed 9 September supervised-rollout cohort (0/3 round trips) goes. Documents citing it, such as the 13 September independent-layout documents and `go2_goal_recorded_results_2026-09-22.md`, lose their backing data.
- The manifest and this summary remain.
