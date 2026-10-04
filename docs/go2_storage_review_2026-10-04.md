# Storage review: stale artefacts that could be removed, 4 October 2026

**This is a read-only review, requested by Andrew ("do a full review on what stale artefacts can be removed"). Nothing has been deleted.**

Every scan reads filenames and sizes only. Scans skip `sealed*` paths, and the sealed backups folder on the workspace drive is excluded entirely.

Scripts:
- `scripts/review_go2_storage_candidates_2026_10_04.py`: navigation roots, lineage and references.
- `scripts/classify_go2_storage_dirs_2026_10_04.py`: per-directory file types.

Sizes are allocated bytes. Where hard links exist, sizes were measured with `du`, which counts each shared block once.

## The three drives

| Drive | Size | Free | Project data on it |
|---|---:|---:|---|
| Data drive (`/`), which holds RecoveryStorage and home | 1.8 TB | 84 GiB | 942 GiB of RecoveryStorage, 182 GiB temporal cache, 113 GiB in `~/.local/share` |
| Workspace drive (`/mnt/workspace_drive`, which is also `~/Workspace`) | 3.7 TB | **5.5 GiB** | the repository and its `.generated` folder (3.52 TiB), of which 3.33 TiB is the May training corpus |
| Third drive (`/mnt/steam_drive`) | 458 GB | **0.5 GiB** | 51 GiB of navigation roots moved there in September; the rest is the Steam library |

The repository, git and every committed document live on the workspace drive, which has 5.5 GiB free. That makes this drive the most urgent; the third drive is also full.

## Protected; not candidates

- **By rule:**
  - the capability programme root (`go2_navigation_capability_v1_attempt_001`, 167 GiB), including the sealed and preliminary sets;
  - the decision-headroom lineage (`go2_decision_headroom*`, `go2_headroom_*`; Andrew: do not touch).
- **Current data lineage** (any root named by a current manifest):
  - the decoder-fit feature-cache sources (`dev_c3_cache_v1`; the stage-2 refit rebuilds from these);
  - the C3-v2, C3-v3 and C4 data manifests;
  - the maze-view readout and transfer manifests;
  - the frozen predictor's frame list.
- **In practice this protects:**
  - 12 roots on the data drive (201 GiB), including the predictor's three source roots;
  - three workspace roots: `go2_balanced_start_horizon_actions`, `go2_full_heading_training` and `go2_full_heading_readout`.
- **Parts of the temporal cache in use:**
  - `proprio_v1` (the current harness loads `proprio_norm_stats.json`);
  - `factorial_v1`, `temporal_rows.jsonl` and `temporal_manifest.json` (used by the headroom checks);
  - the authorisation files.
- **Roots excluded by the retention policy:**
  - `go2_all_phase_adapter_maze02_matched_native_v1_attempt_001` (a pinned reference);
  - `go2_clearance_preferred_arc_recovery_learned_round_trip_native_layout03_v1_attempt_001` (may overlap the retained arc-recovery recordings).
- **Active environments:** `genesis_rocm_0_4_6_v1` (runs every mission), `world_model_rocm_7_2_1_v1` (imported by it) and `genesis_render_vulkan` (rendering).
- **Kept as evidence:**
  - the 45 clean-export source copies beside the repository (custody evidence);
  - `models/checkpoints` (current).

No current manifest names the May training corpus (`datagen_full`) or the temporal cache, apart from the `proprio_v1`, `factorial_v1` and `temporal_rows` files above.

## Candidates, ranked by what is lost

### Tier 1: depth in closed navigation roots, 460 GiB (recommended)

This is the same depth-only procedure used on 26 September and on 2 and 4 October:
- exact per-root file counts;
- every non-depth file hashed before and after;
- a `depth_retention.json` marker in each root;
- a receipt.

Results, failure records, RGB, physics, commands and trajectories all stay. What is lost is historical full-depth replay of these recordings.

Every root here was last written between 5 and 18 September, before the capability programme. None is named by a current manifest or pinned by the policy.

| Drive | Roots | Depth files | GiB | Largest families |
|---|---:|---:|---:|---|
| Data drive | 212 | 927,864 | 400.4 | `independent_rgb_body_collection` 50.6, `fresh_stable_reference` 19.4, `heading_first_terminal_reactive` 8.4 |
| Workspace drive | 30 | 54,034 | 28.0 | `coupled_room_return` 5.3, `room_return_pulse` 4.8; 28 early-September roots plus 2 in the workspace navigation folder |
| Third drive | 25 | 107,944 | 32.1 | `view_replan_repeatability`, `polygon_floor_repeatability`, `return_routing_memory` |
| **Total** | **267** | **1,089,842** | **460.5** | |

### Tier 2: bulk arrays and checkpoints of closed May–August programmes, about 470 GiB

In this tier I would keep:
- each run's `result.json`, reports, receipts, logs, configurations and small JSON;
- every file a repository document cites by path.

I would retire only the bulk feature arrays (`.f16`, `.u8`, `.npy`, `.npz`) and intermediate checkpoints (`.pt`).

The 7 September review called the temporal cache and the planning-utility folder "not blanket cleanup candidates", because documents cite results inside them. This tier respects that: it removes no cited file. Each programme in it closed by the end of August.

| Item | GiB | Bulk content | Newest |
|---|---:|---|---|
| Temporal cache, minus its protected parts (`~/.cache/lewm_go2_temporal_v03`) | 173 | `temporal_action_jepa_v1` 80.5, `horizons` 25.3, `two_step` 23.9, `four_step_rollout_v1` 11.3, `dense_temporal_true_future…` 10.7, `place_head_dev_v2` 7.2, `frozen_spatial_reference` 6.7 | 21 Aug |
| Workspace `.generated`, May–August folders (197) | 151 | `go2_memory_closed_loop` 48.7 (JSON 33.5, npz 12.2), `dev` 30.5, `jepa_phase3a` 28.0 (JSONL 27.2), `go2_hidden_target_memory` 5.5, `jepa_counterfactual` 5.2, `task_aligned_*` 7.9, May review videos and pilots about 10 | 21 Aug |
| Planning utility (`~/.local/share/lewm_go2_planning_utility_v1_2`) | 84.8 | `.f16` 66.6, `.pt` 17.6 (counterfactual fidelity, V-JEPA ablation, oracle scorer) | 17 Aug |
| August qualifications on RecoveryStorage | 64.5 | failed copies of `jepa_local_waypoint_planning_cost_qualification_v1` 32.0 (arrays; four large, measured with shared blocks counted once), its final run 7.9, `plan_aware_monotone` failed copy 5.1, `body_centric`/`minimum_multi_origin`/`non_greedy`/`occluded_goal` 19.5 | 5 Sep |

The `dev` folder holds the frozen dense-representation screen (V-JEPA 2.1 vs DINOv2), a result still cited. Its report stays; only its 9.2 GiB feature arrays would go.

Before acting on this tier I would run one more filename-and-citation pass to list exactly which files are kept. I would act only on Andrew's approval of that list.

### Tier 3: old Python environments, 28.8 GiB

These are `lewmquad-v12-runtime-torch291-rocm64` (14.5 GiB), `lewmquad-v12-runtime-rocm711` (13.5 GiB) and `lewmquad-torch-hash-cpu` (0.8 GiB).
- No running process, active environment or current script uses them. They were last referenced in July and August execution records.
- **Lost:** exact re-execution of those July and August runs until the environments are rebuilt.
- The 7 September review asked for an explicit historical-retention decision before removing them.

### Tier 4: the May training corpus on the workspace drive (`datagen_full`, 3.33 TiB)

This is the original LeWM corpus: 1,450 scenes, 48 environments each, 1,000 steps. The capability programme does not use it.
- No current manifest names it.
- The newest script reading it dates from 18 August, apart from the 22 September deduplication.
- It is the only large item on the drive the repository lives on.

| Part | GiB | Lost if removed | Recovery |
|---|---:|---|---|
| `render_textured_v03`: 69.6 million rendered RGB frames | 2,667.5 | Retraining or evaluating models on the May corpus images | Re-render from the kept bags, plans and scene corpus. The original render ran 28–30 May, about two days of GPU time, which is over the 24-hour line, and bit-identical output is not guaranteed. |
| `rollout/*/raw/messages.jsonl`: decoded message logs | 335.9 | Nothing permanent: the logs are decoded from the kept `.mcap` bags | The convert step takes about 27 s per scene, so about 40 min on 16 workers. Check that the converter reproduces a sample byte for byte before relying on this. |
| Keep: `.mcap` bags (about 163), `frames.jsonl` (162.7), `labels.jsonl` (77.1), plans, scene corpus | about 404 | | These are the sources for any regeneration |

Options, from least to most loss:
- (a) Keep everything.
- (b) Remove only the message logs: 336 GiB, regenerable in about an hour.
- (c) Also remove the render for the training split only. The split shares are 69% train, 10% val, 10% `test_id` and 10% `test_hard` by rollout size, so this is about 1,835 GiB (1.8 TiB). The held-out splits stay renderable for evaluation.
- (d) Remove the whole render: 2.6 TiB.

This is Andrew's call. It decides whether the project keeps the ability to retrain the May-era world model without about two days of re-rendering.

### Not recommended

- `RecoveryStorage/.../models`, May–June checkpoints (23.4 GiB): `textured_v03_full` (the seq4 sweep, the June navigation base), `rollout_stage2`, `scaled_ablation` and the pose-aux ladders. Trained models are costly to recreate and small here.
- Small regenerable caches (about 4.4 GiB): npm, Triton, Gstaichi, COMGR, Quadrants and the Genesis `.gsd` cache.
- Old sibling folders on the workspace drive (about 2.8 GiB): `LeWMQuad-v2`, `TinyQuadJEPA-v2`, ComfyUI, the stable-diffusion web UI, April smoke checkpoints. Not worth the review time.
- Personal items are outside this review: the Steam libraries (146 GiB on the data drive, 382 GiB on the third drive), Games/GOG 144 GiB, Downloads 35.6 GiB and a video file on the workspace drive.

## Summary

| Tier | Data drive | Workspace drive | Third drive | Loss |
|---|---:|---:|---:|---|
| 1. Depth in closed roots (recommended) | 400.4 | 28.0 | 32.1 | historical depth replay only |
| 2. Closed-programme bulk arrays and checkpoints | 322 | 151 | – | intermediate arrays; results and cited files kept |
| 3. Old environments | 28.8 | – | – | re-execution of July and August runs |
| 4b. Corpus message logs | – | 335.9 | – | none permanent (about 1 h to regenerate) |
| 4c/4d. Corpus render (train only / all) | – | about 1,835 / 2,667.5 | – | May-corpus images (about 2 days to re-render) |

Recommended now: Tier 1. That gives the data drive about 484 GiB free, the workspace drive about 33 GiB and the third drive about 33 GiB. Tier 4b would add 336 GiB on the workspace drive at little cost. Tiers 2, 3, 4c and 4d each need Andrew's decision.

If approved, each tier runs as its own receipted job:
- exact file lists and counts asserted;
- non-deleted files hash-checked where the tier keeps files beside deleted ones;
- no mission writes to a root while it is processed;
- an entry in the retention policy.
