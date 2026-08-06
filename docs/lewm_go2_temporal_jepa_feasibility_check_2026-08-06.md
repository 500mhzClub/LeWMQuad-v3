# Feasibility check: temporally conditioned encoder-moving V-JEPA 2.1 successor

Date: 2026-08-06
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** Minimal contract and feasibility
check only. No training run was launched. No new corpus, preregistration or
authorization chain was created. Roles `probability_calibration`, `evaluation`,
`untouched` and sealed data were never opened.

**Verdict: BLOCKED on temporal data coverage, not on memory.** Top-block training
is comfortable on the R9700. The WP-E corpus cannot supply genuine same-episode
multi-frame history at usable scale or family coverage.

---

## 1. Shortest genuine temporal clip

`vjepa2_1_vit_large_384`, local repo `204698b45b37…`, measured directly:

| property | value |
|---|---|
| `tubelet_size` | 2 |
| `patch_size` | 16 |
| depth | 24 blocks, `embed_dim` 1024 |
| `img_temporal_dim_size` | 1 (the single-frame image path used in the frozen screen) |

Through the **video** path the frame count must be a multiple of the tubelet, so
the shortest genuine clip is **T = 2**. Measured at `384×512`:

| clip | tokens | temporal positions | frozen peak VRAM |
|---|---|---:|---:|
| T=2 | `(1, 768, 1024)` | 1 | 1.26 GiB |
| T=4 | `(1, 1536, 1024)` | **2** | 1.31 GiB |
| T=6 | `(1, 2304, 1024)` | 3 | 1.36 GiB |

A masked context-target objective with **prediction at a distinct future temporal
position** needs at least two temporal positions in one clip, so the operative
minimum is **T = 4**: context tubelet `(t−480, t−240)`, target tubelet
`(t, t+240)`. A two-clip variant (context clip ending at `t`, target clip ending
at `t+240`) needs only 3 distinct frames.

## 2. Temporal spacing and the 0.5 s target — verified

All 5,172 corpus pairs have frame-index delta **exactly 240**, same `env_index`,
with no exceptions. The paired-navigation `transition_contract` states
`nominal_duration_s: 0.5` and `one_complete_command_block: true`. The 0.5-second
prediction target is therefore **documented in the corpus contract**, not
inferred from an assumed control rate.

## 3. Temporal-data coverage — the blocker

Same-episode history was counted by chaining strictly within the corpus
(`scene_id`, `env_index`, `episode_id`, `reset_count`), using only `train` and
`checkpoint_selection` rows. **No duplicated frames, no cross-episode
histories.**

| role | rows | 2-frame history | 3-frame | 4-frame |
|---|---:|---:|---:|---:|
| `train` | 4,262 | 747 (17.5%) | 306 (7.2%) | 140 (3.3%) |
| `checkpoint_selection` | 495 | 66 (13.3%) | 28 (5.7%) | 17 (3.4%) |

For the **two-clip minimum** (3 distinct frames: `t−240, t, t+240`):

| role | usable | scenes | families | worst skew |
|---|---:|---:|---:|---|
| `train` | 747 / 4,262 | 64 / 72 | 8 / 8 | `open_obstacle_field` **5 rows** |
| `checkpoint_selection` | 66 / 495 | 7 / 8 | 7 / 8 | `small_enclosed_maze` 31 / 66 |

`checkpoint_selection` per-family: `small_enclosed_maze` 31, `loop_alias_stress`
7, `visual_sensor_stress` 7, `large_enclosed_maze` 6, `medium_enclosed_maze` 6,
`rough_local_dynamics` 5, `local_composite_motifs` 4, **`open_obstacle_field` 0**.

For the **single masked 4-frame clip** the selection set falls to 28 rows across
4 scenes and 4 families, 23 of them from `small_enclosed_maze`, again with zero
`open_obstacle_field`.

**Why the corpus is like this.** It is not damage; it is the sampling design. The
upstream paired-navigation dataset selected rows by
`hash_rank_within_primitive_env_episode_strata_then_round_robin`, capped at
`max_transitions_per_scene: 64`. That deliberately spreads transitions across
episodes to maximise diversity, which makes temporally adjacent transitions rare
by construction.

**Frames on disk cannot rescue it.** The render is a selection, not an episode
dump: 10,311 frames across 96 scenes. Only **1,030 of 4,757** allowed rows
(21.7%) have a `t−240` frame rendered at all. Supplying `t−240` and `t−480` for
every allowed row would need **7,301 additional rendered frames** (8,701 total,
1,400 already present).

**Upstream has the trajectories but not the pixels.** The source
`frames.jsonl` records 915,141 primitive transitions, 202,490 of them
configuration-valid — but 196,596 of those were rejected as
`transitions_missing_rendered_metadata`, leaving 5,894 candidates from which
5,641 rows were written. Dense temporal chains exist in simulation; the
constraint is that their RGB was never rendered.

## 4. Memory feasibility — not a blocker

ViT-L/16-384, `384×512`, T=4, last 2 of 24 blocks plus `norms_block` trainable
(**25,200,640 / 304,680,960 parameters, 8.3%**), AdamW step included:

| configuration | peak VRAM |
|---|---:|
| frozen inference, bs=4 | 1.60 GiB |
| frozen inference, bs=8 | 1.99 GiB |
| top-block training, bs=2 | 1.78 GiB |
| top-block training, bs=4 | 2.56 GiB |
| **top-block training, bs=8** | **3.68 GiB** |
| card total | 31.86 GiB |

Both arms fit with an order of magnitude of headroom. Larger batches, a longer
clip, an EMA target encoder and a dense-token predictor can all be afforded.

## 5. Why the experiment was not launched

The matched comparison was specified to report per-family results with
`open_obstacle_field` explicit, and to test whether encoder movement preserves
inherited spatial information. With 66 selection rows, 7 scenes and **zero
`open_obstacle_field`**, neither the spatial comparison nor the per-family
requirement can be met, and the frozen-versus-moving contrast would rest on 747
training transitions — 17.5% of the WP-E training set, with 5 rows in the family
that already fails hardest.

Running it anyway would produce a number that could not distinguish encoder
movement from sampling noise. That is the failure mode the WP-E closure was
written to stop.

## 6. Options, in order of cost

1. **Render the missing history frames (recommended).** ~7,301 additional frames
   into the existing `go2_render_selected_v04` layout, using the recorded
   `render_replay` plans and the documented Vulkan venv. For scale, the existing
   10,311-frame render ran at ~61 fps. This restores the full 4,262 / 495 split
   with all 8 families and changes no scene assignment, no split and no label
   definition. It is nevertheless **new supervision data**, so it needs an
   explicit decision rather than being done silently.
2. **Reselect temporally contiguous transitions upstream** from the 202,490
   configuration-valid transitions, then render them. Larger, and it would
   replace the corpus rather than extend it.
3. **Two-frame context on the 747/66 subset**, reported honestly as a small,
   family-incomplete pilot with no `open_obstacle_field`. Cheap, but it cannot
   answer the acceptance question as posed.
4. **Defer temporal context**; adapt the encoder on single frames with a masked
   context-target objective over spatial tokens only. This drops the "genuine
   multi-frame visual context" and "distinct future temporal position"
   requirements, so it answers a different question.

Option 1 is the only one that lets the specified experiment run as written.
