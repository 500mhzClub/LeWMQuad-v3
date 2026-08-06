# Feasibility check: temporally conditioned encoder-moving V-JEPA 2.1 successor

Date: 2026-08-06
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** Minimal contract and feasibility
check only. No training run was launched. No new corpus, preregistration or
authorization chain was created. Roles `probability_calibration`, `evaluation`,
`untouched` and sealed data were never opened.

**Verdict: FEASIBLE, with one contract decision to make.** Top-block training is
comfortable on the R9700. Genuine same-episode multi-frame history exists for
96.6% of `train` and 99.2% of `checkpoint_selection` rows — but in the dense
`render_textured_v03` render (`224×224`, `textured_v03`), not in the
`go2_render_selected_v04` render (`224×168`, `textured_v04`) that the corpus and
the frozen screen used. Within v04 alone the coverage is 17.5% / 13.3% with zero
`open_obstacle_field` in selection.

An earlier reading of this check reported a hard data blocker. That was wrong: it
looked only at the v04 render and missed the dense v03 render on the 3.7 TB
workspace pool.

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

## 3. Temporal-data coverage

### 3a. Within the v04 render used by the corpus — insufficient

Same-episode history counted by chaining strictly within the corpus
(`scene_id`, `env_index`, `episode_id`, `reset_count`), using only `train` and
`checkpoint_selection` rows. **No duplicated frames, no cross-episode
histories.**

| role | rows | 2-frame history | 3-frame | 4-frame |
|---|---:|---:|---:|---:|
| `train` | 4,262 | 747 (17.5%) | 306 (7.2%) | 140 (3.3%) |
| `checkpoint_selection` | 495 | 66 (13.3%) | 28 (5.7%) | 17 (3.4%) |

The selection subset collapses to 7 scenes, 7 families and **zero
`open_obstacle_field`**, with 31 of 66 rows from `small_enclosed_maze`. Only
1,030 of 4,757 rows (21.7%) even have a `t−240` frame rendered under the v04
contract; supplying `t−240` and `t−480` for all of them would need **7,301
additional v04 renders**.

The cause is the upstream sampling design, not damage: rows were selected by
`hash_rank_within_primitive_env_episode_strata_then_round_robin`, capped at
`max_transitions_per_scene: 64`, which spreads transitions across episodes by
construction.

### 3b. In the dense v03 render — sufficient, under a different camera contract

**Correction to the first reading of this check.** The 3.7 TB workspace pool
holds `.generated/datagen_full/render_textured_v03`: 1,450 scenes at **48,000
frames each** (1,000 steps × 48 envs). It covers **80 / 80 (100%) of the corpus
`train`+`checkpoint_selection` scenes**, and all 4,757 corpus current frames
exist there under **identical filenames** (`frame_NNNNNN_env_NN.png`).

Same-episode history availability in v03:

| role | 2-frame | 3-frame |
|---|---:|---:|
| `train` | 4,204 / 4,262 (98.6%) | **4,116 / 4,262 (96.6%)** |
| `checkpoint_selection` | 495 / 495 (100%) | **491 / 495 (99.2%)** |

Selection per-family, 3-frame history available: `large_enclosed_maze` 64/64,
`local_composite_motifs` 64/64, `loop_alias_stress` 62/64,
`medium_enclosed_maze` 62/64, **`open_obstacle_field` 64/64**,
`rough_local_dynamics` 64/64, `small_enclosed_maze` 47/47,
`visual_sensor_stress` 64/64. **All eight families survive, including the one
that was absent under v04.**

**The two renders are not interchangeable within a clip:**

| | corpus/frozen-screen contract | dense temporal source |
|---|---|---|
| directory | `go2_render_selected_v04` | `datagen_full/render_textured_v03` |
| resolution | `224×168` | `224×224` |
| horizontal FOV | 78.323° | 78.323° (same geometry config) |
| vertical FOV | 62.837° | larger (square aspect) |
| visuals | `textured_v04` | `textured_v03` |
| frames per scene | ~120 (selection) | 48,000 (dense) |

Same rollout, same poses, same frame indices, same horizontal FOV — but a
different vertical field of view and a different texture set. A clip must not mix
them. The label geometry contract (`config/go2_generalization_geometry_v2.json`)
pins only `horizontal_fov_deg: 78.323`; the `raster_labels` observability mask
was nevertheless derived under the v04 frustum, so a v03 frame shows *more*
ground than the labels mark observable — a superset, not a deficit, but a
deviation that must be stated wherever a v03-based spatial number is reported.

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

## 5. The contract decision

Both arms of the specified comparison — frozen control and encoder-moving — are
trained and evaluated on the *same* data, so a v03-native experiment is
internally valid: the frozen control provides the matched spatial reference in
whatever contract both arms share. The v04 figure `0.5103` would then be a
cross-reference from a different visual contract, **not** the comparator.

That makes the choice:

| | v03-native | v04-extended |
|---|---|---|
| rendering required | **none** | 7,301 frames |
| train / selection rows | 4,116 / 491 | 4,262 / 495 |
| families in selection | **8 / 8** | 8 / 8 |
| visual contract | `224×224`, `textured_v03` | `224×168`, `textured_v04` |
| label observability match | image is a superset of the labelled frustum | exact |
| frozen reference | must be re-measured in-contract (cheap) | `0.5103` carries over |
| creates new data | no | yes |

## 6. Options, in order of cost

1. **Run the successor v03-native (recommended).** No rendering, no new corpus,
   no new data — it uses a render that already exists and already covers 100% of
   the corpus scenes. Both arms use v03 frames for history, current and target,
   so no clip mixes camera contracts. Cost: re-measure the frozen V-JEPA 2.1
   spatial reference on v03 frames so the comparison is in-contract (one
   extraction plus one probe, roughly 25 minutes), and state everywhere that the
   v03 image shows more ground than the v04-derived observability mask marks.
2. **Render 7,301 v04 history frames.** Keeps the label frustum and the `0.5103`
   reference exact, at the cost of a render job and of creating new supervision
   data — which needs an explicit decision rather than being done silently.
3. **Two-frame context on the v04-only 747 / 66 subset.** Cheap, but 7 families,
   no `open_obstacle_field`, and 17.5% of the training set: it cannot answer the
   acceptance question as posed.
4. **Defer temporal context** and adapt on single frames. Answers a different
   question — it drops "genuine multi-frame visual context" and "distinct future
   temporal position".

## 7. Operational constraint discovered

`/home/andrewknowles/Workspace` (the 3.7 TB pool) is **100% full, 658 MB free**.
The frozen screen's feature caches alone are 9.2 GB, and a three-frame temporal
cache at ViT-L would be roughly 22 GB. Any new cache must be written to `/`
(546 GB free) or the pool must be cleared first. This is a real blocker for the
run and is unrelated to the data question.
