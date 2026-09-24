# Recurrent-H4 six-transition metadata census preregistration

Date: 2026-07-27
Status: **PREREGISTERED — NOT EXECUTION AUTHORITY**

## Decision this audit may make

Run one read-only census to answer only this question: does the already-frozen
development pair index contain enough exact, role-preserving recurrent-H4
sequences to justify writing a separate causal-belief-state JEPA experiment
preregistration?

Here recurrent H4 means three ordered causal observations
`(o[t-2], o[t-1], o[t])`, the two intervening past actions, and four ordered
future actions/targets through `t+4`. Its metadata witness is therefore six
joined transitions and seven endpoints: `p0` and `p1` supply history, while
`p2` through `p5` supply the four-action future horizon. The current
observation at the `p1 -> p2` boundary is the fixed anchor for a later
reversed-past control; this census does not construct or execute that control.

The census may return `H4_METADATA_FEASIBLE` or a complete
`STOP_H4_METADATA_INADEQUATE` receipt. It may not create a dataset, choose a
training schedule, train or evaluate a model, or authorize a later run.

## Frozen public evidence base

Only committed public source and receipts were used to write this draft. The
current modified `lewm/datasets/go2_paired_navigation.py` and all untracked raw
supervision modules are expressly excluded.

- Frozen primitive-transition publisher:
  `lewm/datasets/go2_paired_navigation.py` at
  `713dee6f841deec3624e568edb9ac454bee0c6e6`, SHA-256
  `05288b641aff838fa1b91d66f9b668217dab5514edc9683e6bb910048b5b7084`,
  84,750 bytes. It groups frames by `(env_index, episode_id, reset_count)`,
  rejects duplicate/nonconsecutive episode steps, and publishes only complete
  reset-safe nominal 0.5-second primitive blocks.
- Frozen H1 temporal-index prototype:
  `scripts/run_go2_rgb_causal_temporal_perception_v1.py` at
  `75240453b69cbbe34e6dbbdd5e65765aba7d26e6`, SHA-256
  `941db26b14a956aac89b0d762e64448e6efdbf1ca1a4d79741eb305d9096200b`,
  118,143 bytes. Its `_selection_temporal_index` validates the exact pair
  schema, endpoint identities, stream context, unique predecessor and
  population before constructing the already-reviewed one-step temporal map.
- Frozen H1 contract at the same commit:
  `lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py`, SHA-256
  `ba3fd9cda5c1d3d4b3383b192bfb3ccafa6e5bd08e581c0e1d147c34d0c9e949`,
  91,834 bytes. It binds the exact pair-index file hash and byte count, all
  three role populations, the eight-family set and the inherited canonical
  JSON rules used by the reviewed H1 runner.
- Frozen index validator:
  `scripts/run_go2_shared_jepa_v5_matched_training_v1.py` at
  `d6c517b0adcd266ba0c4110e3cdf4910f1305e8e`, SHA-256
  `e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578`,
  103,456 bytes. It independently binds the complete pair population.
- Frozen matched contract at the same commit:
  `lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py`, SHA-256
  `53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a`,
  41,189 bytes. It defines the raw-manifest validator and the exact ordered
  pair-content identity.

No prior execution authorization is reused as present authority.

## Sole proposed runtime inputs

An eventual separately reviewed census may open only these two existing
development-metadata files after checking each exact binding:

| Input | SHA-256 | Bytes / population |
|---|---|---|
| `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1/manifest.json` | `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360` (file), `74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a` (canonical content) | 311,598 bytes |
| `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1/pairs.jsonl` | `5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d` | 6,207,286 bytes; 5,172 rows |

The manifest must be one canonical self-hashed JSON line and must reproduce
the exact frozen manifest field set and role populations. Its `pair_index`
must be exactly `{"path":"pairs.jsonl","row_count":5172,
"file_sha256":"5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d"}`,
and its `files` inventory must contain exactly
one `pairs.jsonl` row with the same file hash and byte count `6,207,286`.
The census must also reproduce the existing ordered pair-content hash
`76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`
as SHA-256 of the canonical JSON encoding of the list of all 5,172
`content_sha256` strings in physical JSONL order. The endpoint index is not
needed: exact pair endpoint identities and stream contexts contain the
complete join witness.

Expected immutable populations are:

| Role | Pairs | Unique endpoints | Scenes |
|---|---:|---:|---:|
| `train` | 4,262 | 7,777 | 72 |
| `checkpoint_selection` | 495 | 924 | 8 |
| `probability_calibration` | 415 | 759 | 8 |

Calibration rows are validated only for exact population, schema, content
self-hash, unique row identities and the global endpoint-use integrity check
defined below. They are never graph-joined or eligible
for a sequence. Because this census does not open `endpoints.jsonl`, it makes
no endpoint-index referential-validation claim.

## Exact schema, projection and join

Each canonical JSON row must first have exactly these 20 fields:
`schema`, `dataset_role`, `global_row`, `scene_id`, `family`, `episode_id`,
`env_index`, `reset_count`, `source_split`, `frames_jsonl_sha256`,
`scene_manifest_sha256`, `primitive`, `relative_se2_current_frame`,
`current_endpoint_sha256`, `next_endpoint_sha256`,
`label_shard_path_metadata_only`, `label_shard_sha256`, `label_shard_row`,
`sidecar_row_identity_sha256`, and `content_sha256`.

The frozen role order is exactly `train`, `checkpoint_selection`, and
`probability_calibration`. The frozen primitive order is exactly `arc_left`,
`arc_right`, `backward`,
`forward_fast`, `forward_medium`, `forward_slow`, `hold`, `yaw_left`, and
`yaw_right`. The frozen family order is exactly `large_enclosed_maze`,
`local_composite_motifs`, `loop_alias_stress`, `medium_enclosed_maze`,
`open_obstacle_field`, `rough_local_dynamics`, `small_enclosed_maze`, and
`visual_sensor_stress`.

Every physical line must be nonempty ASCII JSON with no duplicate keys and
must equal its canonical reserialization plus newline. Canonical JSON is
`json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=True, allow_nan=False).encode("ascii")`. The `schema` literal is
exactly `lewm_go2_shared_jepa_v5_raw_supervision_pair_v1`;
`dataset_role` is exactly one of the three frozen roles; `global_row`,
`env_index`, `reset_count`, and `label_shard_row` are nonnegative plain
integers; the scene, family, episode, split, and metadata-only label path are
nonempty strings, and `family` belongs to the frozen eight-family vocabulary;
all declared SHA-256 values are 64 lowercase hexadecimal
characters; current and next endpoint identities differ; `primitive` belongs
to the frozen nine-action vocabulary; and `relative_se2_current_frame` is a
three-element list of finite non-Boolean real values. `global_row` and
`content_sha256` must each be unique across the complete 5,172-row population.
`global_row` is a source identity only: it is not required to equal the
zero-based physical line index and must never determine temporal order.

Remove only `content_sha256`, recompute the canonical self-hash over the other
19 fields, and require exact equality with `content_sha256`. This validation
parses label and relative-pose *metadata already inside the bound pair row*
only for type and self-hash validation; it never dereferences a path, opens an
external label/sidecar/pose/RGB source, or retains those values. Immediately
after self-hash validation retain only this join projection:
`content_sha256`, `dataset_role`, `global_row`, `scene_id`, `family`,
`episode_id`, `env_index`, `reset_count`, `source_split`,
`frames_jsonl_sha256`, `scene_manifest_sha256`, `primitive`,
`current_endpoint_sha256`, and `next_endpoint_sha256`. No other row value may
enter graph construction or the receipt.

Before role filtering, build one endpoint-use table across all 5,172 projected
rows, including calibration. An endpoint may have at most one current-owner
row and at most one next-owner row in the complete population. If it occurs in
both positions, both uses must have exactly the same role, family, scene,
episode, environment, reset, split, frames-source and scene-manifest context.
Any duplicate owner or cross-role/context reuse is an integrity failure.
Calibration uses participate only in this global identity check; calibration
edges are never followed or admitted to a temporal graph.

A recurrent-H4 candidate is an ordered tuple of six distinct pair rows
`(p0, p1, p2, p3, p4, p5)` and seven endpoints `(e0, ..., e6)`, where each
`pk` is the edge `ek -> e(k+1)`, for which:

1. `pk.next_endpoint_sha256 == p(k+1).current_endpoint_sha256` for
   `k = 0, 1, 2, 3, 4`;
2. all six pair rows have exactly the same `dataset_role`, `scene_id`,
   `family`, `episode_id`, `env_index`, `reset_count`, `source_split`,
   `frames_jsonl_sha256`, and `scene_manifest_sha256`;
3. current and next endpoint identities each have unique ownership within the
   complete population, so the graph has neither branching nor ambiguous
   predecessors; and
4. the role is `train` or `checkpoint_selection`.

Exact endpoint identity is the continuity witness. Row order, `global_row`,
timestamps, paths, image hashes, pose, odometry, or approximate equality must
never infer continuity. Reject self-edges, cycles, duplicate current owners,
duplicate next owners, cross-context endpoint reuse, and edges not reachable
from an indegree-zero head. Sort graph heads by the exact tuple
`(dataset_role, family, scene_id, env_index, episode_id, reset_count,
source_split, frames_jsonl_sha256, scene_manifest_sha256,
head_endpoint_sha256)`, then follow each unique successor to build
maximal paths. Report every sliding six-edge/seven-endpoint window. Also
report a row-disjoint lower bound obtained by chunking each maximal path at
edge offsets `0, 6, 12, ...`; leftovers are counted. No alternative alignment,
random choice or filtering is allowed. Chunk membership is not written out.
For a maximal path of `m` edges, its sliding `Hh` count is
`max(m - h + 1, 0)` for each `h` from one through six, its packed H6 count is
`floor(m / 6)`, and its leftover count is `m mod 6`.

Within each eligible sequence, `e0 = o[t-2]`, `e1 = o[t-1]`, and
`e2 = o[t]`; recurrent history is `e0, p0, e1, p1, e2`; and future H4 is
`p2 -> e3 = o[t+1]`, `p3 -> e4 = o[t+2]`, `p4 -> e5 = o[t+3]`, and
`p5 -> e6 = o[t+4]`. Later reversed-past controls must keep `e2`, all four
future actions, and all four future targets fixed.

## Preregistered receipt

The sole output is created exclusively at
`.generated/go2_rgb_causal_belief_state_h4_chain_metadata_census_v1/receipt.json`.
The parent and file must both be fresh; an existing output consumes no input
authority and causes a fail-closed refusal. The receipt is one canonical ASCII
JSON line under schema literal
`lewm_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1_receipt_v1`.
Its top-level fields are exactly `schema`, `status`, `decision`,
`preregistration`, `input_bindings`, `populations`, `integrity`, `graph`,
`adequacy`, `access`, `work`, and `content_sha256`. The first three are strings;
`preregistration`, `input_bindings`, `populations`, `integrity`, `graph`,
`adequacy`, `access`, and `work` are plain JSON objects; and `content_sha256`
is SHA-256 of the canonical JSON encoding of the other eleven fields.
`status` is exactly `COMPLETE`; `decision` is exactly one of
`H4_METADATA_FEASIBLE` or `STOP_H4_METADATA_INADEQUATE`. The exact
nested keys and types are frozen in reviewed source before authorization and
must be source-tested against both feasible and inadequate synthetic cases.

The receipt contains only aggregate counts and canonical hashes:

- all input paths, byte counts, file/content hashes, and expected/reproduced
  row and ordered-content counts;
- missing, duplicate, malformed, self-edge, cycle, branch, ambiguous
  predecessor and uncovered-edge counts;
- cross-role, cross-scene, cross-family, cross-episode, cross-environment,
  cross-reset, cross-split and cross-source-provenance endpoint counts;
- exact sliding H1/H2/H3/H4/H5/H6 graph-window counts and deterministic
  row-disjoint H6 lower bounds by eligible role and family; here H6 names the
  six-edge metadata window, while H4 names its four-edge future target;
- maximal-path length histogram and unused row-disjoint-chunk leftovers;
- primitive histograms at future positions `p2` through `p5` over all sliding
  train H6 windows, with duplicate tuples retained, plus a canonical hash of
  the sorted four-future-action tuple multiset. Every such hash is domain
  separated by role and family before canonical hashing;
- explicit open/read/decode counters for every forbidden category; and
- one terminal decision with every failed adequacy predicate listed.

Malformed, self-edge and uncovered-edge values count pair rows. Duplicate
global-row and pair-content values count extra rows after the first owner.
Duplicate current-owner, duplicate next-owner, and every cross-context value
count unique endpoint identities. Cycles are reported as both component and
edge counts. These identity and context counters cover all three roles;
cycle, uncovered-edge, path, window and packing counters cover only the two
eligible graph roles. Every integrity counter is an integer and must be zero.

No row identity, endpoint identity, row-derived path, RGB commitment, label
commitment, relative pose, timestamp, or chain membership may appear in the
receipt. The only paths allowed are the fixed public preregistration path, the
two exact public input paths, and the one exact output path bound above.

## Adequacy rule

Return `H4_METADATA_FEASIBLE` only if all integrity counts are zero and all of
these deliberately modest falsification thresholds pass:

- at least 64 row-disjoint train H6 sequences;
- at least one row-disjoint train H6 sequence in each frozen family;
- every one of the nine frozen primitives appears at every future position
  `p2`, `p3`, `p4`, and `p5` across all sliding train H6 candidates;
- at least eight row-disjoint checkpoint-selection H6 sequences, with at
  least one in each of the eight frozen selection families; and
- every reported chain remains within one eligible role, scene and family.

Conditional on the required zero-branch, zero-cycle simple-path integrity
predicates, these thresholds respect the metadata-only combinatorial upper
bounds: six edges require five shared internal endpoints, so the known overlap
surpluses bound row-disjoint populations by `floor(747 / 5) = 149` train
sequences and `floor(66 / 5) = 13` checkpoint-selection sequences. Actual
counts remain unknown until the census.

Otherwise return `STOP_H4_METADATA_INADEQUATE`, list the exact failing
predicates and counts, and stop. Inadequacy does not authorize rebuilding,
refining, filtering, rebalancing or resampling data.

## Prohibited access and effects

The census has zero authority to open or decode RGB; open label arrays,
shards, sidecars, raster evidence, frame plans or scene payloads; inspect any
external odometry/pose source beyond type/self-hash validation of the bound
row's existing `relative_se2_current_frame` value; open a schedule, checkpoint,
model, optimizer, trace or
prior runtime output; use GPU; train, qualify or navigate; or access any
held-out, sealed, G2/G8, benchmark, promotion, production or deployment
material. It performs no network access and writes only one fresh public
aggregate receipt after separate source freeze, review and execution
authorization. A binding/schema mismatch is a fail-closed receipt, never a
fallback search.

## Assumptions requiring independent review

- Six pair rows mean six consecutive primitive blocks: two past transitions
  provide the three-observation recurrent history and four future transitions
  provide the two-second nominal H4 prediction horizon.
- Exact shared endpoint identity plus the frozen publisher and validator is a
  sufficient continuity witness; upstream episode metadata need not be
  reopened.
- The 64/8 H6 and per-family thresholds are feasibility gates, not claims of
  training sufficiency. Passing them authorizes only preparation of a separate
  model preregistration.

This document itself authorizes no input opening or execution.
