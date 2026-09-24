# Go2 N32 dynamic cell-square geometry binding

Date: 2026-07-11

Status: frozen before any dynamic-geometry diagnostic label-shard open,
diagnostic result write, successor model output, G2, runtime, held-out, or
sealed access.

## In-place amendment: finalizer candidate-hash bootstrap

This amendment was frozen before implementation manifests, candidate output,
or any label-shard byte open. The pre-amendment SHA-256 was
`fafdd7ae130ba80e6bf1f109666c8fa676836c178291cdebb0bd9a4cecebfcd8`.
It resolves the future candidate-file hash and controls conflicting argument
and placeholder wording below.

The runner still accepts only
`--implementation-manifest-sha256 <hash>`. The finalizer accepts exactly two
required hash-only arguments in this order:

```text
--implementation-manifest-sha256 <verified machine-manifest file SHA-256>
--candidate-sha256 <verified runner-candidate file SHA-256>
```

Both paths remain fixed in source. The machine manifest's finalizer
`authorized_read_path_template` contains, in addition to the machine-manifest
self entry defined below, exactly one candidate entry:

```json
{"path":".generated/go2_dynamic_cell_square_projection_diagnostic/v1/candidate.json","role":"candidate","sha256_source":"candidate_sha256_argument"}
```

The stdlib finalizer bootstrap hashes the fixed candidate path before any
repository import and requires equality to the supplied candidate hash. It
then substitutes the verified value into only that template entry and records
the actual entry/path-set hash in the finalizer ledger. The candidate hash must
also equal the `candidate.file_sha256` stored in the final result.

Thus the runner phase permits exactly one late hash source (machine manifest)
and the finalizer phase permits exactly two (machine manifest and candidate).
No path placeholder, third hash source, environment-derived value, or
unverified substitution is permitted. Synthetic tests must reject a missing,
extra, reordered, malformed, incorrect, or unverified argument and any
candidate changed between bootstrap hashing, parsing, and final publication.

## In-place amendment: manifest self-entry allowlist template

This amendment was frozen before the implementation manifests and before any
label-shard byte open. The pre-amendment SHA-256 was
`74fbe4cacd61c2f6dfa0d823fade8dc162938f99ecde16d880133f60e9563d9f`.
It resolves the last machine-manifest self-reference and controls conflicting
wording below.

The machine manifest cannot contain its own final file SHA-256. For each
runtime phase it therefore freezes an ordered `authorized_read_path_template`
whose exact machine-manifest entry is:

```json
{"path":"docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.json","role":"machine_manifest","sha256_source":"implementation_manifest_sha256_argument"}
```

Every other entry is exact `{path,role,sha256}`. The manifest stores the
canonical template-list SHA-256, exact expected roles, and all numeric open
counts; it does not claim the future actual path-set hash. At runtime the
stdlib bootstrap first hashes the fixed machine-manifest path and requires it
to equal the sole supplied `--implementation-manifest-sha256` value. It then
replaces only the declared self-entry with
`{path,role,sha256:<supplied-and-verified-hash>}`, computes the actual ordered
path-set SHA-256, and stores the actual records/hash in the candidate/final
phase ledger. The finalizer independently performs the same substitution.

Outside the finalizer candidate entry frozen above, no other placeholder,
late-bound path, hash source, environment value, or template substitution is
permitted. Synthetic tests must reject a literal self-hash claim, the wrong
self/candidate role or path, substitution before verifying file bytes, or any
difference between the instantiated list and the stored phase ledger.

## In-place amendment: candidate/final schemas and non-leaking evidence

This amendment was frozen after the second independent review and before any
label-shard byte open. The pre-amendment SHA-256 was
`091e7d3d51ad58e6e64ad23814f01792a675a59d4ea7cfe3c80ed59647c92e7d`.
This section controls every conflict with the body below.

### Execution sequence

There are two immutable generated artifacts:

1. the runner exclusively creates
   `.generated/go2_dynamic_cell_square_projection_diagnostic/v1/candidate.json`;
2. after independently recomputing every scientific value, the finalizer
   exclusively creates
   `.generated/go2_dynamic_cell_square_projection_diagnostic/v1/result.json`.

The runner candidate contains only the preparation and runner ledgers. The
final result binds the candidate file/content hashes and contains separate
preparation, runner, and finalizer ledgers. The finalizer never edits or
replaces the candidate. Failure before a valid final result leaves the
authoritative result path absent.

Neither passing artifact may contain a scene ID, global row, label row, image
hash, shard-row tuple, quaternion, yaw, Cartesian row/column identity, ordered
violation record, or static/dynamic remaining-identity record. Predecessor
parity and remaining support use counts plus canonical identity-list SHA-256
only. If a gate fails, exact identities may be written exclusively to
`.generated/go2_dynamic_cell_square_projection_diagnostic/v1/failure_diagnostic.json`.
That file is diagnostic-only, sets every model/runtime/promotion license false,
and may not be an input to a passing successor binding.

### Canonical bytes and exact schemas

Candidate and final JSON bytes use exactly:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("utf-8") + b"\n"
```

Every JSON object has the exact key set declared here. Every integer is an
exact JSON integer and not boolean; every boolean is exact; every SHA-256 is a
64-character lowercase hexadecimal string; every count is nonnegative; and
every list has the frozen order. `content_sha256` is the SHA-256 of the same
object omitting only its own `content_sha256` key and without the trailing
newline. No other key is omitted or normalized.

The candidate schema literal is
`lewm_go2_dynamic_cell_square_projection_candidate_v1` and its exact top-level
keys are:

```text
schema
created_at_utc
execution_binding
implementation_manifests
inputs
source_map
scope
preparation_access_ledger
runner_access_ledger
label_reconciliation
support
family_class_rows
frame_summary_records_sha256
scientific_core_sha256
gates
content_sha256
```

The final schema literal is
`lewm_go2_dynamic_cell_square_projection_diagnostic_v1` and its exact
top-level keys are:

```text
schema
created_at_utc
execution_binding
implementation_manifests
candidate
inputs
source_map
scope
preparation_access_ledger
runner_access_ledger
finalizer_access_ledger
label_reconciliation
support
family_class_rows
frame_summary_records_sha256
scientific_core_sha256
independent_recomputation
gates
content_sha256
```

`execution_binding` is exactly `{path,file_sha256}`.
`implementation_manifests` is exactly `{human,machine}`, where `human` is
`{path,file_sha256}` and `machine` is `{path,file_sha256,content_sha256}`.
`candidate` is exactly `{path,file_sha256,content_sha256}`. `source_map` is
exactly `{entries,entry_count,source_map_sha256}`; each ordered entry is exactly
`{path,role,sha256}`, roles and paths are unique, and the map hash covers only
the ordered `entries` list.

`inputs` is exactly
`{predecessor_report,predecessor_result,dynamic_geometry,label_shard_manifest,selected_targets}`.
The first and third records are `{path,file_sha256}`; `predecessor_result` is
`{path,file_sha256,content_sha256}`; `label_shard_manifest` is
`{entry_count,manifest_sha256}`; and `selected_targets` is
`{frame_count,byte_count,sha256}`.

`scope` is exactly
`{dataset_role,learning_performed,frame_count,transition_count,families,endpoint_sides,class_order,forbidden_roles}`.
Its fixed values are train, false, 320, 160, the five registered family names,
current then next, unknown/free/occupied, and the sorted forbidden-role list
`[g2,heldout,image,model_output,physical_nontrain,runtime_result,sealed,selection_calibration,source_geometry]`.

`label_reconciliation` is exactly
`{byte_count,byte_sha256,class_totals,known_total,per_frame_cell_count,per_frame_count,per_frame_totals_sha256,all_counts_reconcile}`.
`class_totals` is exactly `{unknown,free,occupied,all}` with the frozen values
below. `all_counts_reconcile` must be true.

`support` is exactly
`{level_center,level_cell_square,static_cell_square_known,dynamic_cell_square_known}`.
`level_center` is exactly
`{support_cell_count,support_mask_sha256,free_total,free_supported,occupied_total,occupied_supported,known_violation_count,known_violation_identities_sha256}`.
`level_cell_square` is exactly `{support_cell_count,support_mask_sha256}`.
Each static/dynamic known record is exactly
`{known_total,supported_count,unsupported_count,unsupported_free_count,unsupported_occupied_count,unsupported_frame_count,unsupported_identities_sha256}`.

`family_class_rows` contains exactly ten rows ordered by registered family then
FREE before OCCUPIED. Each exact row is
`{family,class_id,class_name,total,level_center_supported,static_cell_square_supported,dynamic_cell_square_supported}`.
`frame_summary_records_sha256` hashes 320 ephemeral records in canonical frame
order; each record is exactly
`{family_rank,frame_rank,unknown_total,free_total,occupied_total,level_center_free_supported,level_center_occupied_supported,static_free_supported,static_occupied_supported,dynamic_free_supported,dynamic_occupied_supported}`.
The records themselves are not persisted.

`gates` is exactly
`{binding_and_source_hashes_pass,predecessor_authority_pass,label_manifest_and_bytes_pass,label_count_reconciliation_pass,level_center_parity_pass,level_cell_square_frozen_pass,static_all_known_scored_pass,dynamic_all_known_scored_pass,dynamic_zero_known_unsupported_pass,access_reconciliation_pass,independent_recomputation_pass,all_passed}`.
The candidate sets `independent_recomputation_pass` false and therefore
`all_passed` false; this is not a scientific failure. The final result requires
every gate true. No other authorization/license key exists in either artifact.

The scientific core is exactly
`{label_reconciliation,support,family_class_rows,frame_summary_records_sha256}`.
`scientific_core_sha256` hashes that object. `independent_recomputation` is
exactly
`{candidate_scientific_core_sha256,finalizer_scientific_core_sha256,exactly_equal}`,
and the final result requires both hashes equal and `exactly_equal` true.

### Phase allowlists, imports, and ledgers

Repository code must be authorized and hashed before it is imported. The
runner and finalizer begin as stdlib-only bootstraps, validate the binding and
reviewed machine manifest, build their exact lexical/resolved allowlists, hash
every required source, and only then add the repository to `sys.path` or import
the pure geometry/diagnostic implementation. Dependency imports do not
authorize repository paths.

To avoid a manifest/source self-hash cycle, the runner accepts exactly one
required manifest-hash argument and the finalizer accepts that argument plus
the candidate-hash argument frozen above. The implementation manifest and
candidate paths are fixed in source and cannot be supplied. The reviewed human
manifest is written first and does not claim the later machine-manifest file
hash; the machine manifest binds the human-manifest hash and all runner/
finalizer source hashes. The exact machine-manifest file hash is then frozen in
the reviewed execution commands and supplied through the frozen hash-only
arguments. No other argument, environment override, alternate path, or output
selector is accepted.

The preparation allowlist contains exactly the binding, predecessor report and
result, pure geometry, diagnostic core, runner, finalizer, preparation source,
all focused tests, and the two manifest destinations. It permits metadata
existence/size checks for the 20 committed shard paths but zero shard bytes.
The preparation manifest records the pre-amendment metadata observation: one
existence/size check per committed shard path, no content open.

The runner read allowlist contains exactly the binding, both reviewed
manifests, predecessor result, pure geometry, diagnostic core, runner source,
all source-map tests, and the 20 committed label shards. Its only write path is
the candidate. The finalizer read allowlist contains exactly the binding, both
manifests, predecessor result, candidate, pure geometry, finalizer source, all
source-map tests, and the same 20 shards. It does not import the diagnostic
core or runner. Its only write path is the final result, or the failure-only
diagnostic when a scientific gate fails.

Every phase ledger has exact keys:

```text
schema
phase
authorized_read_paths
authorized_read_path_set_sha256
authorized_write_paths
authorized_write_path_set_sha256
role_byte_open_counts
label_shard_pre_hash_byte_opens
label_shard_post_hash_byte_opens
label_shard_npz_parses
array_decompression_counts
selected_label_rows_read
unselected_rows_scored
unselected_rows_retained
metadata_only_shard_stats
denied_attempt_records
denied_reason_counts
unexpected_path_attempts
forbidden_role_open_counts
all_counts_reconcile
```

The ledger schema is `lewm_go2_dynamic_projection_access_ledger_v1`.
`authorized_read_paths` and `authorized_write_paths` are sorted exact
`{path,role,sha256}` records (write records use `sha256:null` before creation);
their hashes cover those lists. `role_byte_open_counts` has exactly the roles
in the corresponding read allowlist; every non-shard authorized file is opened
exactly once and every unlisted role count is impossible. Preparation has zero
shard byte opens/parses/array decompressions and exactly 20 metadata-only shard
stats. Runner and finalizer each have 20 pre-hash byte opens, 20 post-hash byte
opens, 20 in-memory NPZ parses, 40 total label-array decompressions expressed
as exactly `{current_labels:20,next_labels:20}`, 320 selected rows, and zero
metadata-only stats, unselected scoring, or retained rows. Finalizer additionally
opens the candidate exactly once. The machine manifest freezes each ordered
allowlist template, role set, numeric role-open map, and template-list hash
before execution. The sole self-entry substitution above produces the actual
path-set hash recorded by the phase ledger.

`denied_reason_counts` has exact ordered keys
`path_alias_or_escape,symlink_component,outside_repository,unallowlisted,forbidden_role,modality_mismatch,hash_mismatch`.
When more than one applies, that order selects the primary reason. A denied
record is exactly
`{requested_role,declared_role,modality,lexical_path,resolved_path,primary_reason}`;
`resolved_path` is null when safe resolution was impossible. Passing phases
require no denied records, every denial count zero, zero unexpected attempts,
and exact-zero `forbidden_role_open_counts` for every scope forbidden role.
Hash mismatches fail after their authorized byte open is counted and can never
be reclassified as an unlisted path.

## In-place amendment: full-label execution and independent finalization

This amendment was frozen after independent source review found that the
pre-amendment document did not fully specify its executable trust boundary,
and before any label-shard byte open or diagnostic result. The pre-amendment
SHA-256 was
`6e515534f3fc3d13276720a0a129d2863324582b9cb2275369b7074f1c50f563`.
No runner or test may open a label shard until a reviewed implementation
manifest binds this amended document.

The authoritative result schema is exactly
`lewm_go2_dynamic_cell_square_projection_diagnostic_v1`. The final result must bind
this document's final file SHA-256, the predecessor report and result hashes,
the pure geometry hash, the label-shard manifest and selected-target-byte
hashes, the complete reviewed source map, and preparation, runner, and
finalizer access ledgers. The result may report only the predecessor violation
count and canonical identity-list hash as a parity check. It must never persist
or use those records as the cells selected for static or dynamic scoring. Both
support arms must iterate all 1,310,720 selected label cells in canonical
frame/row/column order.

Before the runner, create and review both:

- `docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.md`;
- `docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.json`.

The machine manifest must bind exact hashes for this binding, the pure
geometry, diagnostic core, runner, independent finalizer, and every focused
test; record the interpreter and dependency versions; reproduce all frozen
input hashes; prove the exclusive candidate and result paths are absent; and include a
metadata-only preparation ledger. The human manifest must record independent
review findings and the exact verification commands/results. Any source or
test hash change after review invalidates authorization and requires a new
manifest before label access.

The executable source-map roles and paths are exactly:

```text
dynamic_geometry  lewm/benchmarks/go2_dynamic_cell_square_projection.py
diagnostic_core   lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py
preparation       scripts/prepare_go2_dynamic_cell_square_projection.py
runner            scripts/diagnose_go2_dynamic_cell_square_projection.py
finalizer         scripts/finalize_go2_dynamic_cell_square_projection.py
geometry_test     lewm/tests/test_go2_dynamic_cell_square_projection.py
diagnostic_test   lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py
preparation_test  lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py
finalizer_test    lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py
```

No missing, extra, duplicate, aliased, or substituted source-map role is
permitted. The manifest may separately bind this binding and the two manifests
as document roles; they are not executable-source entries.

The runner has no path-selection arguments. Its only output is the exact
candidate path frozen above. Before any input byte open it constructs a lexical and resolved
allowlist containing only:

- this binding, the reviewed human and machine manifests, and their exact
  hashes;
- the exact predecessor result path and hash;
- the exact pure geometry, diagnostic core, runner, and focused test sources
  named by the source map;
- the 20 exact lexical NPZ paths and hashes in the predecessor's committed
  label-shard manifest;
- the exclusive candidate path as a write-only destination.

Every input path must be absolute after anchoring at the repository root,
lexically canonical, below that root, not a symlink at any path component, and
equal to its expected resolved path. Aliases, `..`, alternate roots, arbitrary
output paths, and paths whose semantic role is missing, nontrain, selection,
calibration, G2, runtime, held-out, or sealed fail before byte access. Denied
attempts are recorded by requested role, declared role, modality, lexical
path, resolved path if safely available, and primary reason.

The predecessor JSON uses a strict UTF-8 parser that rejects duplicate object
keys, nonfinite constants, non-exact JSON container/scalar types, malformed
SHA-256 values, unexpected fields in every consumed compact record, and any
content-hash mismatch. The runner must validate the complete committed
label-shard manifest and its SHA-256 before allowlisting a shard.

Each NPZ is read into immutable bytes only after its pre-read SHA-256 passes.
Its ZIP inventory must contain exactly these 12 root-level, unique,
unencrypted NPY members, with matching central/local names and no directory,
absolute, parent, or path-qualified member:

```text
current_labels.npy
current_supervision_mask.npy
next_labels.npy
next_supervision_mask.npy
current_observed_mask.npy
next_observed_mask.npy
relative_se2_current_frame.npy
primitive.npy
current_image_path.npy
next_image_path.npy
current_image_sha256.npy
next_image_sha256.npy
```

Only `current_labels` and `next_labels` may be decompressed. Each is C-order
`uint8` with shape `(storage_rows,64,64)` and values only 0, 1, or 2. The
selected `(shard,row,side)` tuples must be unique, in range, and match the
committed family, scene, global row, side, and label row. No unselected value
may enter a metric or be retained after its shard is released. Every shard is
rehash-read after parsing; any pre/post difference fails. Archive bytes,
arrays, and selected copies are released before the next shard is opened.

The runner's exact label-access ledger is:

- 20 allowlisted shards;
- 20 pre-parse hash byte opens and 20 post-parse hash byte opens;
- 20 NPZ parses;
- 40 label arrays decompressed;
- 320 selected endpoint rows copied;
- 0 supervision, observed-mask, odometry, primitive, image-path, or image-hash
  arrays decompressed;
- 0 retained or scored unselected rows;
- 0 denied, unexpected, changed, or forbidden accesses.

The selected rows are ordered only by the predecessor's frozen canonical frame
order and concatenated as exact C-order uint8 targets. Require byte count
1,310,720 and SHA-256
`6952c1f9604da1d9fd4c94a3f33deb142451836609b7059970ff6c459737ce05`.
Before any support result is accepted, require exact aggregate class totals:

- UNKNOWN (0): 1,181,699;
- FREE (1): 118,793;
- OCCUPIED (2): 10,228;
- all classes: 1,310,720;
- known total: 129,021.

Every frame must contain exactly 4,096 cells; per-frame, per-family, per-class,
and aggregate counts must reconcile exactly to those denominators. The runner
must independently rebuild the level centre mask over all cells. Applying it
to all labels must reproduce centre-supported FREE 118,792 and OCCUPIED 9,856,
with exactly 373 ordered known violations whose identities and hash equal the
predecessor report. This is parity evidence only. Static cell-square and
dynamic full-attitude scoring then independently scan every known cell.

The independent finalizer is a separate stdlib source and does not import the
runner or diagnostic core. It revalidates the binding, manifests, result
schema/content hash, allowlists, strict predecessor JSON, NPZ inventory, all 20
pre/post shard hashes, selected-byte stream, class totals, frame identities,
camera inputs, level mask, full static/dynamic support loops, every breakdown,
every remaining identity, all gates, and all access ledgers. It may import only
the frozen pure geometry module for the registered projection primitive. Its
label ledger has the same exact counts as the runner and is stored separately;
runner and finalizer accesses are never merged. It exclusively writes the
authoritative final result; its canonical stdout summary is hashed into the
dated result report.

Preparation may open/hash only this binding, predecessor report, reviewed
source/tests, and predecessor result metadata needed to construct the exact
allowlist. It must record zero label-shard, RGB, model, G2, runtime, held-out,
and sealed opens. The runner and finalizer each record zero RGB/image,
checkpoint/model output, source-geometry, selection/calibration/nontrain, G2,
runtime-result, held-out, and sealed opens. A self-declared empty forbidden
list is insufficient: every ledger counter and reconciled path set is explicit
and independently checked.

## Purpose

The independently finalized camera-frustum audit established that the
observable-physical-v3 labels are reconstructible and physically observable,
but that mapping each Cartesian cell centre to one polar bin omits 373 known
label occurrences. This binding tests the smallest source-grounded successor:
the centre and four closed corners of each 0.10 m output cell, at the five
registered vertical anchors, projected with the full deployment-available body
attitude and fixed camera extrinsic.

This is a geometry and label-compatibility diagnostic only. It performs no
learning and cannot pass N32, G2, runtime, or any promotion gate.

## Frozen predecessor

The controlling predecessor is
`docs/lewm_go2_n32_camera_frustum_observability_audit_v2_result_2026-07-11.md`,
SHA-256
`8bfb4c9a8b69f67b3b9e4d6e3b21e9ff89ecaff89a2bab3eb83d759ca4fe6d22`.
The immutable finalized runner result may be reopened only as exact
post-finalization research evidence at:

- path:
  `.generated/go2_n32_camera_frustum_observability_audit/v2/result.json`;
- file SHA-256:
  `7725ecddf2fa77bb762733fd35df2efd2fb60d4f9aa8ab6fdf2bee660522909e`;
- canonical content SHA-256:
  `11420607d2c4f8e79af9214d43bbc6259669ee84c9ccc0aaefd4167cc1d809a1`.

It is not a training input. A successor model binding may consume only this
dated diagnostic report and frozen source hashes, not mine the runner payload
or violation identities for hyperparameter selection.

## Pre-output correction

Before this freeze, a source-only implementation reproduced these already
known predecessor facts in memory: 373 centre-support known-label violations
at 52 unique cells, static cell-square recovery of 369/373, and full-attitude
cell-square recovery of the remaining 4/4 occurrences. No diagnostic result
file was written and no label shard, RGB, model, G2, runtime, held-out, or
sealed payload was opened.

That check was insufficient: a dynamic camera can gain support at one frustum
edge while losing level-camera support at another. A geometry-only inclusion
check found that two recorded high-attitude frames do not contain the complete
1,990-cell level centre mask inside their dynamic cell-square masks. This does
not establish a label failure because the labels use the recorded camera, but
it invalidates the shortcut from `4/4 recovered` to `all known labels
supported`. The authoritative diagnostic below must therefore score every
known label cell directly. The earlier in-memory subset result is debugging
evidence only.

## Frozen geometry

The pure stdlib implementation is
`lewm/benchmarks/go2_dynamic_cell_square_projection.py`, SHA-256
`ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf`.

The exact contract is:

- output grid: 64 x 64, rows forward and columns left;
- cell size: 0.10 m;
- centre ranges: forward `[-0.95, 5.35]` m and left
  `[-3.15, 3.15]` m;
- horizontal support, in deterministic order: centre, then
  `(-0.05,-0.05)`, `(-0.05,0.05)`, `(0.05,-0.05)`,
  `(0.05,0.05)` m;
- vertical anchors: `(-0.333,-0.133,0.067,0.267,0.467)` m;
- camera horizontal/vertical FOV: `78.323` / `62.8370386364` degrees;
- near plane: 0.05 m, with inclusive near and image-plane boundaries;
- fixed body mount: xyz `(0.326,0,0.043)` m, RPY `(0,0,0)`;
- input attitude: finite deployment-style `base_quat_world_xyzw` plus finite
  stored base yaw, with norm and wrapped-yaw residual at most `1e-5`;
- compose the standard raw XYZW body-to-world rotation without quaternion
  renormalization, then express the camera in the stored-yaw-aligned frame;
- a Cartesian query is supported iff any of its 25 registered
  horizontal/vertical support points lies inside the closed rectilinear
  frustum;
- no body footprint, configuration inflation, privileged map, scene identity,
  label, target, or simulator world pose is an inference input.

The level-attitude mask must contain exactly 2,062 cells and have row-major
uint8 SHA-256
`4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b`.
The independently rebuilt level centre-only mask must contain exactly 1,990
cells and have row-major uint8 SHA-256
`026d7654864bea7ae0545bd6448f6def64519a3bedcbc7ea747e7b4b95f82b3a`.

At deployment, the equivalent attitude must come from calibrated IMU/
proprioception plus the fixed camera mount. Recorded simulator world pose is
authorized only as train-role audit metadata and supervision for parity tests.

## Frozen full-label input

The diagnostic may reopen only the 20 train-role label shards committed by the
finalized result's `label_shard_manifest`:

- entry count: 20;
- manifest SHA-256:
  `998ce5a768029c23c931fbbec730c1fe31b9ed1fe155494fc68f34a0c23d3d1b`;
- selected scope: 160 transitions, current and next endpoints, 320 frames;
- canonical selected target bytes: 1,310,720 row-major uint8 bytes;
- selected target-byte SHA-256:
  `6952c1f9604da1d9fd4c94a3f33deb142451836609b7059970ff6c459737ce05`.

Each shard path and SHA-256 must come from that exact manifest. The runner and
finalizer must each hash-read every shard before parsing, parse it exactly once
in memory, read only the exact selected row/side label arrays named by the
manifest, reconstruct the canonical frame order, reproduce the selected
target-byte hash, and hash-read every shard again after parsing. Any missing,
duplicate, extra, malformed, re-ordered, or changed row fails closed.

The diagnostic may inspect supervision member names only in the strict ZIP
inventory. It must not decompress supervision or any non-label array, and may
not open images, checkpoints, model outputs, source geometry,
selection/calibration/nontrain roles, G2, runtime results, held-out data, or
sealed paths.

## Immutable gates

The result must report aggregate and per-family/per-class counts for all 320
frames and pass every gate below:

1. finalized predecessor file/content hashes, source hashes, camera parity,
   reconstruction, and access reconciliation remain exact;
2. label-shard manifest, all 20 shard hashes, 320 frame identities, and the
   canonical selected target-byte hash remain exact;
3. level cell-square count and mask hash remain 2,062 and the frozen hash;
4. the predecessor centre-support counts reproduce exactly: FREE
   `118792/118793` and OCCUPIED `9856/10228` supported, with 373 ordered known
   violations whose identity-list SHA-256 is
   `f85a9ece8f4a34fe0f175de900934780a750d076f70a7e672be8337cffb64bcc`;
5. static level-attitude cell-square support is scored over every known target,
   not only predecessor violations, and supports exactly 129,017/129,021 known
   cells with zero FREE and four OCCUPIED unsupported across four frames; its
   remaining identity-list SHA-256 is
   `c574f35890ef68114fb36ebf701eec7552262d03c49cf4d1c07b47740fc505f0`;
6. dynamic full-attitude cell-square support is scored over every known target;
7. every FREE and every OCCUPIED target is dynamically supported in every
   family and frame; required dynamic unsupported count is exactly zero and
   the empty identity-list SHA-256 is
   `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`;
8. all forbidden access counters, denied attempts, duplicate identities, and
   post-read hash changes are zero.

Failure of gate 7 rejects this geometry before training. Labels, thresholds,
losses, family gates, optimizer exposure, and support points may not be changed
in response; the next action must diagnose the exact source-grounded mismatch.

## Output and next license

The only normal generated outputs are the immutable candidate and final result
paths frozen above. The failure-only quarantine path is permitted only on a
failed scientific gate. All use exclusive creation and canonical JSON with an
embedded content hash. The dated human result report binds both normal files,
the finalizer summary, exact source map, and all phase ledgers. Independent
source review and focused tests must pass before the runner is executed.

If and only if every gate passes, the result licenses a separate pre-output
N32 model binding for one dynamic cell-square Cartesian evidence head. It does
not license G2, memory fusion, closed-loop execution, held-out access, or any
claim of learned generalization.
