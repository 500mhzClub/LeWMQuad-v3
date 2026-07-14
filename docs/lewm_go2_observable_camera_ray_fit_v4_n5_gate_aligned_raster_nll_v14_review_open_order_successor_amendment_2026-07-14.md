# Observable Camera-Ray Fit V4 N5 Gate-Aligned Raster-NLL V14 Review Open-Order Successor Amendment

Date: 2026-07-14

Amendment author: `/root`

Status: **source construction and different-agent review only; no exact authority**

## Trigger and terminal V13 review

The source-free V13 amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_strict_review_binding_successor_amendment_2026-07-14.md`

Its file SHA-256 is
`2eaaaa7b896dd42bed02d5a75072d1933b11ad4cce5e8d83f35f1d137ba89633`.

The V13 implementation author froze this source/proof closure:

| Role | Path | SHA-256 |
|---|---|---|
| retained V12 model/loss | `lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py` | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |
| V13 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `e5c03f0ed4a9cb82daeb040c2fe8f87a68911500c47c85992b2780b06f53082f` |
| V13 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `92d6fef2a32498b4dc80566f73422b3735d2d9bbb39612b8a8946d7aa3a34d43` |
| V13 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `7fe1fa1f107478303c10cecd0b591388e1fdb042e14f0ad289f0b36ee399686b` |
| V13 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `77d7782078dc8b089f97144117d7dd0d8d0116dbfbe55a8b665335ee9de55a54` |
| V13 synthetic proof | `lewm/tests/n5_gate_aligned_raster_nll_v13_synthetic_execution.py` | `19c6a1897b247760653c1329e46d389ab7a1b760074967f0e29ace9a19fd36b3` |
| V13 science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `2ebac0d62fa6c67e97ff174b301882cce73bda3b0f11bfa008ef23ff20745596` |
| V13 lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_lifecycle.py` | `d204e5ca88960bc8dc57f3acc328bff2387ca58cc15689624a61c357bc49ea85` |
| V13 handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_implementation_handoff_2026-07-14.md` | `054b64612b02623d6afc8d3c6cb5074a92855f00be007329b04451759b9f0c3d` |

The eligible different-agent review is the canonical terminal BLOCK at
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_independent_review_2026-07-14.json`,
file SHA-256
`55ade66e943e3de1328fc63f536239ae3605f7edd6e8b7aae5a9b09bb33bdc3e`
and canonical content SHA-256
`3125e0ca414d8baf3979cecea0464eee0830738345cf37706420d7d44b335330`.

The reviewer independently passed the V13 suite `226/226`, retained V12 suite
`202/202`, retained V11 suite `190/190`, focused binding matrix `26/26`, real
isolated verifier smoke, source rehashes, compilation, whitespace, and AST
parity. The only blocking finding is
`self_consistent_changed_digest_opens_governed_target_before_rejection`.

A canonical self-consistent review changed the first source digest from
`735563...` to `ffff...`, recomputed its enclosing content and file hashes, and
was rejected after one canonical source target open. The analogous first proof
mutation was rejected after all five source targets and one proof target were
opened. This contradicted V13's literal demand for zero source-target opens for
every invalid review, even though opening a target is the only way to compare
its bytes with a self-consistent caller-supplied digest.

No V13 exact attempt, reservation, output root, checkpoint, RGB/data open, GPU
operation, later rung, or downstream authority exists. The V13 BLOCK is
terminal for the exact V13 bytes and did not consume the sole scientific N5
attempt.

## Preserved scientific and lifecycle contract

V14 is a source-only governance successor. It preserves the V13 scientific
experiment and runtime lifecycle byte-for-value:

- fresh model initialization and no predecessor checkpoint use;
- seed `20260710`, the exact five N5 train-role frames, targets, mappings, and
  frozen schedule;
- the retained hierarchical model and rasterizer, four V11 loss terms, and
  additive exact `0.25 * derived_raster_cell_nll` V12 objective;
- exact gather, float32 epsilon clamp, negative log, all-cell mean, and
  aggregate-only class/family diagnostics;
- batch five, 4,000 updates, 20,000 exposures, AdamW learning rate and weight
  decay `1e-4`, float32/no autocast, clip norm `1`, and final-update selection;
- matched and cyclic wrong-RGB controls, all 26 checks and thresholds, isolated
  verifier, owned-directory transaction, recovery, failure, cleanup, and
  publication semantics; and
- false retry, second-seed, later-rung, training, G2, navigation, held-out,
  runtime, hardware, production, promotion, and deployment licenses.

V14 may not change a tensor operation, loss value, gradient, parameter, data
mapping, schedule, metric, threshold, diagnostic, result meaning, hardware
contract, or exact-attempt count.

## Sole permitted correction: satisfiable review open order

V13 combined two distinct failure classes into an impossible zero-open rule.
V14 freezes their exact separation.

### Phase A: review artifact and structural validation

Preflight may first open only the canonical V14 review artifact supplied with
its caller-bound file SHA-256. Before any governed successor source or proof
target is opened, it must validate the complete review structure and authority
core:

1. canonical JSON plus newline and self-consistent top-level content hash;
2. exact outer schema, status, author/reviewer separation, authority fields,
   predecessor bindings, source/proof outer key sets, and all other fields;
3. every nested source/proof binding is a plain `dict` with exact keys
   `{"path", "file_sha256"}`;
4. both values are plain strings; the path exactly equals its fixed outer key
   and fixed source/proof role; the digest is lowercase 64-hex SHA-256;
5. no missing, extra, duplicate, noncanonical, escaped, aliased, or role-swapped
   path exists; and
6. the complete expected review core equals the parsed core before any
   candidate rehash.

Missing/extra keys, mapping subclasses, nonstrings, malformed digests,
noncanonical paths, duplicates, role swaps, ineligible reviewers, changed
authority fields, and changed digests whose enclosing review hashes were not
recomputed must all reject in Phase A with exactly zero successor-source and
successor-proof target opens.

The parsed nested `path` value is never used as an open path. After equality
with its fixed key is proved, Phase B opens only `ROOT / relative` for entries
drawn from the frozen source and proof tuples.

### Phase B: canonical candidate rehash

A well-formed, canonical, fully self-consistent review can still contain a
wrong but syntactically valid source/proof digest. Such a mismatch is not
detectable without reading the fixed canonical target: no implementation can
embed its own future file hash, and the handoff/source pair cannot provide a
non-self-referential closed digest authority.

Only after Phase A passes may preflight open and hash candidates in this fixed
order:

1. all V14 successor sources in their frozen tuple order;
2. all V14 successor proofs in their frozen tuple order.

Each target is a no-follow, singly linked regular file beneath the canonical
repository root and is read through the retained descriptor-safe reader. The
first file-hash mismatch rejects immediately. Therefore a self-consistent wrong
digest may cause exactly the fixed prefix of canonical candidate reads needed
to reach the mismatch. It may never cause an open of a caller-selected path,
data, RGB, checkpoint, GPU, exact output, `.generated` payload, or any file
outside the frozen tuples.

Tests must assert exact prefix counts for a wrong digest at the first, middle,
and final source and proof positions. They must separately spy data, RGB,
checkpoint, GPU, output, and `.generated` openers and require zero calls for
every Phase A and Phase B rejection.

This is the only V14 behavior change. It resolves the V13 contradiction; it
does not relax the exact nested schema or allow a wrong digest to pass.

## V14 source and proof namespace

Fixed implementation author:
`/root/camera_v12_gate_aligned_implementer`.

The retained model/loss remains the exact V12 file and hash above. The V14
production closure is:

1. `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py`;
2. `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py`;
3. `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py`;
4. `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py`.

The V14 proof closure is:

1. `lewm/tests/n5_gate_aligned_raster_nll_v14_synthetic_execution.py`;
2. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py`;
3. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_lifecycle.py`;
4. `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_implementation_handoff_2026-07-14.md`.

Canonical review:
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_independent_review_2026-07-14.json`.

The only possible exact output root is
`.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v14/`.

The V14 policy must bind and validate the V13 amendment, complete V13
source/proof/handoff closure, and terminal canonical V13 BLOCK above. It may
retain all earlier V12/V11 and lifecycle bindings transitively, but it may not
alter their bytes or reinterpret their terminal status.

## Required author and reviewer proof

All source tests are CPU-only, hide every accelerator selector, set every
native numerical thread count to one, and use only synthetic temporary roots.
They must not open canonical RGB/data, `.generated` payloads, predecessor
checkpoints, or GPUs.

The implementation author and different-agent reviewer must:

1. rehash this amendment, V13 amendment/closure/handoff/BLOCK, and all retained
   V12/V11 bindings;
2. mechanically compare normalized V14/V13 production ASTs, permitting only
   namespace, terminal-predecessor authority, canonical-path, and the clarified
   review/open-order changes;
3. prove a valid minimal review passes unchanged;
4. reproduce every V13 nested source/proof schema attack with recomputed
   enclosing hashes and zero candidate opens;
5. test changed well-formed digests without recomputing enclosing hashes and
   require Phase A rejection with zero candidate opens;
6. test self-consistent wrong digests at the first, middle, and final source
   and proof positions, require exact fixed-prefix candidate reads, and require
   immediate rejection at the mismatch;
7. prove every opened candidate is selected from the frozen tuple and opened
   as `ROOT / relative`, never from a nested caller path;
8. spy data, RGB, checkpoint, GPU, output, and `.generated` openers and require
   zero calls on every invalid review;
9. rerun the full V14 suite plus retained V13 `226`, V12 `202`, and V11 `190`
   suites, exact loss/gradient/parity/diagnostic proofs, actual isolated child,
   all 26 checks, and lifecycle fault injections; and
10. run compilation, LF/final-newline/whitespace, source identity, absence,
    author separation, no-authority, and no-output-root checks before handoff or
    review publication.

The V14 reviewer must have a `/root/` identity and differ from `/root`, the
amendment author, fixed implementation author, V12 reviewer, V13 reviewer
`/root/camera_v13_independent_review`, and future exact execution agent. The
implementation author may not self-review.

## One-attempt lifecycle and non-authority

Only a canonical different-agent V14 `PASS` binding every exact frozen byte may
authorize one future fresh V14 N5 attempt. The future exact execution agent
must differ from the implementation author and V14 reviewer. The attempt runs
once on discrete GPU0 R9700 only, keeps the Raphael iGPU unused, and is
serialized with every `.generated` mutator.

No retry is permitted after success, numerical failure, runtime failure,
verification failure, publication failure, timeout, or interruption. A full
unchanged 26-check PASS may license only a later source-free ladder design and
review. It cannot by itself authorize later-rung execution, checkpoint reuse,
Shared-JEPA training, G2, selection, calibration, navigation, held-out,
runtime, hardware, production, promotion, or deployment.

This amendment grants only V14 source construction and different-agent source
review. It grants no exact execution, data/RGB/checkpoint open, GPU operation,
retry, later rung, training, G2, navigation, held-out, runtime, hardware,
production, promotion, or deployment authority.
