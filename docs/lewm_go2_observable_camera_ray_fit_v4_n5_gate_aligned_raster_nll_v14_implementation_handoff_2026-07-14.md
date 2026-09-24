# Camera-ray N5 gate-aligned raster NLL V14 implementation handoff

Date: 2026-07-14

Implementation author: `/root/camera_v12_gate_aligned_implementer`

Status: **source and synthetic CPU closure complete; independent review required; no exact authority**

## Frozen authority

The source-free V14 amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_review_open_order_successor_amendment_2026-07-14.md`

File SHA-256:

`39e9f840ede8f245d850b7eaaedf0a007fb5f083923629850ced11c8055cd1f6`

The amendment author is `/root`. The fixed implementation author matches the
amendment. A canonical reviewer must start with `/root/` and differ from
`/root`, this implementation author, the V12 reviewer
`/root/raw_v11_builder_auditor_diff`, the V13 reviewer
`/root/camera_v13_independent_review`, and the future exact execution agent.
The implementation author did not write a canonical review.

## Frozen production closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| retained V12 model/loss | `lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py` | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |
| V14 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py` | `7290d0021d04a3408d36bebede7a7726d764320ff7083fedbb958c98f9069f5a` |
| V14 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py` | `05c1da21db799e224b1623f032ac15e42b445a55812d5c06e16d31fe8d0701f4` |
| V14 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py` | `5393dc04ada6ce77a49fa4506235fb431567712d4472db58a74e8ea1761f7d04` |
| V14 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py` | `e1ab693555467202a2fa6a84f67e6a076313aa241d703c89df3a1adab19b7f51` |

No V14 model/loss copy exists. V14 binds and imports the retained V12
model/loss at the exact hash above.

## Frozen proof closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| synthetic lifecycle and native V14 gate fixture | `lewm/tests/n5_gate_aligned_raster_nll_v14_synthetic_execution.py` | `5a2f8cf73607cb61c870016d3881875038e13c345490a9e0bd1f97d9b2ae1887` |
| V14 loss, parity, gradient, and diagnostic tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py` | `c540dd83143fe33c94275c39090a99725d635acc476336d44022e75d992fd702` |
| V14 lifecycle, review-order, schema, gate, and subprocess tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_lifecycle.py` | `120d6f8334b36b59e6b3fd5f47210c901ff3ae30817e1882383c8be630effb9b` |

This handoff is the fourth proof file. The independent reviewer must hash its
final bytes and bind that hash in the canonical review.

## Terminal V13 closure

V14 binds the exact V13 amendment, complete source/proof/handoff closure, and
terminal different-agent BLOCK:

| Role | Path | File SHA-256 |
| --- | --- | --- |
| V13 amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_strict_review_binding_successor_amendment_2026-07-14.md` | `2eaaaa7b896dd42bed02d5a75072d1933b11ad4cce5e8d83f35f1d137ba89633` |
| retained model/loss | `lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py` | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |
| V13 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `e5c03f0ed4a9cb82daeb040c2fe8f87a68911500c47c85992b2780b06f53082f` |
| V13 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `92d6fef2a32498b4dc80566f73422b3735d2d9bbb39612b8a8946d7aa3a34d43` |
| V13 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `7fe1fa1f107478303c10cecd0b591388e1fdb042e14f0ad289f0b36ee399686b` |
| V13 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `77d7782078dc8b089f97144117d7dd0d8d0116dbfbe55a8b665335ee9de55a54` |
| V13 synthetic proof | `lewm/tests/n5_gate_aligned_raster_nll_v13_synthetic_execution.py` | `19c6a1897b247760653c1329e46d389ab7a1b760074967f0e29ace9a19fd36b3` |
| V13 science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `2ebac0d62fa6c67e97ff174b301882cce73bda3b0f11bfa008ef23ff20745596` |
| V13 lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_lifecycle.py` | `d204e5ca88960bc8dc57f3acc328bff2387ca58cc15689624a61c357bc49ea85` |
| V13 handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_implementation_handoff_2026-07-14.md` | `054b64612b02623d6afc8d3c6cb5074a92855f00be007329b04451759b9f0c3d` |
| terminal V13 review BLOCK | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_independent_review_2026-07-14.json` | `55ade66e943e3de1328fc63f536239ae3605f7edd6e8b7aae5a9b09bb33bdc3e` |

The V13 review canonical content SHA-256 is
`3125e0ca414d8baf3979cecea0464eee0830738345cf37706420d7d44b335330`.
Its terminal status is
`blocked_changed_digest_zero_open_contract_unsatisfied`, and its sole finding
is
`self_consistent_changed_digest_opens_governed_target_before_rejection`.
V14 preserves that finding as terminal evidence; it does not reinterpret V13
as approved and does not spend the sole scientific N5 attempt.

All V13 and V12 bytes above remained unmodified during V14 construction.

## Sole V14 correction

V14 separates review preflight into the two satisfiable phases frozen by the
amendment.

### Phase A: review artifact and complete structural/core validation

Preflight first opens only the canonical V14 review artifact through its
caller-bound file SHA-256. `load_hashed_json` requires canonical JSON plus one
newline and validates the review content SHA-256. Before any successor target
open, preflight then requires:

- the exact outer review schema and every authority/core field;
- eligible author/reviewer separation, including V12 and V13 reviewers;
- exact frozen source/proof outer key sets;
- a plain `dict` for every nested binding;
- exactly `{"path", "file_sha256"}` before either nested value is consumed;
- exact plain-string values, fixed path/outer-key/role equality, and lowercase
  64-hex digests;
- no duplicate, noncanonical, escaped, aliased, or role-swapped path; and
- equality between the entire parsed review core and
  `expected_source_review_core` before candidate hashing.

Missing/extra fields, mapping subclasses, nonstrings, malformed digests,
noncanonical paths, duplicates, role swaps, ineligible reviewers, and changed
authority fields all reject with zero successor candidate reads. A
well-formed wrong digest whose enclosing content hash is left unchanged also
rejects while reading only the review artifact and opens zero successor
candidates.

### Phase B: fixed canonical candidate rehash

Only a canonical, structurally valid, fully self-consistent review reaches
candidate hashing. Preflight iterates the frozen tuples in this exact order:

1. all five `SUCCESSOR_SOURCE_PATHS` entries; then
2. all four `SUCCESSOR_PROOF_PATHS` entries.

Every open expression is `read_regular_bytes(ROOT / relative, ...)`, where
`relative` comes from those tuples. The parsed nested `path` value is never an
open path. The retained descriptor-safe reader enforces canonical-root,
no-follow, singly linked regular-file semantics. The first digest mismatch
raises immediately.

This is the only V14 behavior change. No wrong digest can pass, and no caller
can select an open path.

## Exact review-open-order proof

The lifecycle suite proves a valid minimal review still passes unchanged and
reproduces every V13 nested-binding schema attack with recomputed enclosing
hashes. Every Phase A structural/core attack records exactly zero successor
candidate reads.

For syntactically valid but stale digest mutations, the suite writes a
canonical review artifact with a current caller-bound file hash but the old
enclosing content hash. Both source and proof cases read only that review and
reject with zero candidate reads.

For fully self-consistent wrong digests, enclosing content and file hashes are
recomputed. The observed candidate prefixes are exactly:

| Wrong digest position | Source opens | Proof opens | Total candidate opens |
| --- | ---: | ---: | ---: |
| first source | 1 | 0 | 1 |
| middle source | 3 | 0 | 3 |
| final source | 5 | 0 | 5 |
| first proof | 5 | 1 | 6 |
| middle proof | 5 | 3 | 8 |
| final proof | 5 | 4 | 9 |

Each case rejects at the mismatching file. The read spy requires the exact
ordered `ROOT / relative` prefix and rejects any non-tuple path. Thus data,
RGB, checkpoint, GPU, output, `.generated`, and caller-selected opens remain
zero in every Phase A and Phase B rejection.

## Preserved scientific and lifecycle contract

V14 makes no scientific or runtime-lifecycle change. After mechanical
`V14`/`v14` to `V13`/`v13` normalization, the trainer, verifier, executor,
synthetic execution, and science-test modules have complete AST identity with
V13. Every normalized policy top-level function is identical to V13 except the
four predecessor-authority functions:

- `authority_bindings`;
- `preflight_static_authority`;
- `expected_source_review_core`; and
- `preflight_source_review`.

No policy function was added or removed. The existing strict nested-binding
validator is unchanged after normalization. Lifecycle behavior remains in the
normalized executor; the V14 lifecycle proof adds only terminal-V13 binding,
reviewer separation, and the clarified open-order cases.

V14 therefore retains fresh initialization; seed `20260710`; the exact N5
frames, targets, and mappings; four V11 loss terms plus exact additive
`0.25 * derived_raster_cell_nll`; gather, float32 epsilon clamp, log, all-cell
mean; 4,000 batch-five AdamW updates and 20,000 exposures; schedule SHA-256
`fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`;
final-update selection; matched and cyclic wrong-RGB controls; class/family
diagnostics; isolated verification; transaction/recovery/failure semantics;
and all 26 checks and thresholds.

## Proof results

Every test command hid all accelerator selectors, removed
`HSA_OVERRIDE_GFX_VERSION`, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to one, and disabled external
pytest plugin autoload. No command used more than three concurrent one-worker
pytest processes.

The final V14 command ran:

```text
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_lifecycle.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_ladder_gate.py
```

Result: **235 passed in 19.06 seconds**.

The count partitions as 23 V14 science tests, 192 V14 lifecycle tests, and 20
frozen ladder-gate tests.

The retained V13 command passed **226 tests in 23.74 seconds**. The retained
V12 command passed **202 tests in 23.19 seconds**. The retained V11 command
passed **190 tests in 22.22 seconds**.

All eight Python production/proof files compiled in memory without bytecode
creation. They have LF line endings, a final newline, and no trailing
whitespace. Source identity, normalized AST, reviewer separation, absence,
no-authority, and no-output-root checks passed.

## Real isolated verifier smoke

The production V14 executor ran its source-only smoke through:

```text
sys.executable -I -B scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py --cpu-verifier-contract-smoke
```

It reported:

```text
real_subprocess=true
isolated=true
no_bytecode=true
accelerators_hidden=true
success_validated=true
independent_v14_raster_nll_recomputed=true
native_class_family_diagnostics_recomputed=true
native_to_retained_compatibility_boundary_exercised=true
shared_production_gate_reconstruction=true
retained_gate_check_count=26
phase_failures_validated=11/11
process_cases_validated=timeout,signal,nonzero,malformed,oversized,stderr
exact_rejects_synthetic_schema=true
smoke_rejects_exact_schema=true
temporary_tree_removed=true
publication_performed=false
```

## Access and authority closure

During V14 implementation and proof:

- no canonical experiment data or RGB payload was opened;
- no `.generated` receipt or payload was opened or mutated;
- no V11, V12, or V13 checkpoint was opened, copied, hashed, loaded, compared,
  or used;
- no GPU or iGPU operation ran;
- all executable proofs used synthetic temporary roots only;
- the canonical V14 output root is absent;
- the canonical V14 independent-review file is absent;
- no exact training, exact verification, finalization, or publication ran; and
- no canonical review was written by the implementation author.

V14 currently grants no exact authority. Retry, second seed, N16, later-rung
execution, Shared-JEPA training, checkpoint use, held-out, G2, selection,
calibration, navigation, runtime, hardware, production, promotion, and
deployment remain unauthorized.

## Independent review

The next action is a different-agent source review only. The eligible reviewer
must:

1. rehash the V14 amendment, full terminal V13 closure/handoff/BLOCK, every V14
   source/proof, this handoff, and all retained V12/V11 bindings;
2. reproduce normalized V14/V13 AST identity and inspect only the terminal
   predecessor authority and clarified review/open-order delta;
3. rerun the V14 `235`, V13 `226`, V12 `202`, and V11 `190` hidden-CPU suites,
   compilation, whitespace, absence, and real-child proofs;
4. independently repeat every Phase A zero-candidate-open attack;
5. independently repeat first/middle/final source/proof self-consistent wrong
   digests and require exact fixed prefix reads with no non-tuple open; and
6. publish
   `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_independent_review_2026-07-14.json`
   as `PASS` or `BLOCK` last.

Only an eligible canonical `PASS` over these exact frozen bytes may authorize
one future fresh V14 N5 attempt. This handoff is not an execution receipt and
grants no data, RGB, checkpoint, GPU, retry, later-rung, training, held-out,
navigation, production, or promotion authority.
