# Camera-ray N5 gate-aligned raster NLL V13 implementation handoff

Date: 2026-07-14

Implementation author: `/root/camera_v12_gate_aligned_implementer`

Status: **source and synthetic CPU closure complete; independent review required; no exact authority**

## Frozen authority

The source-free governance amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_strict_review_binding_successor_amendment_2026-07-14.md`

File SHA-256:

`2eaaaa7b896dd42bed02d5a75072d1933b11ad4cce5e8d83f35f1d137ba89633`

The amendment author is `/root`. The fixed implementation author matches the
amendment. A canonical reviewer must start with `/root/` and differ from
`/root`, this implementation author, the terminal V12 reviewer
`/root/raw_v11_builder_auditor_diff`, and the future exact execution agent.
The implementation author did not write a canonical review.

## Frozen production closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| retained V12 model/loss | `lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py` | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |
| V13 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `e5c03f0ed4a9cb82daeb040c2fe8f87a68911500c47c85992b2780b06f53082f` |
| V13 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `92d6fef2a32498b4dc80566f73422b3735d2d9bbb39612b8a8946d7aa3a34d43` |
| V13 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `7fe1fa1f107478303c10cecd0b591388e1fdb042e14f0ad289f0b36ee399686b` |
| V13 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `77d7782078dc8b089f97144117d7dd0d8d0116dbfbe55a8b665335ee9de55a54` |

No V13 model/loss copy exists. V13 binds and imports the retained V12
model/loss at the exact hash above.

## Frozen proof closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| synthetic lifecycle and native V13 gate fixture | `lewm/tests/n5_gate_aligned_raster_nll_v13_synthetic_execution.py` | `19c6a1897b247760653c1329e46d389ab7a1b760074967f0e29ace9a19fd36b3` |
| V13 loss, parity, gradient, and diagnostic tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `2ebac0d62fa6c67e97ff174b301882cce73bda3b0f11bfa008ef23ff20745596` |
| V13 lifecycle, strict-binding, schema, gate, and subprocess tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_lifecycle.py` | `d204e5ca88960bc8dc57f3acc328bff2387ca58cc15689624a61c357bc49ea85` |

This handoff is the fourth proof file. The independent reviewer must hash its
final bytes and bind that hash in the canonical review.

## Retained terminal V12 closure

The V13 policy rehashes and binds the frozen V12 source-only closure and its
terminal review BLOCK. The author independently rehashed these files before
freezing V13:

| Role | Path | File SHA-256 |
| --- | --- | --- |
| V12 amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_successor_amendment_2026-07-14.md` | `77de8c69b1bef69ab3d1b976567eb20371f53d47d81af757ef8c7fdaade93c1b` |
| retained model/loss | `lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py` | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |
| V12 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `ad8a77c4f201f00891e7e6b45c395966eaa8f3723a3b2720d26eeb0b1ca23fc6` |
| V12 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `91018ecd28483fbbc3399eea70d720a9b327e7e03b4920dbe349ca9b81603d54` |
| V12 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `f8814836c1073f13c563ba11035f806a0faa70be9a0d44b7d3e900350b1a8baf` |
| V12 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `4e4c45c85827ad4db6e65a4f02557fd6c5b1e9d97ada4ac4577cb0b6b099b521` |
| V12 synthetic proof | `lewm/tests/n5_gate_aligned_raster_nll_v12_synthetic_execution.py` | `1cbcb80d3f6bec5b9ce536d6b4fa9bad645d170a4be6b4e8d1b261ad5f5dc453` |
| V12 science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `98a11ec91865ff106dd943a6b6468ca227018d92db4a346fcbbe9497a7d8d099` |
| V12 lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_lifecycle.py` | `77b5d05373613220a0de1d78236659f0c038e9f1a91f3a4efbf5cbcaa73936c1` |
| V12 handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_implementation_handoff_2026-07-14.md` | `21d4858035225e2454a3e7fec3e71fb8571e4d69e7a592c5822c4a435b17b0b9` |
| terminal V12 review BLOCK | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_independent_review_2026-07-14.json` | `076855183730bcff58b507d8fde6c613a023b633681c7516daaf0d80b5e27158` |

The terminal V12 review content SHA-256 remains
`4a56c46ede9482f72b5ae304734e12a706d8f7075873b4e5de135f9fa6cc289d`.
Its sole blocking finding is
`nested_source_and_proof_bindings_accept_extra_fields`. It authorizes no exact
attempt, retry, checkpoint use, or downstream work. The original V12 review
object and every V12 source/proof byte remain unmodified.

## Sole V13 correction

`preflight_source_review` now performs two phases before any governed source
or proof target is opened.

First, `_validate_exact_successor_review_bindings` validates every nested item
in both binding classes. It requires:

- `type(binding) is dict`;
- the exact key set `{"path", "file_sha256"}` before either value is read;
- exact plain-string values;
- a lowercase 64-character SHA-256 digest;
- no duplicate bound path across source and proof bindings; and
- exact equality between the bound path, its canonical outer key, and its
  source or proof role.

Only after all nested items pass does preflight reconstruct and compare the
entire canonical review core and content hash. Only after that succeeds does it
open each governed candidate source/proof and compare its actual file hash.
There is no nested `.get`, mapping-subclass acceptance, key filtering,
normalization, truthiness coercion, or compatibility fallback.

Preflight also rejects `/root`, the amendment author, the implementation
author, and the terminal V12 reviewer as ineligible V13 reviewers.

## Preserved scientific and lifecycle contract

V13 makes no scientific change. The complete trainer, verifier, and executor
ASTs equal V12 after mechanical `V13`/`v13` to `V12`/`v12` normalization. For
the policy, every top-level function is AST-identical after that normalization
except the four authority-path functions:

- `authority_bindings`;
- `preflight_static_authority`;
- `expected_source_review_core`; and
- `preflight_source_review`.

The only added policy function is the strict nested-binding validator above.
The normalized V13 science test and synthetic-execution ASTs also equal their
V12 predecessors. The lifecycle proof differs only for V13 namespace/authority
coverage and the new strict-binding adversarial tests.

Consequently V13 retains byte-for-value scientific behavior: fresh model
initialization; the five N5 frames; seed `20260710`; the exact four V11 loss
terms and additive `0.25 * derived_raster_cell_nll`; the gather, float32
epsilon clamp, log, and all-cell mean; 4,000 batch-five AdamW updates; 20,000
exposures; the frozen schedule hash
`fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`;
final-update-only selection; matched and cyclic wrong-RGB controls; class and
family diagnostics; isolated verification; transaction/failure cleanup; and
all 26 retained checks and thresholds.

## Strict-binding adversarial proof

The V13 lifecycle suite constructs self-consistent review objects and proves:

- a valid minimal review still passes unchanged;
- one extra field in the final nested source binding is rejected after all
  enclosing canonical hashes are recomputed, with zero governed-target opens;
- the equivalent final nested proof attack is rejected with zero
  governed-target opens;
- missing `path` or `file_sha256`, independently in both binding classes, is
  rejected;
- hostile dict subclasses are rejected before `.get`, `__getitem__`, or any
  value is consumed;
- non-string paths/digests, malformed digests, and noncanonical paths are
  rejected;
- duplicate source and duplicate proof paths are rejected;
- a source/proof binding-role swap is rejected; and
- a well-formed but changed source or proof digest reaches only the governed
  candidate rehash and then fails closed.

All invalid binding-schema cases spy the governed source/proof opener and
require exactly zero calls. Preflight contains no data, RGB, checkpoint, GPU,
or output opener.

## Proof results

Every test command hid all accelerator selectors, removed
`HSA_OVERRIDE_GFX_VERSION`, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to one, and disabled external
pytest plugin autoload. No command used more than one pytest worker.

The final V13 command ran:

```text
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_lifecycle.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_ladder_gate.py
```

Result: **226 passed in 19.08 seconds**.

The count partitions as 23 V13 science tests, 183 V13 lifecycle tests, and 20
frozen ladder-gate tests.

The retained V12 command ran its science, lifecycle, and frozen ladder-gate
files. Result: **202 passed in 21.24 seconds**.

The retained V11 command ran its science, lifecycle, and frozen ladder-gate
files. Result: **190 passed in 20.73 seconds**.

All eight Python production/proof files compiled in memory without bytecode
creation. They have LF line endings, a final newline, and no trailing
whitespace.

## Real isolated verifier smoke

The final source-only smoke used the actual production executor entrypoint:

```text
sys.executable -I -B scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py --cpu-verifier-contract-smoke
```

It reported:

```text
real_subprocess=true
isolated=true
no_bytecode=true
accelerators_hidden=true
success_validated=true
independent_v13_raster_nll_recomputed=true
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

During V13 implementation and proof:

- no canonical experiment data or RGB payload was opened;
- no `.generated` receipt or payload was opened or mutated;
- no V11 or V12 checkpoint was opened, copied, hashed, loaded, compared, or
  used;
- no GPU or iGPU operation ran;
- all executable proofs used synthetic temporary roots only;
- the canonical V13 output root is absent;
- the canonical V13 independent-review file is absent;
- no exact training, exact verification, finalization, or publication ran; and
- no canonical review was written by the implementation author.

V13 currently grants no exact authority. A future exact attempt remains
conditional on a canonical, different-agent `PASS` binding every exact frozen
source and proof byte. Retry, second seed, N16, later-rung execution, Shared
JEPA training, checkpoint use, held-out, G2, selection, calibration,
navigation, runtime, hardware, production, and promotion remain unauthorized.

## Independent review

The next action is a different-agent source review only. The eligible reviewer
must:

1. rehash the V13 amendment, full V12 closure/handoff, terminal V12 BLOCK, all
   V13 sources/proofs, this handoff, and the retained V11/gate bindings without
   opening governed numeric payloads or checkpoints;
2. mechanically reproduce the V13/V12 AST comparison and inspect the sole
   strict-binding and authority-path delta;
3. rerun the V13 `226`, retained V12 `202`, and retained V11 `190` hidden-CPU
   suites, compilation, whitespace, absence, no-open, and real-child proofs;
4. independently repeat every nested source/proof adversarial mutation and
   verify invalid schemas open no governed target; and
5. publish
   `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_independent_review_2026-07-14.json`
   as `PASS` or `BLOCK` last.

Only an eligible canonical `PASS` over these exact frozen bytes may authorize
one future fresh V13 N5 attempt. This handoff is not an execution receipt and
grants no data, RGB, checkpoint, GPU, retry, later-rung, training, held-out,
navigation, production, or promotion authority.
