# V4 N5 full-panel V2 independent review

Date: 2026-07-13

Verdict: **BLOCK**

Reviewer: `/root/v4_full_panel_v2_block_record`

This is a source and CPU-test review of the frozen V2 successor. The retained
scientific and recovery tests pass, but the execution-authority lifecycle is
not isolated from ordinary imported Python code. Public function closure cells
expose the mutable production and test record mappings. Those mappings can be
edited to reactivate a consumed authority or register a reconstructed object.

The canonical PASS review JSON
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_independent_review_2026-07-13.json`
was deliberately not created. The exact attempt remains unauthorized.

The separate BLOCK JSON has canonical content SHA-256
`c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a`
and file SHA-256
`ddca89e467e4cc30e52bacf57b28c040465e712843fde465f472f3cc8b38fc73`.

## Frozen artifacts

The implementation handoff hash matched the requested
`3056b00f7b5f224c0507f07505c005f4f5ea2171fb97e6f78585cf7f0460bb61`.

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py` | `096b597b0e84a6822fd8fcdd8221da27e95757aaa2c05ca148afad6e23ad60d2` |
| `scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py` | `03311bb48da80b912c2576844adf5cd488c1b9a0818268d2252902d860436591` |
| `scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py` | `357369b652c489ab99937c06afaed0ec4cf66aa1f46017f74f5dac46da93d3aa` |
| `scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py` | `cab757839c3d784cb5760f30c2bde6163311bfbf87df1620c9c0f77ff69b624b` |
| `scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py` | `a5dc625b8b270913df56d8b5044c263ba3fdbd1ef6cb3e6f62e084a5335ee323` |
| `lewm/tests/n5_full_panel_v2_test_support.py` | `95892d289798580e0911eab1be43e8e899125ee8484eb2fa4e3afd5af2ed0557` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py` | `e249dce8af66d8e6709f1823f433ba76a56be8a54129f0620e20efa61d9ed8dd` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_independent_review.py` | `a53c5e5d351784ff2a4824231998194e15040597897411c91e7727ec73a95e69` |

The retained V1 policy, launcher, trainer, verifier, and finalizer hashes were
respectively `875edc86efbe25d246b24c2ef2467cc7956b1b3bb90e6d8d1e03e4a9c5b11d88`,
`3cb9ff782a15bc97dd3cca2cc25705e006d6af19a7dbef6d27dee893d9b570c8`,
`48ac856c080906a8d73d5a9b97d1dcf7fe21f5bc99217cce669c43b9c091acca`,
`00c62cec39e1eb05bf23a96a9153aa8ff350235c2e5c6662f6148934ab9d85b0`,
and `1d4471381a6c3b29f0b077e44e3126f956281ff105d4e38aa8e0f6ba18675b8b`.
The V1 review, BLOCK JSON, exploit tests, and handoff also retained their bound
file hashes: `11479b03ff9eac24dd5541d38faeda480739c8d17de7b2b658759e306ace2d5e`,
`ccd8d97988d2ce165722703fbfcf813758ee42a5408e02d26bf7db38d8ea506e`,
`387147a8dd6fe1a20184284a05c18df73419ca91c21054eb378e79a8194d5b3b`,
and `8f4735a3ecd20a8c19bd729fdaf71ceb60a3a884de717423e8f84ef6ef2745f7`.
The V1 BLOCK canonical content hash remained
`99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7`.

No author source or author test source was edited by this review.

## Blocking finding

`_build_authority_api()` creates the mutable `production_records` and
`test_scopes` dictionaries at policy lines 491-499. Nested functions read and
write those containers at lines 582-740, are returned at lines 742-750, and are
published as module functions at lines 754-763. Deleting the builder name does
not hide the cells: Python exposes them through each function's `__closure__`
and `__code__.co_freevars`.

The direct public closures expose `production_records` and `test_scopes`. The
test record mapping is also reachable by following the returned
`test_transition` function's `test_scope` closure into `test_scopes`. Although
each `_AuthorityRecord` value is frozen, its containing dictionary is not.

The independent reproducer demonstrated both consequences without exact-mode
or protected-role access:

1. After a test authority transitioned from `active` to `terminal`, replacing
   its dictionary entry with an otherwise identical record whose state was
   `active` allowed the same authority to transition again.
2. A slot-for-slot reconstructed `VerifiedAuthorityV2` object was inserted with
   its computed issuance digest and state `active`; validation accepted it.

The validator's exact-identity, issuance-digest, and lifecycle checks are all
derived from the mutable record selected from that mapping. They therefore do
not detect caller replacement or insertion of the record itself. The same
storage exposure exists for production lifecycle records even though this
review did not create the canonical PASS record needed to issue a production
authority.

## Verification

All three pytest commands used `/usr/bin/python3`, disabled external plugins,
set OMP, MKL, OpenBLAS, and NumExpr native threads to one, removed
`HSA_OVERRIDE_GFX_VERSION`, and set HIP, CUDA, and ROCr visibility to empty.

```text
independent closure reproducer: 1 passed, 3 failed in 0.05s
V2 adversarial/recovery suite:  20 passed in 1.11s
retained 48-test closure:       48 passed in 1.80s
aggregate:                      69 passed, 3 failed, 72 collected
```

The failing tests were:

- `test_importable_authority_api_exposes_no_mutable_lifecycle_registry`
- `test_consumed_authority_cannot_be_reset_through_importable_closure`
- `test_reconstructed_authority_cannot_be_registered_through_importable_closure`

The source-binding test passed. The author suite and retained closure passing
do not negate the independently reproduced lifecycle failure; they establish
that the frozen scientific/recovery regression surface remained unchanged.

After testing, both the canonical PASS JSON and the entire canonical output
root `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1` remained
absent. Exact mode was not run. No protected role data, model output, G2,
held-out, runtime, hardware, production, navigation result, or GPU was opened.

## Required successor condition

The exact attempt remains blocked until an additive successor prevents ordinary
imported object graphs from yielding mutable authority lifecycle state, passes
the three independent rejection probes plus the 20 V2 and 48 retained tests,
and receives a new different-agent review. This BLOCK grants no retry, later
rung, second seed, V5, G2, held-out, runtime, hardware, navigation, production,
or promotion authority.
