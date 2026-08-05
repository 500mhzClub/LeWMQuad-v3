# Go2 N32 camera-frustum audit implementation manifest

Date: 2026-07-11

Status: reviewed post-failure pre-run implementation manifest. This report
supersedes the stale manifests with SHA-256
`ef8d1a8a768c430caad82505634ec7e25e703c50c4b4a8d098b7a41267b113e6`
and `67b594ccd200af1b8e970dd65b78fd569b1a0f22a0166e419c278eb423b1c21a`.
It authorizes only a fresh metadata-only manifest-preparation pass and, after
its machine companion is independently reviewed and hashed, the fit-only
camera-frustum audit. It does not authorize training, G2, holdout, sealed,
seed-20260711, runtime, or promotion access.

## Binding and incident

- execution binding:
  `docs/lewm_go2_n32_camera_frustum_observability_audit_binding_2026-07-11.md`
- binding SHA-256:
  `96eb4b9eb11b0924056ffb89590ecf13bb20ffcc72c6aca5e6cb51e92bb8132e`
- controlling combined pre-authoritative incident:
  `docs/lewm_go2_n32_camera_frustum_manifest_preparation_failure_2026-07-11.md`
- incident SHA-256:
  `5c3fad3b8e296aed239c3573e263af766b52e391fb9fe86e0e31d26c94845db3`
- required incident status: `acknowledged_pre_authoritative_run`

The combined incident incorporates the earlier out-of-ledger search incident
and the first failed metadata-only preparation. That attempt parsed the fit
panel and an initial prefix of committed source metadata, then failed because
the implementation incorrectly required the complete rendered set to equal
the 320-frame fit subset. It emitted no inventory or phase ledger and opened
no label-shard, image, model, G2, held-out, runtime, or sealed payload. The
incident is not evidence and does not contribute to the fresh preparation
ledger.

The amended implementation now enforces the producer's actual structure:
frame-selection keys equal render-summary keys over the complete per-scene
render set; the 320 unique fit endpoints are an exact subset; every selected
render key occurs exactly once in the source JSONL with the committed
timestamp; and source rows outside the selected render set are strict-parsed
and ledgered but never used as fit evidence. Since the failed attempt, only
synthetic fixtures and repository source/documents have been opened during
implementation, testing, and review.

The first preparation under the subset-corrected implementation completed
with a zero-boundary ledger, but its candidate machine manifest, SHA-256
`42b72e7d78b034d85134c05539d1912bbeebe6544307d695fb333436174a5dce`,
was rejected by the independent finalizer before fit-panel or label-shard
access. The finalizer incorrectly treated the `test` token in the registered
implementation role `audit_core_test` as a physical non-training dataset
role. That candidate inventory and machine manifest are stale and provide no
authorization. The correction exempts only the exact registered
`audit_core_test`, `audit_runner_test`, and `audit_finalizer_test` source roles;
all dataset `test` roles and all higher-priority path, modality, alias, and
allowlist denials remain unchanged.

## Closed source graph

The authoritative source graph contains exactly eleven roles:

| Role | Repository path | SHA-256 |
| --- | --- | --- |
| `binding` | `docs/lewm_go2_n32_camera_frustum_observability_audit_binding_2026-07-11.md` | `96eb4b9eb11b0924056ffb89590ecf13bb20ffcc72c6aca5e6cb51e92bb8132e` |
| `audit_core` | `lewm/benchmarks/go2_n32_camera_frustum_observability.py` | `c243760a6984181274b4733127c5c39d3b31c1f3cf9c83a1fae6601014325820` |
| `audit_core_test` | `lewm/tests/test_go2_n32_camera_frustum_observability.py` | `cd199b8b22294fa0fdedea7592d1ec2577193dcd0978de47d62a9871cbf075df` |
| `audit_runner` | `scripts/audit_go2_n32_camera_frustum_observability.py` | `42221662449eca7931f8156bb4040779f7d89d5800ef005627fae0a650f73853` |
| `audit_runner_test` | `lewm/tests/test_audit_go2_n32_camera_frustum_observability.py` | `91ea76189ca0b90dff5b1f30bf2001f8693ff26a0d5e97104b07322441836c2c` |
| `audit_finalizer` | `scripts/finalize_go2_n32_camera_frustum_observability.py` | `5308d60ea44e5f4bebda879810c671c4467a31848b2b8b9b23c25cf26c9bb8c5` |
| `audit_finalizer_test` | `lewm/tests/test_finalize_go2_n32_camera_frustum_observability.py` | `956f6115ccf247d5858a37bde1b7b9aa4e44babd20d40ea51e943e5246cf0c66` |
| `label_semantics` | `lewm/datasets/go2_paired_navigation.py` | `14df0cf59ab7554431b1be2ef91e3ab7229200be94bb9afa88127e3ea53c2c08` |
| `geometry_contract_semantics` | `lewm/planning/geometry_contract.py` | `6873a9550399a5decc90e4a31b2945e54074bdb56855a035924f49b4511c813b` |
| `scene_manifest_semantics` | `lewm_worlds/lewm_worlds/manifest.py` | `5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888` |
| `planning_grid_semantics` | `lewm_worlds/lewm_worlds/planning_grid.py` | `e6f7e26d584dfd7923493803fc95a75135122b37a1f95cb51f9267b284649510` |

The independent stdlib evidence helper is test-only and is not imported by the
authoritative finalizer:

- `lewm/benchmarks/go2_n32_camera_frustum_fit_evidence_stdlib.py`:
  `abc687e10e2ea1e54242edce1d90fad3024f48c73e29fcc1fb9098be5c1a53c9`
- `lewm/tests/test_go2_n32_camera_frustum_fit_evidence_stdlib.py`:
  `ceae04215acc3b173846a30519eee1f47b9a10357d94d137a49dbebf1a817aba`

## Runtime

- Python implementation: `cpython`
- implementation version: `[3,12,3,"final",0]`
- Python version:
  `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- NumPy version: `1.26.4`

## Verification evidence

All commands ran from the repository root and exited zero.

1. Category `pytest`, deterministic result `passed_test_count=145`:

   ```text
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3:/home/andrewknowles/Workspace/LeWMQuad-v3/lewm_worlds /usr/bin/python3 -m pytest -q lewm/tests/test_go2_n32_camera_frustum_observability.py lewm/tests/test_go2_n32_camera_frustum_fit_evidence_stdlib.py lewm/tests/test_audit_go2_n32_camera_frustum_observability.py lewm/tests/test_finalize_go2_n32_camera_frustum_observability.py
   ```

   Captured-output SHA-256:
   `140093549e158044c6030b7b944108dfd8e7e626c0f680dff13ed208e42f9ac1`.

2. Category `py_compile`, deterministic result `compiled_file_count=4`:

   ```text
   /home/andrewknowles/TinyQuadJEPA/bin/python -m py_compile lewm/benchmarks/go2_n32_camera_frustum_observability.py lewm/benchmarks/go2_n32_camera_frustum_fit_evidence_stdlib.py scripts/audit_go2_n32_camera_frustum_observability.py scripts/finalize_go2_n32_camera_frustum_observability.py
   ```

   Captured-output SHA-256:
   `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

3. Category `import_isolation`, deterministic result
   `forbidden_import_count=2`:

   ```text
   PYTHONNOUSERSITE=1 /usr/bin/python3 -c "import sys; from scripts import finalize_go2_n32_camera_frustum_observability; assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules"
   ```

   Captured-output SHA-256:
   `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

4. Category `diff_check`, deterministic result `whitespace_error_count=0`:

   ```text
   git diff --check
   ```

   Captured-output SHA-256:
   `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Review record

Successive adversarial reviews closed failures in manifest completeness,
archive handling, path and role denial, canonical ordering, selected shard
rows, camera extrinsics, numeric types and tolerances, box parity, source JSONL
policy, verification commands, denial precedence, and independent finalizer
reconstruction.

The final post-failure review independently checked the amended full-render
and fit-subset relation in the runner and finalizer. It required and verified
positive fixtures containing fit-selected, non-fit-selected, and outside-
selection source rows; strict malformed-outside-row rejection; missing,
duplicate, and retimestamped non-fit selected-key rejection; and isolated
selection/render mismatch mutations that reach the intended equality checks.
Its final verdict was `CLEAN` on the exact runner, finalizer, and test hashes
listed above. The frozen suite passed 145/145 tests.

A final read-only review then adversarially checked the implementation-test
role carve-out. Eighteen role/path/modality combinations verified that only
the three exact registered source roles pass, while lookalikes, dataset test
roles, nontrain paths, wrong modalities, aliases, and hash/path substitutions
remain denied. Its verdict was `CLEAN` on the exact finalizer and finalizer-
test hashes listed above.

Reviewer identity for the machine companion is
`independent_adversarial_review_2026-07-11_v8_source_role_corrected`; review status
is `reviewed_and_authorized`.

## Authorization boundary

This human report authorizes a fresh metadata-only preparation ledger and
inventory under the amended binding. The retry must start from a zero ledger;
the earlier failed attempt remains only in the controlling incident record,
and the rejected candidate inventory and machine manifest remain stale.
The machine companion must bind this report's exact file hash, the fresh
preparation output, the source graph and command evidence above, prove the
exclusive result path absent, and be independently validated before the
authoritative runner opens a label shard or writes a result.
