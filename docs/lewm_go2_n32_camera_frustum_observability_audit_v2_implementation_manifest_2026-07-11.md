# Go2 N32 camera-frustum audit v2 implementation manifest

Date: 2026-07-11

Status: reviewed pre-run v2 implementation manifest. This report authorizes
only a fresh metadata-only v2 preparation and, after its machine companion is
independently validated and hashed, the fit-only v2 audit. It does not
authorize a camera-frustum representation, training, G2, holdout, sealed,
seed-20260711, runtime, or promotion access.

## Binding and predecessor

- execution binding:
  `docs/lewm_go2_n32_camera_frustum_observability_audit_binding_2026-07-11.md`
- binding SHA-256:
  `c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9`
- controlling combined pre-authoritative incident:
  `docs/lewm_go2_n32_camera_frustum_manifest_preparation_failure_2026-07-11.md`
- incident SHA-256:
  `5c3fad3b8e296aed239c3573e263af766b52e391fb9fe86e0e31d26c94845db3`
- required incident status: `acknowledged_pre_authoritative_run`
- immutable v1 result/finalizer report:
  `docs/lewm_go2_n32_camera_frustum_observability_audit_v1_result_2026-07-11.md`
- v1 report SHA-256:
  `9882465826a848be303694efbac1c76468026a8c05ce1152e0089d9b6849a365`

The v1 result and its v1 human/machine manifests remain immutable and are not
v2 inputs. The v2 runner and finalizer must not open, hash, parse, or otherwise
inspect the generated v1 result. The binding commits its dated Markdown report
as historical provenance.

V1 completed the runner with a false representation decision because the
camera support omitted 1 FREE and 372 OCCUPIED targets. The independent
finalizer then rejected one camera evidence record because Python 3.12
`sum()` changed the quaternion norm by one ULP relative to the runner's
explicit four-term expression. V2 changes only that finalizer arithmetic and
the versioned manifest/result paths. It does not change any data, camera
geometry, label, mapping, tolerance, coverage gate, or authorization rule.

## Closed source graph

The authoritative v2 source graph contains exactly eleven roles:

| Role | Repository path | SHA-256 |
| --- | --- | --- |
| `binding` | `docs/lewm_go2_n32_camera_frustum_observability_audit_binding_2026-07-11.md` | `c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9` |
| `audit_core` | `lewm/benchmarks/go2_n32_camera_frustum_observability.py` | `ab97c34a8a07a93d6b49b5adb0b1a82bc66d38be206baab362b7b1f1b59f3cc3` |
| `audit_core_test` | `lewm/tests/test_go2_n32_camera_frustum_observability.py` | `a04a139a3685d7b14656eab6a111d2a476acc42a5b5726712d1f2abf9da4a45d` |
| `audit_runner` | `scripts/audit_go2_n32_camera_frustum_observability.py` | `f7e3a3e60937caabbe003ff41af6aec44248df137b0a53c383364272152f3079` |
| `audit_runner_test` | `lewm/tests/test_audit_go2_n32_camera_frustum_observability.py` | `91ea76189ca0b90dff5b1f30bf2001f8693ff26a0d5e97104b07322441836c2c` |
| `audit_finalizer` | `scripts/finalize_go2_n32_camera_frustum_observability.py` | `8ef40a4bc3f416728dd176cbe9989736e429b4645470452e46e3f15bce4794c2` |
| `audit_finalizer_test` | `lewm/tests/test_finalize_go2_n32_camera_frustum_observability.py` | `9a8c50a292b46bbfb9fbe113c1320f56fc55df560cbf2d2917cc73dbaecb70ed` |
| `label_semantics` | `lewm/datasets/go2_paired_navigation.py` | `14df0cf59ab7554431b1be2ef91e3ab7229200be94bb9afa88127e3ea53c2c08` |
| `geometry_contract_semantics` | `lewm/planning/geometry_contract.py` | `6873a9550399a5decc90e4a31b2945e54074bdb56855a035924f49b4511c813b` |
| `scene_manifest_semantics` | `lewm_worlds/lewm_worlds/manifest.py` | `5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888` |
| `planning_grid_semantics` | `lewm_worlds/lewm_worlds/planning_grid.py` | `e6f7e26d584dfd7923493803fc95a75135122b37a1f95cb51f9267b284649510` |

The independent stdlib evidence helper remains test-only:

- `lewm/benchmarks/go2_n32_camera_frustum_fit_evidence_stdlib.py`:
  `abc687e10e2ea1e54242edce1d90fad3024f48c73e29fcc1fb9098be5c1a53c9`
- `lewm/tests/test_go2_n32_camera_frustum_fit_evidence_stdlib.py`:
  `ceae04215acc3b173846a30519eee1f47b9a10357d94d137a49dbebf1a817aba`

## Versioned outputs

- human manifest: this file
- machine manifest:
  `docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.json`
- exclusive v2 result:
  `.generated/go2_n32_camera_frustum_observability_audit/v2/result.json`

The v1 result and manifest paths are not aliases for these v2 paths.

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

The prior adversarial reviews of source provenance, camera composition,
numeric types, modality and role denial, archive handling, reconstruction,
mapping, and full-selection-versus-fit-subset behavior remain represented in
the synthetic suite. The v2-specific review reproduced the v1 frame-13
quaternion and verified:

- the runner and finalizer now compute the exact same left-associative
  four-term quaternion norm;
- the complete camera-evidence dictionaries and canonical hashes are equal;
- rotation, camera geometry, tolerance, pass/fail logic, and every other
  evidence field are unchanged;
- all 145 frozen tests pass.

The final v2 reviewer identity is
`independent_adversarial_review_2026-07-11_v9_ulp_corrected`; status is
`reviewed_and_authorized`.

## Authorization boundary

This report authorizes a zero-based metadata-only v2 preparation. The machine
companion must bind this report's exact hash, the v2 source graph, the exact
verification evidence, a zero-boundary preparation ledger, and absence of the
exclusive v2 result. Both independent machine validators must pass before the
v2 runner may reopen the exact fit-only label shards. A v2 runner result
authorizes nothing until the standard-library finalizer completes.
