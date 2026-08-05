# Shared JEPA V5 raw-supervision Auditor V12 author handoff

Date: 2026-07-14

Implementation author: `/root/raw_v11_builder_auditor_diff`

Status: **source-only candidate complete; different-agent review required; no exact authority**

## Frozen governing amendment

This candidate implements only
`docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_hsa_worker_isolation_successor_amendment_2026-07-14.md`,
file SHA-256
`f4892405cf0fd97f9096f99d840b5590810fd8640822ed2e8c4c254c0c3e6adf`.

The implementation did not open the canonical dataset, any `.generated`
payload, RGB, checkpoint, G2, held-out, runtime, navigation, hardware, or
production artifact. It did not run an exact audit, rebuild, training job,
navigation job, or GPU job. All executable proof used CPU-only temporary
synthetic roots with native math threads fixed to one and accelerators hidden.

## Candidate artifacts

| Role | Path | File SHA-256 |
| --- | --- | --- |
| Auditor source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v12.py` | `f435406c7ff8d42a549cd678a65584bc88ac49f96b590247b811c6bb4b934943` |
| Auditor CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v12.py` | `45f93534b02afe99722144509fc9b7dde72e735daa8bed1dc58951d3c0bb8471` |
| Auditor test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v12.py` | `dbefb4dc455b45873e14256d5fa647e22fcf1eff1a43ba249e7b9fe7f5ed5dd7` |

The handoff intentionally does not self-hash. The independent reviewer must
compute its file hash after these bytes are frozen.

## Implemented correction

V12 preserves the V11 science, Builder V9 parity, eight-array replay,
comparison, report arithmetic, and atomic publication behavior. Its additive
authority map has exactly 25 ordered roles and deeply binds the complete V11
candidate plus the terminal V11 BLOCK review, in addition to the frozen V9 and
V10 authorization and terminal chains. V9, V10, and V11 success leaves must
all remain absent before authority acceptance, during success publication,
and during terminal-failure publication.

The closed accelerator selector tuple is exactly:

```text
CUDA_VISIBLE_DEVICES
HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES
GPU_DEVICE_ORDINAL
HSA_VISIBLE_DEVICES
```

Every real spawn initializer and each real production task clears all five
selectors and sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `1` before authorization or an
opener. The CLI performs the same sanitation before importing the auditor or
NumPy and separately unsets `HSA_OVERRIDE_GFX_VERSION`.

## Author proof

The focused V12 suite passes `62 passed`. It includes:

- hostile nonempty inheritance for all five selectors through actual spawned
  workers;
- dynamic proof that the real initializer clears selectors before accepting
  authority;
- independent re-poisoning and setter-before-authorization-before-opener proof
  for `_validate_one_shard_task`, `_hash_source_file`, and
  `_recompute_exact_sample_task`;
- AST enumeration of exactly those three production pool targets and the real
  initializer on every pool;
- real one-worker and six-worker synthetic replay with identical canonical
  science bytes and identical bytes for all eight arrays;
- hostile subprocess import proof for CLI pre-import sanitation and override
  removal;
- exact 25-role phase-one validation with zero target opens and rejection of
  incomplete, repeated, reordered, noncanonical, changed, symlinked, and
  hard-linked authority;
- deep exact validation of the V11 BLOCK, candidate, blocker, findings,
  verification, and all-false authority;
- closed V11-to-V12 AST delta and exact HSA-only selector delta; and
- real success/failure publication preserving the dataset, V9 failure, V10
  failure, V11 BLOCK, and absence of all predecessor success leaves.

The unchanged V11 and retained V10/V9 Builder/Auditor suites pass `227 passed`.
Source, CLI, and test pass `py_compile`. Whitespace checking found no trailing
whitespace. The proof environment explicitly hid all five selectors, unset
`HSA_OVERRIDE_GFX_VERSION`, fixed all four native thread variables to one, and
disabled user-site and automatic pytest plugin loading.

## Reviewer boundary

The next action is an independent source review by an eligible different
agent. The reviewer must independently hash all four candidate artifacts,
rehash the frozen V11 candidate and BLOCK, reproduce the temporary-root
CPU-only proof, and publish one canonical `PASS` or `BLOCK` review at the
amendment's fixed path.

This handoff grants no exact audit attempt, retry, rebuild, RGB decode, dataset
use, training, selection, calibration, G2, held-out, runtime, navigation,
hardware, production, promotion, or deployment authority.
