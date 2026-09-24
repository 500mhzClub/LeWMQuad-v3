# Camera-ray N5 hierarchical-first-hit V10 implementation handoff

Date: 2026-07-14

Implementation author: `/root/coordinator_v2_qa`

Status: **source/proof candidate frozen; different-agent review required; no exact authority**

## Scope

This is the additive lifecycle-only implementation authorized by
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_durable_verifier_lifecycle_amendment_2026-07-14.md`,
SHA-256
`1d4e4e315c880ef8b1362093f41b1d1cb5cabf6052c13886fac0a9fe2573501f`.

The frozen V9 loss remains the sole loss implementation. V10 changes no model
capacity, input panel, target construction, optimizer, schedule, loss weight,
diagnostic cadence, checkpoint selection, evaluation, threshold, or gate
arithmetic. It adds only a fresh V10 authority namespace and the durable
verifier lifecycle required by the amendment.

No canonical data, RGB, V9 checkpoint/result, V10 output, `.generated`
artifact, GPU, exact attempt, G2, held-out, runtime, navigation, hardware, or
production path was opened or executed during construction or author tests.

## Implemented lifecycle

- The verifier advances through the exact closed eleven-phase vocabulary via
  a forward-only state object that retains only the current phase.
- Every caught child exception emits one canonical, self-hashed failure
  envelope on stdout. Messages, traceback frames, and total bytes are bounded;
  paths are repository-relative; numerical message content is redacted.
- The parent uses a real `Popen` boundary and retains integer return code,
  derived signal, timeout state, full stream byte counts and SHA-256 values,
  64 KiB overflow flags, 2,048-character sanitized excerpts, and either a
  validated child envelope or one closed parse reason.
- The request and diagnostic bind source-review content plus artifact file,
  content, and byte-count identities.
- On verifier failure, the parent publishes and fsyncs attempt-local
  `verification_failure.json` before cleanup. The terminal `failed.json` binds
  its path, file SHA-256, content SHA-256, byte count, and every cleanup result.
- If diagnostic publication or its directory fsync fails, checkpoint, result,
  and completion are preserved and the attempt uses status
  `diagnostic_publication_failed_preserved_owned_artifacts`.
- Every diagnostic and terminal license is false. There is no retry, repair,
  in-process verifier, or fallback path.

## Real subprocess proof

The production executor exposes only the no-argument
`--cpu-verifier-contract-smoke` review mode. It launches the actual V10 script
as `sys.executable -I -B ... --verification-child`, hides every accelerator,
caps native threads, uses a private temporary production-ineligible checkpoint
and five fixed synthetic frames, loads fresh CPU model state, computes matched
and wrong-input results, validates the response in the parent, and removes the
temporary tree.

The same real boundary proves success, injected failure at all eleven phases,
timeout, signal, nonzero, malformed, oversized, and stderr cases. Exact-child
code rejects the synthetic schema and smoke-child code rejects the exact
schema. The smoke has no caller-controlled review, path, seed, checkpoint,
data, output, backend, or authority argument and cannot reach reservation,
training, publication, or finalization.

## Frozen candidate hashes

| Role | Path | SHA-256 |
| --- | --- | --- |
| Retained loss | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| V10 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `9ff40daadcda1962de2d9d54def09b7ec5a128c0f7f3f14ee2449367f15481d5` |
| V10 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `ec22c49855fe310f43bc72132a53e867604126db096e1064451e56f080259b1a` |
| V10 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `4ea17008e7805aba63a50415e8e9aefed31ebf70f1ccf803ec7e64e29a72cdbc` |
| V10 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `5387fbce1eb4c7c8cd1628fcf97c33a6bc7d15f8afd748a65760124d8f7002b4` |
| Synthetic lifecycle | `lewm/tests/n5_hierarchical_first_hit_v10_synthetic_execution.py` | `843a75dc295451190af43c255475cbe6541d6d305b448f5dde9bc173fcbb76d5` |
| Science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `d27e6a6e98d5fdec9d70b446d4f6f760b87cf0057ea0299db2131f252561f1a5` |
| Lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_lifecycle.py` | `59f5a1a784586dba97170890a356e73b8b4005fb14b65f640437465289ba60a6` |

The handoff file is the fourth proof artifact and is intentionally hashed by
the different-agent reviewer after this final write.

## Author verification

All commands hid `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`, and `HSA_VISIBLE_DEVICES`, set
all native math thread variables to one, disabled bytecode, and used only
source files or private `/tmp` trees.

- Actual V10 script smoke: exit 0; success, 11/11 phases, and 6/6 process cases
  validated in 11.1 seconds.
- V10 science and lifecycle closure: `150 passed in 18.70s`.
- Retained V9 science and lifecycle closure: `146 passed in 5.97s`.
- Dedicated durable lifecycle selection: `4 passed, 135 deselected in 11.59s`.
- Final ASCII scan: no findings.
- Final `git diff --check`: no findings.
- New source compilation under `python -I -B`: all seven Python files passed.

The lifecycle tests include actual-script subprocess execution without a
subprocess mock, diagnostic fsync-before-cleanup ordering, diagnostic survival,
terminal binding, one-shot diagnostic fsync failure with scientific artifact
preservation, stream bounds/hashes/truncation, all false licenses, absence of
numerical payload, normalized V9/V10 science AST equivalence, and fresh-state
trainer/verifier evaluation equality.

## Review and authority

The required reviewer must start with `/root/` and differ from both `/root`
and `/root/coordinator_v2_qa`. The reviewer must rehash every source and proof,
rerun the real subprocess and retained CPU closures, and publish the canonical
review JSON last at
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_independent_review_2026-07-14.json`.

Until that record is a canonical `PASS`, this candidate grants no exact
attempt. This handoff itself never grants data, checkpoint, GPU, exact, retry,
later-rung, shared-JEPA, G2, held-out, runtime, navigation, hardware,
production, promotion, or deployment authority.
