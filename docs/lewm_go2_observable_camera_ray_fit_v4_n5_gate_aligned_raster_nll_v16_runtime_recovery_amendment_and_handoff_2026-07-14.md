# Camera N5 V16 lean runtime-recovery amendment and handoff

Date: 2026-07-14

Status: source-only successor proposed; exact execution remains blocked until a
fresh different-agent V16 review passes and is supplied by its canonical hash.

## Terminal V15 fact

V15 consumed its only attempt. It failed after durable reservation and before
RGB, model, optimizer, checkpoint, result, metric, or gate creation because the
live visibility check set PyTorch inter-op threads to one and the retained
determinism helper attempted the one-shot setter again. V15 retry remains
forbidden.

Frozen terminal evidence:

- reservation file/content SHA-256:
  `bae23223289aa07ae1951f6d7d1202780856aa555d51e385f1961a229c1ae706` /
  `ecccab261a3e9d5bcb2fb6b3f0fe52c864abd4fd6c5ed07d6dcce53347b17b29`;
- failure file/content SHA-256:
  `797280760654144a156d96148664d956d67b38e3d70cfc07afe9936ad6c3b2fe` /
  `73862b8c640bd4aacaf68917d263a0265a0359fbf26219421729fe04da4e31a0`;
- empty seed lock SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- seed-root inventory: `.n5.reservation-v15.lock`, `n5`;
- output-root inventory: `attempts`, empty `gates`, empty
  `metric_verifications`.

The wrapper revalidates that evidence before and after every V16 visibility or
exact operation. It never redirects, deletes, recovers, or reuses V15 output.

## Sole V16 delta

The only scientific-runtime change is idempotent thread configuration around
the retained `configure_determinism` call. Its original seed, NumPy, PyTorch,
CUDA, cuDNN, deterministic-algorithm calls and returned receipt are preserved.
Each intra/inter-op setter is passed through only when the observed count is
not already exactly integer one; setter failure or a non-one postcondition
fails closed. Booleans are rejected as thread counts or requests.

Everything else is frozen V15: seed `20260710`, N5 subset and data hashes,
model, fresh initialization, five-term loss, weights, AdamW settings, 4,000
updates, schedule hash, evaluation, verifier, thresholds, no-retry rule, GPU0
R9700 requirement, and GPU1/Raphael exclusion.

V16 uses these new identities:

- output root:
  `.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v16/`;
- attempt scope: `one_exclusive_fresh_gate_aligned_raster_nll_v16_attempt`;
- visibility receipt:
  `/tmp/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v16_gpu_visibility_preflight_2026-07-14.json`;
- source review:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v16_independent_review_2026-07-14.json`.

Retained V15 artifact schemas, transaction implementation, and internal
license vocabulary are compatibility machinery only. They do not authorize a
V15 retry. The V16 review binding, terminal-V15 binding, and this compatibility
declaration are injected into `authority_bindings`, which is durably included
in the reservation and every successful downstream result/gate.

## Review and one-shot boundary

The canonical V16 review must be by an agent other than the implementation
author. It must hash the wrapper and focused test as successor sources and this
record as a successor proof; bind all six frozen V15 policy/preflight/executor/
trainer/verifier/review files; bind the complete terminal evidence above; and
state all of:

- source closure approved;
- exactly one fresh V16 attempt authorized;
- no scientific change authorized;
- no V15 retry authorized;
- no later-rung, held-out, G2, navigation, hardware, production, promotion, or
  checkpoint-use authority granted.

The retained Git containment check requires every reviewed successor source
and proof to be present at the current Git `HEAD`. Thus review occurs only
after this three-file source closure is committed. The wrapper accepts the V16
review through the retained `--source-review-sha256` argument and uses it in
the visibility receipt, reservation, verifier request, result, and gate.

The V16 output root must be absent. Exactly one visibility preflight and one
exact invocation are permitted. Any post-reservation outcome is terminal and
must not be retried. A pre-reservation rejection creates no V16 attempt.

Reviewed visibility command shape:

```text
python -I -B scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v16.py --v16-gpu-visibility-preflight --source-review-sha256 <V16_REVIEW_FILE_SHA256>
```

Reviewed exact command shape:

```text
python -I -B scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v16.py --source-review-sha256 <V16_REVIEW_FILE_SHA256> --gpu-visibility-receipt-sha256 <V16_RECEIPT_FILE_SHA256> --gpu-visibility-receipt-content-sha256 <V16_RECEIPT_CONTENT_SHA256> --rgb-workers 5
```

No visibility diagnostic, GPU operation, data/RGB open, `.generated`
mutation, reservation, training, verification, or exact attempt was performed
while authoring this closure.
