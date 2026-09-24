# Camera-ray N5 hierarchical-first-hit V10 durable-verifier lifecycle amendment

Date: 2026-07-14

Amendment author: `/root`

Status: **source construction and different-agent review only; no exact authority**

## Trigger and boundary

The sole reviewed Camera V9 attempt completed fresh training and then
terminalized when its isolated verifier child returned nonzero. V9 discarded
the child return details and captured streams, removed the unverified
checkpoint/result/completion, and retained only a generic failure receipt.
Its numerical outcome is therefore unobserved and unrecoverable.

The frozen diagnosis is
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_terminal_verifier_failure_diagnosis_2026-07-14.md`,
SHA-256
`59e9036a6c052d6ab8aafca23b54a9ef9d3be56e8f9d4c364bb683aa6f65ec69`.

This amendment creates an additive V10 **lifecycle** successor. V9 remains
terminal and may not be retried, repaired, reinterpreted, or used for numeric
evidence. V10 must train from a fresh initialization in a new one-attempt
namespace. No V10 source, proof, review, output, checkpoint, metric, or gate
artifact existed when this amendment was frozen.

This amendment grants no exact execution, data/RGB opening, checkpoint use,
GPU use, later fit rung, shared-JEPA training, selection, calibration, G2,
held-out, runtime, navigation, hardware, production, promotion, deployment, or
retry authority.

## Frozen V9 evidence

| Role | Path | SHA-256 |
| --- | --- | --- |
| V9 scientific amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_preimplementation_amendment_2026-07-13.md` | `ccc8097b4d3bd70aabf3c701226928e360fafb04a12a452c4fd406e9bba3db0a` |
| V9 loss | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| V9 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `00e0cbc796d83ce9137f95f853d6262cac4a464782540ecd05276927267c8be1` |
| V9 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `af8baa9a4aac7f0de19caa55f43e6120010e7d6765e0dceaa7cb18e95a88888f` |
| V9 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `43142be57b105bacf90124223c67d93372482ae0eeb64f4e9a8658f5a951909e` |
| V9 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `94cbe45f290f92a2a5ffaf7e87063e78e1aec17ba8d4fcae9e799e2235374246` |
| V9 synthetic support | `lewm/tests/n5_hierarchical_first_hit_v9_synthetic_execution.py` | `fd12a7dd1d877e507a0d332e4d96e684cc989fe0242fe1ee6ac61598d5702d3e` |
| V9 loss tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `5bb9e1c31e26ef4d4490013b9d377db161fa5ecde7471d4fa9ca4eb44a6a227b` |
| V9 lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_lifecycle.py` | `d7a7048d2242be98aec9f7e2d66d4121d0e5f67e65c9d51292c08b311e7053ee` |
| V9 handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_implementation_handoff_2026-07-13.md` | `50e22a56d2cb49e3b449aa760883c22dec1521abbd0d1b43fdbd0a69c5f374f2` |
| V9 review JSON | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_independent_review_2026-07-13.json` | `20d5abd9327267c5e40a66b464fd6589d30704ee8be7b919cadfd52b30350016` |
| V9 review report | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_independent_source_review_report_2026-07-13.md` | `0e930ef2bd0d0753f4928c69a462de2c05bf13d3e62139a0079e12a66e815522` |
| V9 review QA | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_independent_qa.py` | `8efaaecc2cea0815b31dc883b179d39e65bbd59337c5c9607ca02b2a9ed31119` |
| V9 reservation | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v9/attempts/seed_20260710/n5/reservation.json` | `184628c4518f0a3e7411561ee7f9ed83da1f89c9af7d729ed3e6ffe76ce0f1a2` |
| V9 terminal failure | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v9/attempts/seed_20260710/n5/failed.json` | `285c7bf38975a1ca13063d7b7ca36b31aa1b966cd206e0a418c07198c0719a3a` |

The V9 review, reservation, and failure content SHA-256 identities are
`8d7edcefce04d85a042558aa7ccc638c8da8e0690fcc36d9cff15e99bc6a0347`,
`1ad75999d8d88e9fa3599bec97fe9b18c2d8b893c372cb263dd8a0fa748449e0`,
and `1c22542a02e9e2707872df36c72bc790ca5fe06e57b0e03c63c40c2f6c2ebf7a`.

## Frozen scientific treatment

V10 must preserve V9 science exactly:

- the frozen V9 hierarchical first-hit loss source above;
- `ObservableCameraRayEvidenceV4Model` with no capacity change;
- the exact five-frame panel and target/raster construction;
- seed `20260710`, AdamW, learning rate `1e-4`, weight decay `1e-4`, gradient
  clipping at `1.0`, float32, and no autocast;
- exactly 4,000 full-panel optimizer updates and 20,000 frame exposures under
  schedule SHA-256
  `fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`;
- four equal `0.25` loss components, including equal-weight presence plus
  conditional-depth NLL;
- diagnostics at update 1 and every 100 updates through 4,000;
- final-update-only checkpoint selection;
- matched and wrong-RGB evaluation with independent reconstruction; and
- all 26 unchanged N5 metric thresholds and gate arithmetic.

V10 may not inspect or use the deleted V9 checkpoint/result, infer a V9 score,
change a threshold, add early stopping, select a best update, or use V9 failure
as numerical supervision. Any scientific change requires another amendment.

## V10 source namespace

Implementation author is `/root/coordinator_v2_qa`, distinct from the
amendment author. The frozen candidate must bind exactly these production
sources:

1. retained V9 loss source above;
2. `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py`;
3. `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py`;
4. `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py`;
5. `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py`.

The proof closure is:

1. `lewm/tests/n5_hierarchical_first_hit_v10_synthetic_execution.py`;
2. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py`;
3. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_lifecycle.py`;
4. `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_implementation_handoff_2026-07-14.md`.

The canonical different-agent review is
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_independent_review_2026-07-14.json`.
The reviewer must start with `/root/` and differ from `/root` and the
implementation author. Only a canonical `PASS` binding every source and proof
may authorize one exact attempt.

## Exact namespace and authority

The only exact output root is
`.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v10`.
It permits exactly one fresh attempt at
`attempts/seed_20260710/n5`. The V9 output is read-only terminal evidence. V10
must preflight its frozen identities but may not open V9 checkpoint/result
paths because those artifacts do not exist and are not licensed.

The production executor remains one isolated, canonical-path-only synchronous
operation. It owns reservation, fresh training, fresh child verification,
parent validation, metric publication, gate finalization, and terminal
failure. GPU execution is pinned only to discrete GPU0/R9700; the Raphael iGPU
is forbidden. RGB workers are in `[1,5]`; native math threads are fixed to one.
V10 exact work is serialized against every other `.generated` mutator.

## Phase-tagged child protocol

The V10 verifier child must use a closed phase vocabulary:

1. `request_read`
2. `request_parse`
3. `request_preflight`
4. `bundle_validation`
5. `checkpoint_validation`
6. `resource_validation`
7. `input_reconstruction`
8. `matched_evaluation`
9. `wrong_rgb_evaluation`
10. `gate_reconstruction`
11. `response_construction`

Before each phase, the child records only the phase name in memory. Any caught
exception produces one canonical, self-hashed failure envelope on stdout and
no fallback. The envelope has fixed fields for schema, status, phase,
exception class, sanitized bounded message, bounded source-relative traceback
frames, request content SHA-256 if parsed, and content SHA-256. It may contain
no tensors, RGB bytes, labels, predictions, losses, metrics, model values, or
checkpoint payload.

Fixed bounds are:

- message: at most 512 ASCII characters;
- traceback: at most 12 frames, each only repository-relative path, function,
  and positive line number;
- envelope: at most 64 KiB;
- all longer values reject or truncate with an explicit boolean flag.

Signals, interpreter startup failures, OOM termination, and timeout may
prevent a child envelope. The parent must still diagnose the process boundary.

## Parent observation and durable failure

The parent must retain, for every child completion:

- non-boolean integer return code and derived nonnegative signal or null;
- timeout flag;
- stdout/stderr byte counts and full SHA-256 values;
- explicit 64 KiB capture-overflow flags;
- sanitized bounded excerpts of at most 2,048 ASCII characters per stream;
- request content SHA-256 and frozen artifact file/content bindings;
- parsed child envelope plus its content SHA-256, or a closed reason from
  `absent`, `malformed`, `oversized`, or `schema_mismatch`.

On any non-success condition, the parent must first publish and fsync
`verification_failure.json` inside the claimed attempt. It must be canonical,
self-hashed, non-authoritative, bind the observation above, and grant every
exact/later-rung/checkpoint/retry/downstream license as false. Only after that
durable publication may owned checkpoint/result/completion cleanup begin.

The final `failed.json` must bind the diagnostic path, file SHA-256, content
SHA-256, and byte count, plus every cleanup outcome. The diagnostic survives
cleanup. If diagnostic publication or fsync fails, scientific artifacts must
not be deleted; the attempt instead terminalizes as
`diagnostic_publication_failed_preserved_owned_artifacts`, with all licenses
false. No path may silently lose both the scientific artifacts and the child
failure evidence.

Success still requires return code zero, empty stderr, one bounded canonical
success response, a matching nonce/request hash/process/environment/source/
artifact closure, independently reconstructed metrics, and parent validation.
There is no fallback, repair, retry, or in-process verifier path.

## Mandatory real-subprocess proof

Mocked `subprocess.run` unit tests remain useful but are insufficient. V10
source review must execute a real fresh
`sys.executable -I -B ... --verification-child` boundary on CPU with all
accelerators hidden.

The production executor may expose one source-reviewed
`--cpu-verifier-contract-smoke` mode. It accepts no source review, path, seed,
checkpoint, data, output, or authority argument; uses only fixed in-memory or
temporary synthetic tensors under a private temporary directory; marks every
request, response, and checkpoint `production_eligible=false`; cannot call
the exact reservation/training/finalization operations; and removes its
temporary tree before return.

The smoke must run the real serialization, stdin/stdout transport, `-I -B`
flags, PID/parent binding, sanitized environment, bundle/checkpoint validation,
fresh CPU model-state load, five-frame matched and wrong-input computation,
response parsing, and parent validation. Exact-child code must reject the
synthetic schema, and smoke-child code must reject the exact schema.

Author and reviewer tests must prove real-subprocess success and injected
failure at every closed phase, plus timeout, signal, nonzero, malformed,
oversized, and stderr cases. They must prove caps/hashes/truncation, diagnostic
fsync-before-cleanup ordering, diagnostic survival, preserved-artifact behavior
when diagnostic publication fails, no numerical payload in diagnostics, and
all false licenses. At least one test must execute the actual V10 script as a
fresh process rather than monkeypatching `subprocess.run`.

## Retained lifecycle and science proof

V10 tests and different-agent review must also prove:

- normalized V9/V10 model, loss, training, evaluation, metric, gate, and
  schedule AST equivalence outside the named diagnostic/protocol changes;
- exact V9 schedule hash, 4,000/20,000 counts, 41 diagnostics, four equal loss
  names/weights, and final-update-only checkpoint metadata;
- V9 trainer/verifier evaluation equality after fresh state roundtrip;
- all V9 retained lifecycle mutation, cleanup, isolation, source-binding, and
  terminal-failure tests;
- fixed V10 paths/schemas and absence of V9 checkpoint/numeric access;
- canonical source review, fresh exclusive attempt, and one-shot terminality;
  and
- no dynamic plugin/import path, caller-supplied backend, alternate opener,
  arbitrary callback, test outcome promotion, GPU1 path, threshold change, or
  retry API.

All source/review tests are CPU-only, use temporary roots, hide every
accelerator, cap native math threads at one, and open no canonical data, RGB,
checkpoint, V10 output, G2, held-out, runtime, hardware, or production path.

## Execution sequence

1. Freeze this amendment before any V10 source.
2. The fixed non-root author constructs and freezes the exact source/proof
   closure without exact/data/GPU work.
3. A different agent rehashes all parents, executes the real-subprocess and
   retained closures, and publishes canonical `PASS` or `BLOCK` last.
4. Only `PASS` may authorize one fresh V10 attempt on GPU0, serialized against
   all `.generated` mutation.
5. A full unchanged 26-check numerical pass may license design/review of the
   preregistered representation ladder. Any failure is terminal, grants no
   retry, and must retain the new bounded diagnostic if verification failed.

## Explicit non-authority

This amendment licenses only V10 source construction and different-agent
review. It does not authorize exact execution, data/RGB opening, checkpoint
use, GPU use, V9 retry, N16/N320, a second seed, shared-JEPA training,
selection, calibration, G2, held-out, runtime, navigation, hardware,
production, promotion, deployment, or any numerical claim.
