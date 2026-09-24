# N5 full-panel V8 isolated-verifier implementation handoff (2026-07-13)

## Status and authority boundary

V8 source construction is complete and frozen for different-agent review.
Implementation author: `/root/camera_v5_independent`.

This handoff does not authorize exact execution. The canonical V8 review JSON
and output root are absent. No author test trained a model, opened experiment
RGB/data, loaded a production checkpoint, used a GPU, or mutated V7.

The frozen additive amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v8_isolated_verifier_amendment_2026-07-13.md`

SHA-256:
`9d89acd880849e688480eddddfc8b3129570b0f0fffa7bbf54dc457ade2ec211`.

## Diagnosed V7 failure

V7 successfully trained, then failed during verification. Training had already
called `configure_determinism`, whose implementation calls
`torch.set_num_interop_threads(1)`. The frozen verifier called that same
function again in the same process. PyTorch rejects a second inter-op setter
call after the thread lifecycle has started.

A CPU-only diagnostic subprocess reproduced the boundary exactly: its first
`configure_determinism` call succeeded and its second call raised
`RuntimeError: cannot set number of interop threads after parallel work has
started or set_num_interop_threads called`.

V8 does not change the frozen trainer or verifier. It makes frozen verifier
computation the only inter-op-setter path in a fresh process.

## Frozen V8 artifacts

| Artifact | SHA-256 |
| --- | --- |
| Amendment | `9d89acd880849e688480eddddfc8b3129570b0f0fffa7bbf54dc457ade2ec211` |
| Policy | `99a2777d3ba2ad8baf62b98944f05aa1affb2e74834f337a2ba0644e9c03c84c` |
| Executor | `f163aaf04722bb118796912bcfcdf1f4e24b7e54990e41a9d164acc08b233500` |
| Synthetic lifecycle | `4d11b499d4cc2ffe4a31d0ed5df73a84649947bfd8a78522556719f8af21316c` |
| Author/adversarial tests | `700092f5ea2885e23dba03b65c5a24737060c20e934413af1886ff454ec3e5b4` |

The source review must bind exactly these successor sources:

- `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v8.py`
- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v8.py`

No listed artifact may change without invalidating this handoff and requiring a
new freeze and review.

## V7 terminal closure retained

V8 rehashes the complete frozen V7 source/review closure and validates the V7
terminal directory by exact inventory, type, mode, byte count, file hash,
content hash, failure semantics, empty derived directories, and lock state.
It opens no V7 numeric artifact; none survives.

The terminal receipts remain:

| Receipt | File SHA-256 | Content SHA-256 | Bytes |
| --- | --- | --- | ---: |
| `reservation.json` | `de5972f40743cf960d3a2b2745087504deb0a7e69c0c8ff4f557269e6563a661` | `05fd35e99471659cf507d0985f3a7e82f276456a570b135bc7b51cf8ebfc8334` | 6292 |
| `failed.json` | `fec22c763a0be6ab4796aeea88e6d9d7a59d428f2801ab1d8f827536cf5a7957` | `546a31907b63efdfdf08ed7e793ba14bba1cf6e922f2daaba24c26f21e0a0b94` | 1143 |

The failure remains stage `verification`, class `runtime`, code
`execution_failure`; journal integrity is intact and retry remains false. The
V7 policy, executor, synthetic lifecycle, tests, amendment, handoff, review,
canonical PASS JSON, reservation, and failure hashes all rehashed exactly at
author freeze.

## Isolated verifier implementation

The V8 parent retains exclusive ownership of the attempt descriptor chain,
owned-directory journal, publication, cleanup, terminalization, and final gate.

After training, the parent:

1. rehashes all four attempt artifacts through retained claim descriptors;
2. creates one canonical self-hashed request with a fresh 256-bit nonce,
   reviewed source bindings, artifact `PATH:SHA256` bindings, parent PID,
   executable/executor identity, child mode, environment, and frozen contract;
3. starts exactly `sys.executable -I -B <same V8 executor>
   --verification-child`, with `close_fds=True`, a one-hour timeout, canonical
   request bytes on stdin, and captured stdout/stderr;
4. sanitizes Python variables; sets every native math thread cap to one; binds
   `HIP_VISIBLE_DEVICES=0`; removes CUDA, ROCr, ordinal, HSA selectors and the
   HSA override; and applies the same selector protection to the outer exact
   relaunch;
5. rejects timeout, signal/nonzero exit, any stderr, empty/malformed/
   noncanonical/extra/oversized stdout, or any request/response binding failure
   without fallback;
6. independently validates the returned frozen metric receipt against the
   retained reservation, result, checkpoint, completion, evaluation, resource,
   access ledger, licenses, and frozen gate arithmetic; and
7. only then publishes the metric receipt through the parent's journal and
   invokes retained finalization.

The fresh child independently revalidates static authority, source review,
source files, process flags, environment, request, and canonical artifact
bundle. It calls frozen verifier `_validate_attempt_bundle` and
`_compute_receipt` directly exactly once. It does not pre-call
`configure_determinism`, call the verifier publishing entry point, or contain a
canonical writer. A compatibility `write_exclusive` always raises. Its only
result is one canonical self-hashed stdout response declaring
`publication_performed=false` and binding the child PID/parent PID, resolved
executable/executor, mode, flags, environment, sources, artifacts, request,
nonce, and receipt.

Every failure remains inside the inherited V7 descriptor-relative transaction
and terminal cleanup path. Exact registered owned artifacts are removed;
changed or foreign bytes are preserved invalid; descriptors close; retry stays
false. There is no in-process verification fallback.

## Frozen science and device contract

V8 changes only verifier process lifecycle and uses the fresh root:

`.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v8`

Seed, N=5 train panel, model, initialization, AdamW settings, 400 updates,
batch size, 2,000 frame exposures, schedule hash, float32/no-autocast behavior,
losses and weights, gradient clipping, final-update checkpoint, matched and
wrong-RGB evaluations, thresholds, and final gate are unchanged. Exact work is
R9700 GPU0 only; the Raphael iGPU remains forbidden. At most five RGB workers
and one native math thread per process remain fixed.

## Author verification

External pytest plugins were disabled, native math threads were capped at one,
and all accelerator selectors were empty for author tests.

```text
V8 author/adversarial/source/lifecycle suite:       129 passed in 5.52s
Retained V6 author suite:                            40 passed in 1.32s
Applicable retained V1-V5 author closure:           103 passed, 8 deselected
V8 real isolated CPU contract smoke:                PASS
V8 static authority and V7 terminal rehash:          PASS
py_compile for all four V8 code/test artifacts:      PASS
forbidden refresh/recursive cleanup/interop scan:    PASS
git diff --check for all V8 artifacts:               PASS
canonical V8 output root:                            ABSENT
canonical V8 source-review JSON:                     ABSENT
exact execution or optimization:                     NOT RUN
production payload, checkpoint, data, or GPU access: NONE
```

The eight retained deselections are the same historical assertions documented
by V6 and V7 that consumed V5 review/output state must be absent. No applicable
science, schedule, security, lifecycle, or numerical assertion was deselected.

The isolated smoke reproduced 400 updates, 2,000 exposures, five-frame panels,
schedule SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`,
and synthetic total loss `0.265`. It opened no experiment input and used no
accelerator.

The V8 suite covers exact child command/environment capture, HSA and other
selector injection, canonical request/response binding and tamper matrices,
parent-only single publication, timeout/nonzero/signal/stderr/empty/oversize/
malformed/noncanonical/extra-output failures, no fallback, child write denial,
frozen-compute call count, dispatch isolation, V7 closure, and all inherited
transaction/recovery/cleanup adversarial cases.

## Required different-agent review

The reviewer must be a different agent from `/root/camera_v5_independent` and
should independently:

1. rehash the five frozen V8 artifacts and the full retained V7 closure;
2. reproduce the one-call-versus-second-call inter-op-thread lifecycle failure
   in CPU-only subprocesses and confirm frozen `_compute_receipt` is the only
   setter path in the fresh V8 verifier child;
3. audit the exact subprocess command, `-I -B` flags, PID/executable/executor,
   nonce, request/response, source, artifact, device, thread, timeout, stderr,
   size, publication, and no-fallback bindings;
4. verify that the child has no writer and the parent validates before its
   single metric publication and retained finalization;
5. exercise selector conflicts, request/response tampering, invalid receipts,
   every subprocess failure class, terminal cleanup, changed/foreign artifact
   preservation, and retained V7 immutability;
6. run the V8, V6, and applicable V1-V5 CPU-hidden suites, isolated smoke,
   static preflight, compile, forbidden-symbol, and diff checks; and
7. only on PASS, create the canonical self-hashed review JSON at
   `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v8_independent_review_2026-07-13.json`
   from `expected_source_review_core`, binding exactly the frozen policy and
   executor hashes above.

Review must not run exact training, open experiment data/RGB/checkpoints, use a
GPU, create the V8 output root, mutate V7, or authorize any later rung,
navigation, held-out evaluation, production, promotion, or retry.
