# N5 full-panel V8 isolated-verifier amendment (2026-07-13)

## Status and scope

This is an additive, pre-implementation amendment to the frozen N5 full-panel
experiment. It authorizes construction and different-agent source review of
one V8 infrastructure-replacement lifecycle. It does not authorize exact
execution. No V8 implementation source existed when this amendment was frozen.

The sole V7 exact attempt is terminally consumed. Frozen training completed,
but verification failed before any metric receipt or gate was published. V7's
owned cleanup removed the checkpoint, result, and completion, preserved only
its reservation and terminal failure receipts, kept journal integrity intact,
and set retry authorization false. V8 is not a retry of V7 and may not read or
reuse V7 numeric state. It is one fresh infrastructure replacement for the
same frozen experiment.

## Frozen V7 closure

V8 binds this immutable V7 source and review closure:

| Artifact | SHA-256 |
| --- | --- |
| V7 owned-directory amendment | `17ca6b726d1eaa25662a1823b4c153d496f1e51502b764350ddd6a3a34f249da` |
| V7 policy | `ed50a00c0449c41031f076c5627f6501b93ee2931deaf4cbcd06a0f9e89d16e0` |
| V7 executor | `5043d42aaabb5a4852e9339a7d3e98c9d530c7ff403e5a2f1ac7a21999fbc14e` |
| V7 synthetic lifecycle | `9743786550ede91023b3d96cfa6650c04bd02a2c1a5d3fbb2364728b09980bf1` |
| V7 author tests | `0bf0f77ff5c773891ddd6ab5ed933b74132f0c8194e0aa237d93175619b7a858` |
| V7 author handoff | `020a26678670ac0067a090a2f3c4ba3634185f4a450c48a24c657cd263c9b6be` |
| V7 independent PASS review | `283e852e2dbacc55c61f32759cbd3c14af3a1670e67ca2523865224c5c016425` |
| V7 canonical PASS review JSON | `e581739ffdca18a3302d2fef527a43ef9bf31a87f35f4ca2a8e4cc75116d865e` |

The V7 PASS JSON has canonical content SHA-256
`378a0cd61610800ba65eff9a3ac382fa69640b0c50148f5a00a161bba2641def`
and byte count `19249`.

The sole V7 attempt inventory is exactly:

| Receipt | File SHA-256 | Content SHA-256 | Bytes |
| --- | --- | --- | ---: |
| `reservation.json` | `de5972f40743cf960d3a2b2745087504deb0a7e69c0c8ff4f557269e6563a661` | `05fd35e99471659cf507d0985f3a7e82f276456a570b135bc7b51cf8ebfc8334` | 6292 |
| `failed.json` | `fec22c763a0be6ab4796aeea88e6d9d7a59d428f2801ab1d8f827536cf5a7957` | `546a31907b63efdfdf08ed7e793ba14bba1cf6e922f2daaba24c26f21e0a0b94` | 1143 |

The terminal receipt binds failure stage `verification`, failure class
`runtime`, failure code `execution_failure`, removal of exactly
`checkpoint.pt`, `result.json`, and `completed.json`, intact owned-directory
journal integrity, no restored success eligibility, and no retry. The V7
metric and gate directories are empty. No V7 numeric payload survives or is
admitted by V8.

## Diagnosed lifecycle defect

The frozen verifier calls
`scripts/train_go2_observable_camera_ray_fit_v4_v2.py::configure_determinism`
inside `_compute_receipt`. That function calls
`torch.set_num_interop_threads(1)`. V7 invoked the verifier in the same Python
process after training, RGB worker activity, and Torch parallel work. PyTorch
permits the inter-op thread count to be set only before inter-op work begins or
before a prior inter-op setter call. Verification therefore raised at the
thread-lifecycle boundary after successful training.

V8 may not modify V7 or the retained verifier/trainer. It must put independent
verification in a fresh isolated process where the frozen verifier is the
first numerical workload.

## Required verifier process boundary

The V8 parent remains the sole owner of reservation, canonical publication,
journaling, cleanup, failure terminalization, and finalization. The verifier
child is compute-only and must never publish an authority-bearing file.

The parent must:

1. retain the V7 reviewed no-follow descriptor and owned-directory transaction
   model under a fresh V8 output root;
2. rehash the frozen V8 source review and all four completed-attempt artifacts
   through retained parent descriptors;
3. create a canonical, self-hashed request containing a fresh nonce, exact
   source-review binding, executor/policy hashes, canonical artifact paths and
   byte hashes, process contract, and environment contract;
4. spawn the same frozen V8 executor with `sys.executable -I -B` in a new
   process, a sanitized Python environment, one native thread per library,
   `HIP_VISIBLE_DEVICES=0`, no conflicting device selectors, no HSA override,
   closed unrelated file descriptors, bounded runtime, and request bytes on
   stdin;
5. require exit status zero, empty stderr, one bounded canonical JSON response,
   exact nonce/request/source/process/environment/artifact bindings, a valid
   frozen metric receipt, and an explicit `publication_performed=false` claim;
6. reject timeout, signal, nonzero status, stderr, extra stdout, noncanonical or
   oversized output, source/artifact mismatch, device mismatch, malformed
   receipt, or any missing binding without fallback or in-process recompute;
7. only after complete validation, publish the metric receipt through the
   parent's retained descriptor journal and run frozen finalization in the
   parent; and
8. on every failure, remove only exact registered owned artifacts, preserve
   foreign or changed bytes invalid, write the terminal failure receipt when
   possible, close all descriptors, and leave retry authorization false.

The child must independently revalidate static authority, the V8 source
review, its executable mode, sanitized environment, exact request, completed
artifact bundle, checkpoint state manifest, R9700 GPU0 resource, selected train
inputs, matched evaluation, and wrong-RGB control. It must call the frozen
verifier's validation and computation functions directly, never its publishing
entry point. Its only result channel is the canonical stdout response.

## Required adversarial evidence

Before different-agent review, CPU-only tests with every accelerator hidden
must prove:

1. the child command is exactly the isolated same-executor command and the
   environment/device/thread contract is exact;
2. request and response are canonical, size bounded, nonce-bound, self-hashed,
   source-review-bound, source-file-bound, artifact-bound, and process-bound;
3. the child computation path contains no canonical writer and the parent is
   the only metric/gate publisher;
4. nonzero exit, timeout, signal, stderr, extra stdout, malformed JSON,
   noncanonical JSON, oversize output, binding mismatch, invalid receipt, and
   attempted fallback all fail closed;
5. valid synthetic child output is published exactly once by the parent and
   then accepted by retained finalization wiring;
6. verification failure removes exact owned training and derived partials,
   preserves foreign/changed bytes, terminalizes durably, and cannot retry;
7. the V7 terminal inventory and receipts rehash exactly and no V7 numeric
   artifact is opened or admitted; and
8. all applicable V7 transaction, recovery, source, science, schedule, GPU,
   import-safety, and cleanup regressions remain unchanged.

No author or reviewer test may execute training, open experiment RGB/data,
load a production checkpoint, invoke a GPU, create the canonical V8 output, or
mutate the terminal V7 output.

## Frozen namespace and science

The new canonical root is:

`.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v8`

V8 preserves the numerical experiment exactly:

- seed `20260710`, fit size N=5, and the same five train frames;
- fresh `ObservableCameraRayEvidenceV4Model` initialization with no V5-V7
  checkpoint or state input;
- AdamW at learning rate `1e-4` and weight decay `1e-4`;
- 400 updates, batch size 5, and 2,000 frame exposures;
- schedule SHA-256
  `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`;
- float32, no autocast, gradient clipping norm `1.0`;
- the same four losses weighted `0.25` each;
- final-update-only checkpoint selection;
- unchanged matched-RGB and wrong-RGB-with-target-calibration evaluation,
  metric verifier, final gate, and thresholds; and
- GPU0 only on AMD Radeon AI PRO R9700, Raphael iGPU forbidden, at most five
  RGB workers, and one native math thread per process.

## Authority boundary

This amendment authorizes only V8 source construction and different-agent
review. Exact execution remains forbidden until a reviewer other than the V8
implementation author passes the complete frozen V8 closure. It grants no
retry, scientific retry, V7 numeric read, second V8 attempt, N16, second seed,
later training, checkpoint use beyond frozen metric verification, G2,
held-out, selection, calibration change, runtime, hardware, navigation,
production, or promotion authority.
