# Camera-ray N5 hierarchical-first-hit V9 implementation handoff

Date: 2026-07-13

Implementation author: `/root/coordinator_v2_qa`

Status: **frozen source candidate for different-agent review; no exact authority**

## Scope completed

This candidate implements the single fresh V9 scientific successor authorized
by the frozen amendment. It changes only:

1. the ordered first-hit objective to the preregistered hierarchical loss; and
2. the convergence budget to 4,000 full-panel updates.

The model, target/calibration, normalization, raw outputs, physical raster,
optimizer family and hyperparameters, matched/wrong-RGB evaluation, metric
accumulator, 26 numerical thresholds, resource contract, and reviewed V8
filesystem/verifier lifecycle remain retained.

The hierarchical loss computes normalized no-hit and hit-depth log
probabilities from the existing ordered hazard output, then applies:

- equal target no-hit/target hit presence group means;
- equal represented target-depth-bin conditional means;
- `0.5 * presence_nll + 0.5 * conditional_depth_nll`; and
- four equal top-level weights of `0.25`.

The fixed schedule is 4,000 optimizer updates, five distinct panel frames per
update, and 20,000 frame exposures. Its SHA-256 is
`fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`.
Diagnostics are exactly update 1, every 100 updates, and update 4,000. Only the
post-update-4,000 model is serialized; no best-loss, early-stop, averaging,
gate selection, repair, or retry path exists.

## Frozen candidate

| Role | Path | SHA-256 |
| --- | --- | --- |
| Pure V9 loss | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| Authority-free V9 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `00e0cbc796d83ce9137f95f853d6262cac4a464782540ecd05276927267c8be1` |
| Exact trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `af8baa9a4aac7f0de19caa55f43e6120010e7d6765e0dceaa7cb18e95a88888f` |
| Compute-only verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `43142be57b105bacf90124223c67d93372482ae0eeb64f4e9a8658f5a951909e` |
| Lifecycle-owning executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `94cbe45f290f92a2a5ffaf7e87063e78e1aec17ba8d4fcae9e799e2235374246` |
| Synthetic lifecycle | `lewm/tests/n5_hierarchical_first_hit_v9_synthetic_execution.py` | `fd12a7dd1d877e507a0d332e4d96e684cc989fe0242fe1ee6ac61598d5702d3e` |
| Loss/science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `5bb9e1c31e26ef4d4490013b9d377db161fa5ecde7471d4fa9ca4eb44a6a227b` |
| Lifecycle/contract tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_lifecycle.py` | `d7a7048d2242be98aec9f7e2d66d4121d0e5f67e65c9d51292c08b311e7053ee` |

The handoff itself is the fourth proof artifact and must be hashed by the
reviewer after these bytes are frozen.

## Frozen authority

| Role | Path | SHA-256 |
| --- | --- | --- |
| V9 amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_preimplementation_amendment_2026-07-13.md` | `ccc8097b4d3bd70aabf3c701226928e360fafb04a12a452c4fd406e9bba3db0a` |
| V8 diagnosis | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_v8_numeric_failure_diagnosis_and_successor_design_2026-07-13.md` | `ece7c960f49748776cd73f029e144f91f4c0723908e234a7e71173047777ee9a` |

The policy rehashes those documents, every retained dependency and prior
closure, and the identity-only V8 result/metric/gate evidence before exact
execution. It never parses or returns V8 numerical payloads and contains no V8
checkpoint path or load.

The reviewed V8 source bytes remain unchanged:

| V8 artifact | SHA-256 |
| --- | --- |
| Amendment | `9d89acd880849e688480eddddfc8b3129570b0f0fffa7bbf54dc457ade2ec211` |
| Policy | `99a2777d3ba2ad8baf62b98944f05aa1affb2e74834f337a2ba0644e9c03c84c` |
| Executor | `f163aaf04722bb118796912bcfcdf1f4e24b7e54990e41a9d164acc08b233500` |
| Synthetic lifecycle | `4d11b499d4cc2ffe4a31d0ed5df73a84649947bfd8a78522556719f8af21316c` |
| Lifecycle tests | `700092f5ea2885e23dba03b65c5a24737060c20e934413af1886ff454ec3e5b4` |
| Handoff | `536f31de0d8fe0cec26417b73e29ff3ef396086b05d5b2f104e7202f98df25b1` |
| Review narrative | `5939c60585ddb4b1227cd85fa359de23b11b6aff727ac60569d3959c5451f19b` |
| Review QA | `14ba6f544e6583011cfaceb0ff3b29aac8ac615045408a3fa49f066d706c94d8` |
| Review JSON | `fd095eea8b1f2a0cde67f77a3bd2338f8f13e3a81d824777475600a258758f0f` |

## Verification completed

The complete source-only suite ran with all accelerators hidden, native math
threads capped at one, and third-party pytest plugin autoload disabled:

```text
146 passed in 5.62s
```

It covers hand arithmetic, old group-weight rejection, both invariances,
finite gradients for hit/no-hit and represented bins through logits of
`+/-10000`, empty groups, invalid probability rejection, trainer/verifier
branch parity, exact schedule membership and hash, new-name acceptance and
old-name rejection, strict training/result/checkpoint schema, fresh
initialization, final-only selection, unchanged metric/wrong-RGB/26-threshold
paths, executor isolation, compute-only verifier behavior, complete
descriptor-relative lifecycle transactions, recovery, adversarial filesystem
mutations, cleanup, timeout/nonzero/signal/stderr/extra-output rejection, and
no retry.

Both source-only CPU smokes independently reported:

- update count `4000`;
- frame exposures `20000`;
- every update contains all five frames exactly once;
- the frozen schedule hash above;
- 41 diagnostic indices ending at update `4000`;
- the four V9 loss names and exact `0.25` arithmetic; and
- `checkpoint_selection = final_update_only`.

All eight Python artifacts compile and pass `pyflakes`. All nine candidate
artifacts are ASCII-only.

## Access declaration

Implementation and verification opened no canonical V9 output, experiment
data/RGB, checkpoint, model output, V8 numerical payload, GPU, G2, held-out,
selection, calibration, runtime, hardware, production, or promotion namespace.
No exact training or inference was run. Checks used source text,
hand-constructed tensors, temporary synthetic directories, and CPU-only
contract generation with accelerator visibility empty.

## Reviewer checklist

The reviewer must differ from both amendment author `/root` and implementation
author `/root/coordinator_v2_qa`. Review must bind all five production sources
and all four proof artifacts, including this handoff, by exact file SHA-256.

At minimum, independently verify:

1. the loss equations, invariances, empty-group gradients, and old-loss
   rejection;
2. exact 4,000/20,000 schedule construction and final-update-only state;
3. unchanged model/calibration/raster/metric/wrong-RGB/threshold behavior;
4. fresh initialization and the absence of every V8 checkpoint/numeric input;
5. V9 result, checkpoint, verifier, metric, and gate schema binding;
6. the isolated compute-only child and parent-only metric publication;
7. V8's complete no-follow, journaled, descriptor-relative lifecycle and
   terminal cleanup; and
8. the source-only access declaration and frozen byte identities.

The canonical review JSON must be published last. Only a canonical PASS review
may authorize the one exact V9 attempt. This handoff does not authorize exact
execution, retry, N16, a second seed, V5/shared-JEPA training, checkpoint use
beyond metric verification, G2, held-out navigation, runtime, hardware,
production, promotion, or deployment.

N5 remains an observability/memorization gate. Even a future numerical PASS
does not demonstrate novel-maze generalization; the separately gated larger
and scene-disjoint roles remain required before navigation work.
