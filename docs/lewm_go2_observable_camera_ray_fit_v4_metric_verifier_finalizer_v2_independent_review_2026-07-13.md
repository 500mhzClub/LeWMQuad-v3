# V4 execution successor independent review

Date: 2026-07-13

Reviewer: `/root/v4_execution_successor_review`

Implementation author: `/root/g5_perf_closure`

Decision: **PASS**

## Reviewed closure

| Item | SHA-256 |
|---|---|
| `scripts/verify_go2_observable_camera_ray_fit_v4_metrics_v2.py` | `4640b4e9a65221f9825140aa58f69ce66e51e06dd1ffb28d77355cadacf377e3` |
| `scripts/finalize_go2_observable_camera_ray_fit_v4_ladder_v2.py` | `150ba5f56ff2c5c47794cf87c22b959309856d9a2d921d277d5bf2e2e207d101` |
| `scripts/train_go2_observable_camera_ray_fit_v4_v2.py` | `c9d22fb38acdf5fd3099271661dc65bb9cea989426a3b6021ad28649d6dd74d3` |
| `scripts/launch_go2_observable_camera_ray_fit_v4_v2.py` | `65c58e36cb97d155a58ec1cbc93a1f2f42a75e62f049b5d8e874481a435a614b` |
| verifier V2 tests | `7a8946606252aa7ba680e3e1039e964400da20d7291a15dab2bfed9d287f41d1` |
| finalizer V2 tests | `862bc778c45d02001497cc4f38cf62c673c0a9f16aa1a1fdfba301aac42d75ef` |
| trainer V2 tests | `bbdcda5f935f2af224a9313105e42ff0cb1032487091629fe1755ac4607d7093` |
| launcher V2 tests | `1d536f822b9a9a652b9bdb7f46120e1086bb657ec2df14d07506dbcb1e95b19e` |
| successor candidate | `aa4bd82ffcab0bf69148545e22053875baaa08f819fc8e85dfb0cbbb270b909e` |
| predecessor verifier | `235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f` |
| predecessor finalizer | `375b1dcd3a548cf7b130fb67291ef5116effcc0197a28be42643bfc59e710ec6` |
| frozen trainer | `299980cdcb5ef561102f325bbb3db3dfd7aa8217b8a45446b0437badb8f27cfa` |
| frozen launcher | `71d95ae79cd90c64bee8b06f2787b336d72e2fca1e23fcb0cc52f921350a2ff4` |
| prepublication failure record | `d99fc34ca6584348a3a67939722928287affa925b18ed895ef23f6e1e3954842` |

The canonical PASS record is
`docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_finalizer_v2_independent_review_2026-07-13.json`:

- file SHA-256: `fc339ea4d42e83ac10e99d536dd076b4df50ef6fb2598b814596aafc6e866c41`
- content SHA-256: `493872c56cfe9898de7604e953d0247d08cc09b3ac5766a7725ecf5a929b0daa`
- schema: `lewm_go2_observable_camera_ray_fit_v4_execution_successor_review_v2`

## Findings

1. The failure lineage is exact. The predecessor verifier stopped with
   `PermissionError: V4 spawned RGB terminal differs from captured source`
   inside the captured trainer's RGB decoder, before RGB decode, inference,
   metric-receipt construction, or publication.
2. The immutable seed-`20260710`, N=5 attempt contains exactly
   `reservation.json`, `checkpoint.pt`, `result.json`, and `completed.json`.
   Their four raw hashes and four semantic content hashes reproduce the review
   record. No metric receipt or gate exists, so this review authorizes reopening
   that attempt for verification and does not authorize retraining it.
3. Verifier V2 preserves the predecessor metric, partition, checkpoint,
   matched/wrong-RGB, receipt, and GPU contracts. It invokes the frozen
   trainer's exact RGB decoder once with `maximum_workers=1`; the decoder opens
   and decodes exactly `fit_size` selected train images inline with one native
   thread and no spawned callback.
4. Finalizer V2 captures verifier V2 and recursively byte-reproduces every
   canonical metric receipt and artifact bundle. Same-seed N=16/N=32/N=320
   prerequisites and all seed-`20260711` first-seed prerequisites are validated
   through finalizer V2, including immediate-predecessor ordering.
5. Trainer V2's top-level function set is identical to the frozen trainer.
   AST comparison limits changes to the five reviewed execution-shell
   functions: RGB authority propagation, captured-source binding, captured
   exact dispatch, argument parsing, and direct-main rejection. Dataset,
   targets, model, losses, batches, optimizer, schedule, evaluation,
   reservation, checkpoint, result, completion, and gate computation are
   unchanged.
6. Launcher V2 and each spawned RGB terminal validate the canonical,
   caller-hashed successor review before the frozen trainer authorization and
   protected execution closure. Verifier V2 validates it before attempt or
   heavy imports. Finalizer V2 validates its caller-hashed source review before
   canonical gate/artifact processing and performs the complete review-policy
   validation before metric or gate publication. All runtime modules are read
   by exact hash into isolated private namespaces; imported-library computation
   and publication remain rejected.
7. The resource contract remains exact: `HIP_VISIBLE_DEVICES=0`, device
   `cuda:0`, exactly one visible `AMD Radeon AI PRO R9700`, at least 16 GiB,
   `HSA_OVERRIDE_GFX_VERSION` unset, and all native thread variables equal one.
   Raphael and multiple-visible-device configurations are rejected. Training
   decoding remains spawn-based with at most six processes; verifier decoding
   is inline with one worker.
8. The authority split is deliberate and approved. Frozen artifacts continue
   to bind source-map SHA-256
   `eb8c97dae6f3ef3839a886cac200774c87dfb6e452f71c13e75557eb8c9feac3`,
   whose trainer and launcher are the original frozen sources. The additive
   execution shell is a separate capability bound by the canonical review
   file hash and the four reviewed successor source hashes. Artifact schemas or
   completed N=5 bytes are not rewritten to create that authority.
9. The license is narrow: frozen-ladder metric reverification, stage
   finalization, and unchanged reviewed execution only. It does not authorize
   configuration changes, unreviewed training, held-out access, G2 access,
   runtime deployment, checkpoint use, or promotion.

## Reproduction

- Focused successor suite: **29/29 passed** with GPU visibility disabled and
  all native CPU thread caps set to one.
- Combined predecessor/successor trainer, launcher, verifier, finalizer,
  metrics, and gate suite: **125/126 passed**. The only failure is the frozen
  predecessor test asserting that `development_fit_v2` does not exist. That
  pre-attempt assertion is now necessarily false and does not test successor
  behavior.
- All four successor sources compiled from exact bytes.
- Immutable N=5 raw and semantic hashes, checkpoint source-map binding, frozen
  schedule binding, and non-authoritative/non-aggregating/non-promotable flags
  reproduced independently.

No verifier, finalizer, launcher, training, model inference, GPU, metric
publication, or gate-publication command was executed during this review.

## Authorized next action

The exact N=5 V2 verifier command in the candidate document may now run with
successor review file SHA-256
`fc339ea4d42e83ac10e99d536dd076b4df50ef6fb2598b814596aafc6e866c41`.
No later command is authorized unless its immediate V2 gate prerequisite
passes.
