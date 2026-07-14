# V4 reviewed execution successor candidate

Date: 2026-07-13

Status: **candidate source closure; different-agent review required; no V2
verifier, finalizer, or training command has run**

## What failed

The immutable seed-`20260710`, N=5 training attempt completed. The predecessor
metric verifier then stopped before RGB decode, inference, receipt construction,
or publication with:

`PermissionError: V4 spawned RGB terminal differs from captured source`

The failure is recorded in
`docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_prepublication_failure_2026-07-13.md`
at SHA-256
`d99fc34ca6584348a3a67939722928287affa925b18ed895ef23f6e1e3954842`.

The first successor draft repaired metric verification and finalization, but a
downstream audit found a second closure gap: the frozen trainer imports
prerequisite validators from the predecessor finalizer. N=16/32/320 and all
seed-`20260711` training would therefore re-enter the failed verifier while
reopening their prerequisite gates.

## Complete additive repair

The candidate now has four reviewed sources:

| Source | SHA-256 |
|---|---|
| `scripts/verify_go2_observable_camera_ray_fit_v4_metrics_v2.py` | `4640b4e9a65221f9825140aa58f69ce66e51e06dd1ffb28d77355cadacf377e3` |
| `scripts/finalize_go2_observable_camera_ray_fit_v4_ladder_v2.py` | `150ba5f56ff2c5c47794cf87c22b959309856d9a2d921d277d5bf2e2e207d101` |
| `scripts/train_go2_observable_camera_ray_fit_v4_v2.py` | `c9d22fb38acdf5fd3099271661dc65bb9cea989426a3b6021ad28649d6dd74d3` |
| `scripts/launch_go2_observable_camera_ray_fit_v4_v2.py` | `65c58e36cb97d155a58ec1cbc93a1f2f42a75e62f049b5d8e874481a435a614b` |

Focused tests:

| Test source | SHA-256 |
|---|---|
| `lewm/tests/test_verify_go2_observable_camera_ray_fit_v4_metrics_v2.py` | `7a8946606252aa7ba680e3e1039e964400da20d7291a15dab2bfed9d287f41d1` |
| `lewm/tests/test_finalize_go2_observable_camera_ray_fit_v4_ladder_v2.py` | `862bc778c45d02001497cc4f38cf62c673c0a9f16aa1a1fdfba301aac42d75ef` |
| `lewm/tests/test_train_go2_observable_camera_ray_fit_v4_v2.py` | `bbdcda5f935f2af224a9313105e42ff0cb1032487091629fe1755ac4607d7093` |
| `lewm/tests/test_launch_go2_observable_camera_ray_fit_v4_v2.py` | `1d536f822b9a9a652b9bdb7f46120e1086bb657ec2df14d07506dbcb1e95b19e` |

Verifier V2 preserves the frozen metric code, receipt schema, target
partitions, checkpoint validation, matched/wrong RGB comparison, metric
authorization, exclusive receipt path, and GPU policy. It uses the frozen
trainer's inline RGB decoder with exactly `fit_size` opens/decodes, one worker,
and one native CPU thread. It never trains.

Finalizer V2 captures verifier V2 and preserves the existing stage, seed, and
two-seed gate schemas. Its execution validators recursively reopen and
byte-reproduce the complete prerequisite chain through verifier V2.

Trainer V2 is a mechanical successor of the frozen trainer. An AST comparison
requires every top-level computational function to remain identical. The only
permitted function changes are the execution shell:

- import launcher V2 and finalizer V2;
- bind the trainer source through the successor review;
- pass the successor review hash into spawned RGB workers; and
- expose the successor review arguments to the private captured entry point.

The dataset, selection, target, model, losses, optimizer, schedule, evaluation,
reservation, checkpoint, result, completion, and gate schemas are unchanged.
Training still uses up to six CPU decoder processes with one native thread each.
The learned model work remains on the separately authorized discrete GPU; the
Raphael iGPU is rejected by the frozen runtime policy.

Launcher V2 is stdlib-only until two authorities pass: the frozen trainer
authorization and the new successor review. Both its main terminal and every
spawned RGB terminal revalidate the review, capture all four successor sources,
compare the live launcher with the captured launcher, and load trainer V2.

The frozen result's `source_map_sha256` continues to identify the original
computational closure. The additive execution shell is bound separately by the
canonical successor review and caller hash. This separation is deliberate: it
keeps the completed N=5 bytes and all frozen artifact schemas unchanged. The
different-agent review must explicitly approve this authority split.

## Immutable N=5 attempt

| Artifact | File SHA-256 | Content SHA-256 |
|---|---|---|
| `reservation.json` | `f5926ee9006df8d163a2d1a17882d82124608ddce319ea0fb5e80fcfe2c2a8aa` | `699b4e95ed05cb13a79fe6af8507fae5d987af9ff1977b0e4684f32742aa4943` |
| `checkpoint.pt` | `f1739c742f9c19d5e17753da504a547254eb6e1997bb1ac4eca8b188bbf1dcf0` | `589060417903167bbf9ce7605c906b25cd802edd73b79ec607c77403c6df305a` |
| `result.json` | `39030bb7928a6b078b03156dc9e14fb206c60c73ab2acac88bfd307c5a65bbfa` | `8c38e13f411a5cd9b03362cb5ac98379875065f284a75ac894706944ff252b61` |
| `completed.json` | `4fb9b5629f039ac16692ec6e171a8188f3bf8b7d052ac8cde26b8ac86c10f6af` | `48022dca829a73b7cbd3b665ac7679807825a9aefd56a48e752ae07e6eaa336f` |

The attempt inventory remains exactly these four files. No metric receipt
exists. N=5 training must not be rerun or altered.

The predecessor sources also remain unchanged:

| Source | SHA-256 |
|---|---|
| frozen trainer | `299980cdcb5ef561102f325bbb3db3dfd7aa8217b8a45446b0437badb8f27cfa` |
| frozen launcher | `71d95ae79cd90c64bee8b06f2787b336d72e2fca1e23fcb0cc52f921350a2ff4` |
| predecessor verifier | `235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f` |
| predecessor finalizer | `375b1dcd3a548cf7b130fb67291ef5116effcc0197a28be42643bfc59e710ec6` |

## Review lock

The required review record is deliberately absent:

`docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_finalizer_v2_independent_review_2026-07-13.json`

It must use schema
`lewm_go2_observable_camera_ray_fit_v4_execution_successor_review_v2`, be
self-hashed canonical JSON, and be authored by an agent other than
`/root/g5_perf_closure`. It must bind all four successor sources, the failure
record and phase, the predecessor verifier, all four immutable N=5 artifacts,
both seeds, every rung, the inline verifier decoder, the six-process training
decoder, one native thread per process, finalizer V2 for both prerequisite
classes, the frozen step schedule, GPU0/R9700, and exclusive receipt creation.

Its license permits only this reviewed execution successor under the already
frozen trainer authorization. It does not permit a training configuration
change, unreviewed training, held-out access, G2 access, runtime deployment, or
promotion.

## Ordered commands

No command below may run before the review exists and its exact file SHA-256 is
substituted for `REVIEW_SHA256`.

### 1. Reverify completed N=5

```bash
env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/andrewknowles/TinyQuadJEPA/bin/python -I -B \
  /home/andrewknowles/Workspace/LeWMQuad-v3/scripts/verify_go2_observable_camera_ray_fit_v4_metrics_v2.py \
  --successor-review /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_finalizer_v2_independent_review_2026-07-13.json \
  --successor-review-sha256 REVIEW_SHA256 \
  --metric-authorization /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_authorization_2026-07-12.json \
  --metric-authorization-sha256 091d26f6be0372c003528be370028e6f431bcdef9770ce3855d8b1cf4045a3cf \
  --trainer-authorization /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json \
  --trainer-authorization-sha256 d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802 \
  --trainer-review-record /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json \
  --trainer-review-record-sha256 c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea \
  --reservation /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/reservation.json \
  --reservation-sha256 f5926ee9006df8d163a2d1a17882d82124608ddce319ea0fb5e80fcfe2c2a8aa \
  --result /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/result.json \
  --result-sha256 39030bb7928a6b078b03156dc9e14fb206c60c73ab2acac88bfd307c5a65bbfa \
  --checkpoint /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/checkpoint.pt \
  --checkpoint-sha256 f1739c742f9c19d5e17753da504a547254eb6e1997bb1ac4eca8b188bbf1dcf0 \
  --completion /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/completed.json \
  --completion-sha256 4fb9b5629f039ac16692ec6e171a8188f3bf8b7d052ac8cde26b8ac86c10f6af \
  --seed 20260710 --fit-size 5
```

### 2. Finalize N=5

After step 1 publishes the metric receipt, invoke finalizer V2 with the exact
published receipt hash:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python -I -B \
  /home/andrewknowles/Workspace/LeWMQuad-v3/scripts/finalize_go2_observable_camera_ray_fit_v4_ladder_v2.py \
  --successor-review /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_finalizer_v2_independent_review_2026-07-13.json:REVIEW_SHA256 \
  stage \
  --reservation /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/reservation.json:f5926ee9006df8d163a2d1a17882d82124608ddce319ea0fb5e80fcfe2c2a8aa \
  --result /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/result.json:39030bb7928a6b078b03156dc9e14fb206c60c73ab2acac88bfd307c5a65bbfa \
  --checkpoint /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/checkpoint.pt:f1739c742f9c19d5e17753da504a547254eb6e1997bb1ac4eca8b188bbf1dcf0 \
  --completion /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/seed_20260710/n5/completed.json:4fb9b5629f039ac16692ec6e171a8188f3bf8b7d052ac8cde26b8ac86c10f6af \
  --metric-verification /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/metric_verifications/seed_20260710_n5.json:N5_METRIC_RECEIPT_SHA256 \
  --trainer-authorization /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json:d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802 \
  --trainer-review-record /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json:c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea \
  --seed 20260710 --fit-size 5
```

### 3. Train seed 20260710 N=16 through the V2 path

This command is authorized only if the N=5 stage gate passes. Replace
`N5_STAGE_GATE_SHA256` with its exact published file hash.

```bash
env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/andrewknowles/TinyQuadJEPA/bin/python -I -B \
  /home/andrewknowles/Workspace/LeWMQuad-v3/scripts/launch_go2_observable_camera_ray_fit_v4_v2.py \
  --successor-review /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_finalizer_v2_independent_review_2026-07-13.json \
  --successor-review-sha256 REVIEW_SHA256 \
  --dataset-manifest /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/v1/manifest.json \
  --dataset-manifest-sha256 2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85 \
  --audit-receipt /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json \
  --audit-receipt-sha256 2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c \
  --trainer-authorization /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json \
  --trainer-authorization-sha256 d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802 \
  --trainer-review-record /home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json \
  --trainer-review-record-sha256 c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea \
  --fit-size 16 --steps 1200 --batch-size 1 --eval-batch-size 1 \
  --learning-rate 0.0001 --weight-decay 0.0001 \
  --seed 20260710 --rgb-workers 6 --device cuda:0 \
  --previous-stage-gate /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/gates/stage_seed_20260710_n5.json \
  --previous-stage-gate-sha256 N5_STAGE_GATE_SHA256
```

N=32 uses `--steps 1600` and the passing N=16 stage gate. N=320 uses
`--steps 3200` and the passing N=32 stage gate. No larger rung may start before
its immediate predecessor passes through finalizer V2.

### 4. Start seed 20260711

Seed `20260711` N=5 is authorized only after all four seed-`20260710` stages
pass and finalizer V2 publishes `gates/seed_20260710.json`. It uses the same
launcher command as step 3 with these exact execution arguments:

```text
--fit-size 5 --steps 1000 --batch-size 1 --eval-batch-size 1
--learning-rate 0.0001 --weight-decay 0.0001
--seed 20260711 --rgb-workers 6 --device cuda:0
--seed-20260710-gate /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_observable_camera_ray_fit_v4/development_fit_v2/gates/seed_20260710.json
--seed-20260710-gate-sha256 SEED_20260710_GATE_SHA256
```

Its N=16/32/320 commands require both the same first-seed gate and the immediate
same-seed predecessor stage gate. Tests cover all eight seed/rung prerequisite
classes.

## Verification performed

- 29/29 focused successor tests passed with GPU use disabled and native thread
  caps set to one.
- 125/126 tests passed across the predecessor and successor V4 trainer,
  launcher, verifier, finalizer, metrics, and gate modules.
- The one remaining predecessor-test failure asserts that the V2 output root
  does not exist. That assertion predates the immutable completed N=5 attempt
  and is now intentionally false. The frozen predecessor test was not edited.
- Every candidate source compiles.
- AST closure proves every trainer computational function is unchanged.
- No verifier inference, training, checkpoint publication, metric receipt, or
  gate finalization command ran.

The next action is a different-agent byte, authority, and command review of the
final hashes. Only a passing canonical review can unlock step 1.
