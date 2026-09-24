# Shared JEPA V5 full-training V2 implementation author handoff

Date: 2026-07-13

Implementation author: `/root/coordinator_v2_qa`

Status: **FROZEN AUTHOR CANDIDATE; DIFFERENT-AGENT REVIEW REQUIRED; EXACT EXECUTION BLOCKED**

## Authority boundary

This handoff freezes a source-only candidate for the V1 full-training science
plus the independently passed V2 execution amendments. It is not a source
review, preflight PASS, exact binding, execution authorization, checkpoint
qualification, G2/G3 license, navigation license, or production license.

The exact-execution manifest remains canonical and blocked. All 19 required
bindings are `null`, `exact_execution_authorized=false`, and every G2,
held-out, runtime, hardware, production, and promotion field remains false.
Neither the preflight nor exact namespace was opened while authoring this
candidate.

## Frozen parents

| Artifact | SHA-256 |
| --- | --- |
| V1 full-training amendment | `b21d01d062543cc7b7f3f5281f66ac40df76726c678a9364f7a4e451b035a4a7` |
| V2 amendment | `b521d2885b5dca1a72838282fbb8e193a21ec0f2db0e0a5950074506fba1f66d` |
| V2 independent test | `734a140f2b073e02970cb81897fd5edbb7beb28e56a60ba08f774df43f920e0b` |
| V2 independent review | `f4b22ef6061a54b08b2e2afa5f0e56ecbfa20a5a364f5eda0395d71722182dae` |
| V2 PASS record | `6a53a3c9d72da6499714883676f49a62d0c3ba61c2d2ccde741f1654e6f089d4` |
| Reviewed V5 model | `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9` |
| Reviewed V5 model test | `848aa8be369b89c973a4da916f9c7abeff47eca12aceb4304cf612ed4d53227b` |
| Output/loss review | `83dcd8f8702656c25f4584295827d0c82cf1db113abe2de4a417e7b528abff1f` |

The policy also rehashes the complete governing-design and staged-lifecycle
closure fixed in its `reviewed_source_bindings()` map. The live navigation
readiness document is context only and is excluded from authority.

## Frozen candidate bytes

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| policy | `lewm/benchmarks/go2_shared_jepa_v5_full_training_v2_policy.py` | `e0c3409ce104d954e40aa73ae5bd5b79ec3daa77564e90c6be183c2fbc19f680` |
| payload-free preflight | `scripts/preflight_go2_shared_jepa_v5_full_training_v2.py` | `fbc6d63394625d2c3ccc79821d9a07b507fdfb95e02ee1768ed6325857531eff` |
| preflight verifier | `scripts/verify_go2_shared_jepa_v5_full_training_v2_preflight.py` | `1453a6a6134c25cad21d41f44628e4cc8e1e041ae8994d570413ebb1101e09e3` |
| exact reserver/publisher | `scripts/execute_go2_shared_jepa_v5_full_training_v2.py` | `698fb92f2f854365f2d0bfbf6f034b1c3f04704a8d6227fceff7c3ed275fc271` |
| exact trainer | `scripts/train_go2_shared_jepa_v5_full_training_v2.py` | `bdd8e4b1c24e855f3e3ff535a195f2c370c4ffdadc48eb9e83b214b53362f23b` |
| independent exact verifier | `scripts/verify_go2_shared_jepa_v5_full_training_v2.py` | `d8950c8bf23b0bd5494c7c864f2f2543d533b0bc07af3f70287291227c872543` |
| author checks | `lewm/tests/test_go2_shared_jepa_v5_full_training_v2_implementation.py` | `2dd1053e17aa3adaa5705dfffc1f57e47407845f2dd12b8729696a387fbd3758` |
| blocked manifest | `docs/lewm_go2_shared_jepa_v5_full_training_v2_exact_execution_manifest_2026-07-13.json` | `b75f89b3cf23e3d444898aae707289df9872c7978a247106d77352bd50b8008d` |

The blocked manifest semantic SHA-256 is
`3b227cc2a837e9a5e4bbcbcd4606d7354bc3ddc7c7c2d45e218e12f32c7870fc`.

## Implemented boundary

The candidate provides:

- a standard-library authority policy and canonical blocked manifest;
- a descriptor-retained, exclusive payload-free preflight whose GPU smoke
  runs in a fresh isolated child and terminates before publication;
- an independent standard-library preflight verifier with exact source-ledger
  coverage;
- a standard-library exact reserver that writes the reservation before a fresh
  trainer, then starts a separate independent verifier and writes completion
  last;
- CPU FP32 V4 migration with the initialization seed applied after the V4
  container is loaded and immediately before V5 construction;
- the fixed 128,000-presentation schedule, two matched 8,000-update arms,
  trainable-parameter AdamW, four-way accumulation, clipping, EMA cadence, and
  eight checkpoints per arm;
- complete four-equal V4 supervision and the frozen established JEPA package,
  with no-JEPA changing backward membership only;
- update-zero diagnostics, eight promoted selection candidates, raw
  scope-level JEPA accumulators, deterministic physical controls, exact
  selection, and selected-update-only matched diagnostics;
- independent six-vector calibration, the complete threshold grid, all-class
  aggregate/family checks, and aggregate/family NLL non-regression;
- explicit family-to-scene ablation identity and raw promoted-minus-matched
  metric deltas, with no causal, selection, qualification, or retry effect;
- immutable artifacts, complete pre-completion inventory checks, completion
  rehash closure, and a fresh verifier that reopens every raw input and all 16
  checkpoints rather than trusting trainer metrics.

## Verification performed

All checks used one native CPU thread, empty CUDA/HIP/ROCr/HSA visibility, and
disabled pytest plugin autoload. They imported only the standard-library
policy and inspected neural scripts as source/AST; Torch and model code were
not imported or executed.

```text
py_compile policy + preflight + preflight verifier + executor + trainer + verifier + author test
PASS

pytest -q \
  lewm/tests/test_go2_shared_jepa_v5_full_training_v2_implementation.py \
  lewm/tests/test_go2_shared_jepa_v5_full_training_execution_amendment_v1_independent_review.py \
  lewm/tests/test_go2_shared_jepa_v5_full_training_execution_amendment_v2_independent_review.py
20 passed in 0.16s
```

The candidate files are ASCII and contain no trailing whitespace. Production
sources expose no alternate backend, module, callback, fixture, test-only,
autocast, GPU1, or iGPU selection path. No `.generated` path, dataset, RGB,
label, checkpoint, Torch runtime, accelerator, G2/G3 role, held-out input,
runtime result, hardware input, or production artifact was opened or changed.

## Execution blockers

1. A different agent must review these exact source bytes and publish the
   canonical implementation review at
   `docs/lewm_go2_shared_jepa_v5_full_training_v2_implementation_independent_review_2026-07-13.json`.
2. The policy currently names the V1 raw-supervision builder, auditor, root,
   and audit schema. Builder V7 approval and Auditor V7 review do not silently
   change those frozen bindings. Accepted V7 integration requires an explicit
   additive full-training policy/source successor and different-agent review.
3. The payload-free preflight must later run once on the exact R9700, its
   immutable receipt must receive a separate different-agent review, and both
   identities must be bound before exact reservation.
4. The exact raw manifest/audit, V4 two-seed ladder, primary V4 checkpoint,
   implementation sources, preflight completion/receipt/review, and
   implementation-review file hashes are all still unset.
5. The V1 wording asks `qualified_checkpoint.pt` to use checkpoint schema V5
   before G2, while the reviewed canonical V5 lifecycle schema admits only a
   post-G2 `g3_candidate` or a fully promoted checkpoint. This candidate
   follows the development-only wording but cannot establish canonical
   lifecycle compatibility by assertion. Independent review must treat this
   as a structural question; resolving it requires an additive lifecycle or
   full-training amendment, not an inferred field set.
6. V2 deliberately retains the frozen four-equal V4 objective, including
   `ordered_first_hit_nll`. If Camera V9 establishes hierarchical loss as the
   necessary successor, this V2 candidate must not execute. A dated additive
   full-training successor must bind the accepted Camera V9 science and pass
   independent review first.

## Independent review request

The reviewer must differ from `/root/coordinator_v2_qa`. Review must rehash all
candidate and parent bytes, keep every future manifest binding null, and use
only synthetic temporary fixtures with accelerator visibility disabled unless
a separately authorized payload-free preflight review explicitly permits the
R9700 smoke. It must challenge reservation ordering, source/import closure,
wrong-role/path substitution, schedule and loss arithmetic, CPU migration,
two-arm equivalence, raw metric accumulation, checkpoint/optimizer/RNG state,
calibration and family gates, diagnostic non-authority, completion inventory,
failure terminality, and qualified-checkpoint lifecycle compatibility.

A PASS may approve only the implementation source closure and the separately
defined payload-free preflight boundary. It cannot fill the exact manifest,
open data or checkpoints, run training, contact G2/G3 or held-out roles, or
grant runtime, navigation, hardware, production, or promotion authority.
