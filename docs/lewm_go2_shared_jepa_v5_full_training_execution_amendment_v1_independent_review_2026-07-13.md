# Shared JEPA V5 full-training amendment V1 independent review

Date: 2026-07-13

Reviewer: `/root`

Verdict: **BLOCK**

## Candidate

- amendment SHA-256:
  `b21d01d062543cc7b7f3f5281f66ac40df76726c678a9364f7a4e451b035a4a7`;
- author-handoff SHA-256:
  `fa0a497fad2f17a5d0919e1160b6040cbe13740315cfc180418d99dbf494d6bc`;
- independent-test SHA-256:
  `b2959ea11cff80091a9f94c61dde14750726332001326c0fa30bd186418c6b38`;
- independent result: `4 passed in 0.01s`.

The reviewed V5 model/output/loss and staged-lifecycle source hashes all
reproduce exactly. No generated dataset, role payload, RGB, label, model,
checkpoint, G2, held-out, runtime, hardware, navigation result, or accelerator
was opened.

## Blocking findings

1. **The GPU preflight has two incompatible lifecycle positions.** Lines
   255-259 require the exact-model GPU smoke before the exact reservation,
   while lines 437-440 require the canonical attempt reservation before any GPU
   or model access. An executor cannot satisfy both rules. V2 must either give
   the payload-free smoke its own immutable preflight receipt and explicitly
   exempt only that operation, or reserve and consume the exact attempt before
   the smoke.
2. **A mutable status document is treated as an immutable authority parent.**
   The amendment binds readiness-goal SHA-256 `1095252d...8a12`, while the
   active status document is now `45e82832...adf5` after legitimate progress
   updates. Rebinding it after each update would repeat the defect. V2 must bind
   a separately frozen design snapshot or make the live status record explicitly
   informational and non-authoritative.
3. **The stated causal generalization claim is selection-biased and its final
   precision comparison is incomplete.** The promoted update is chosen on the
   495 checkpoint-selection pairs, the ablation is compared on those same
   pairs, and that comparison is then allowed to support a causal development-
   generalization claim. The phrase `does not reduce ... after the fixed
   calibration below` also has no exact comparator. V2 must label this panel as
   a matched development diagnostic only, or move any causal generalization
   claim to a separately preregistered untouched evaluation and specify the
   exact precision inequality.

## Retained decisions

The role split, primary V4 seed, identical arm initialization, fixed train
schedule, optimizer, joint JEPA plus four-equal current/next V4 loss, promoted-
only selection, fixed calibration grid, per-family gates, immutable output and
ledger requirements, GPU0-only rule, and one-shot G2 boundary are otherwise
coherent and retained.

## Authority

This review grants no trainer implementation, dataset construction/use, V4
execution, training, calibration, checkpoint use, G2/G3, held-out, runtime,
hardware, navigation, production, or promotion authority. Additive V2 must be
reviewed by an agent other than its author.

