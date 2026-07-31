# V18 spatial-token delay-line Joint-JEPA V1 — integrity replacement V2

Date: 2026-07-31

Status: preregistered, source-only, execution denied pending implementation review, a certified narrow export, and a fresh one-shot authority.

## Reason for V2

The consumed update-zero gate-timing integrity replacement V1 correctly passed
all twelve registered update-zero checks while retaining the false absolute
noncollapse diagnostics as visible, diagnostic-only facts. It then terminated
at `train_update_1` with
`V21 batch schema is not the exact one-field extension`, before a training
forward, training presentation, autograd call, optimizer step, EMA step,
snapshot, or checkpoint.

The independently reviewed terminal result is
`docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_update_zero_gate_timing_integrity_replacement_v1_terminal_infrastructure_failure_result_2026-07-31.json`
at commit `c2dd29783aab38c4a04e588a70549e5b85c98d67`.

The frozen V25 physical microbatch builder intentionally nests the V24, V23,
and V21 adapters. The V18 training module exports the exact base and V25 batch
schemas, scene-innovation key, and action-prior key, but omitted the three
intermediate compatibility names `REQUIRED_BATCH_KEYS_V21`,
`REQUIRED_BATCH_KEYS_V23`, and `REQUIRED_BATCH_KEYS_V24`. The V21 adapter
therefore observed an empty compatibility tuple and stopped before calling the
inherited physical tensor builder. No V1 checkpoint or recovery state exists.

## Exactly authorized correction

Authorize exactly one fresh, science-identical integrity replacement V2 that:

- adds only these exact training-module aliases:
  - `REQUIRED_BATCH_KEYS_V21 = v25.REQUIRED_BATCH_KEYS_V21`;
  - `REQUIRED_BATCH_KEYS_V23 = v25.REQUIRED_BATCH_KEYS_V23`;
  - `REQUIRED_BATCH_KEYS_V24 = v25.REQUIRED_BATCH_KEYS_V24`;
- changes only the replacement schema, document, certified-source, authority,
  output-root, and experiment-arm identities needed for a fresh V2 lifecycle;
- adds a CPU-only synthetic regression that reaches the real nested
  V25→V24→V23→V21 schema adapter and proves exact V25 key order without opening
  data or constructing a scientific training result;
- preserves the V1 failure receipt and result unchanged;
- starts from a fresh initialization under a new output root, without using a
  V1 checkpoint, optimizer state, metric artifact, recovery state, or random
  state.

Do not bypass the V25 builder with the pristine V13 builder. The full V25
physical batch is the registered scientific input and is consumed by the V18
training route. The three aliases are compatibility metadata only; they must
not add, remove, reorder, or mutate a batch field.

## Science that must remain identical

V2 must preserve exactly the V1:

- model class, parameterization, encoder, representation, predictor, and
  initialization;
- K4 spatial-token FIFO, local causal depthwise Conv3D reader, action FiLM,
  shared recursive H4 prediction, and EMA target;
- data identities, train/selection splits, RGB loaders, seed `20260731`, and
  schedule order;
- optimizer, learning rates, parameter groups, gradient routing/scaling, EMA
  coefficient, and update order;
- physical objective and full plus 0.5-weight masked-current memory JEPA
  losses;
- 16 memory and 8 physical presentations per update, one optimizer step, and
  one EMA step;
- wrong-action, reset, reverse, shuffle, persistence, and HOLD controls;
- update-zero finite/nonzero and persistence-identity gate, with absolute rank
  diagnostics visible but not terminal at updates 0 and 100;
- first absolute noncollapse enforcement at update 250 and the unchanged
  update-250, update-500, and terminal gates;
- observation updates 0, 100, 250, 500, 750, and 1000; snapshots at 250, 500,
  750, and 1000;
- stage-A cap 500, terminal cap 1000, memory cap 16,000, physical cap 8,000,
  and combined cap 24,000 presentations;
- no probability-calibration, G2, navigation, held-out, or sealed access.

## Terminal rules

- If unchanged absolute noncollapse fails at update 250, close this exact
  delay-line mechanism.
- If V2 encounters another pre-training inherited batch-schema or source-hook
  incompatibility, close this exact mechanism without a V3 integrity
  replacement.
- Passing update 250 does not guarantee continuation; all later frozen gates
  remain conjunctive.

## Current authority

This document grants source implementation and CPU-only synthetic-test
authority only. It grants no dataset/RGB payload, GPU, training, checkpoint,
recovery, navigation, probability-calibration, G2, held-out, sealed,
production, promotion, retry, resume, or execution authority.
