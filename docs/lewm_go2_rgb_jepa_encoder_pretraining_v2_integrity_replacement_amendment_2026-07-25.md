# RGB JEPA encoder pretraining V2 integrity-replacement amendment

Date: 2026-07-25

## Decision

Authorize source preparation and independent review for exactly one fresh,
science-identical V2 integrity replacement of RGB JEPA Encoder Pretraining V1.
Execution still requires a new machine-checkable source review and a new
one-attempt authorization.

V1 did not reach an experiment. Its isolated hardware preflight passed, then
the runner stopped during deferred runtime assembly when it attempted to
assign `loss_adapter` on a frozen `Runtime` dataclass:

`FrozenInstanceError: cannot assign to field 'loss_adapter'`.

The sealed V1 evidence is:

- reservation: file SHA-256
  `dd2b738dc2756b871627f1fc72097a5b052ef16d91536066bac0979ad374aced`,
  content SHA-256
  `64b18711bba105a2eaa0ba89462cc5604d4b79542397677ef80a2a845d40327c`,
  13,284 bytes;
- failure: file SHA-256
  `f351856d2fc1de9d458f9dedd65db16b1eaf66e387464083a97e1f3fe7cae008`,
  content SHA-256
  `232ae4cc07916e2d4f7d22956110fc25a2c47270003536087ec2b60f7571688b`,
  1,297 bytes;
- completion: file SHA-256
  `777cd2ba1b856a131ba55e61d6c5b04e3305b9c1ef36219dfe1e67652b71908c`,
  content SHA-256
  `fad54ad410ebe961e0c987e2452f06ae16b4c25d3fe727e490eb1bf649bb451f`,
  984 bytes.

Those receipts report `gpu_active_started=false`,
`n320_checkpoint_loaded=false`, zero updates, zero presentations, no Phase B,
and terminal read-only sealing. No dataset payload, RGB, Camera supervision,
tensor checkpoint, training batch, metric selection, or scientific gate was
opened or evaluated. V1 therefore yields no scientific evidence for or
against the mechanism.

## Sole implementation delta

Replace the invalid in-place assignment:

`runtime.loss_adapter = ...`

with construction of a new instance of the same frozen `Runtime` dataclass,
using `dataclasses.replace(runtime, loss_adapter=...)`.

A focused CPU test must prove that:

1. the input runtime is a frozen dataclass;
2. replacement succeeds without mutation;
3. every unchanged field retains object identity;
4. the replacement contains exactly the registered tail-depth loss adapter.

No fallback mutation, schema adapter, model change, or additional scientific
mechanism is authorized.

## Exact scientific identity

V2 retains the complete scientific contract preregistered in
`docs/lewm_go2_rgb_jepa_encoder_pretraining_v1_preregistration_2026-07-25.md`
at commit `51dd74e3a74faab6e575d66761e31f0372285ead`, file SHA-256
`2008ab643faa19a410283ef0fe9ec57ad824ded4429789f65a0ff78ea73bd744`.

In particular, V2 preserves exactly:

- Raw V13 data and train/checkpoint-selection roles;
- N320 initialization and seed `20260712`;
- the nine requested-action tokens and absence of realized motion or Camera
  supervision from Phase A;
- the current-only online path and frozen next-RGB EMA target path;
- model architecture, initialization, losses, weights, optimizer, learning
  rates, EMA momentum, preprocessing, and float32 execution;
- the frozen schedule seed `20260713`, ordering, batch/accumulation structure,
  and checkpoint observations;
- Phase-A gates and the conditional Phase-B entry rule;
- target-encoder-only transfer, evidence-head-only Phase-B training, frozen
  physical evaluator, tail-depth loss, wrong-RGB mapping, and Phase-B gates;
- 1,000 updates and 16,000 presentations per phase;
- 60-minute Phase-A and 120-minute cumulative GPU-active caps;
- one fresh attempt, no retry, no resume, and no threshold or role change.

The scientific contract payload and schema remain byte-for-byte logically
identical. Only external custody identities and the sole wiring fix differ.

## Fresh custody boundary

V2 must use:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_jepa_encoder_pretraining_probe_v2_integrity_replacement`

That root must be absent before reservation. The sealed V1 root is historical
evidence only and must not be opened by the V2 launcher or runner.

V2 requires new source-manifest, independent-review, and execution-
authorization receipts whose paths identify the V2 integrity replacement.
The launcher must validate those exact new receipts before hardware access.

Any V2 integrity, operational, Phase-A, or Phase-B failure is terminal and
authorizes no retry or resume. A V2 scientific pass still authorizes only a
separately preregistered qualification step, never direct G2, navigation,
held-out, sealed, production, promotion, or deployment access.
