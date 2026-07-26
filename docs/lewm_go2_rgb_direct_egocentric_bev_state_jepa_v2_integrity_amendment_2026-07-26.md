# Direct Egocentric BEV-State JEPA V2 integrity amendment

Date: 2026-07-26

Status: source implementation and source-only synthetic review are permitted;
execution is not authorized by this amendment.

## Why V2 exists

The single authorized V1 attempt ended in a valid operational-integrity
failure before update-zero observation. The frozen terminal audit is commit
`ae94021d44711bf9ba5fbb1386b4f8caf2617dac`, file SHA-256
`f928c11a2e52349145701b25a21f8b1b987ee80a365aaa2c3858d3cf650220c4`,
content SHA-256
`2974d914f9cde1ae93c34d76d07b1740d8c5ac17beb3b5f4922500bd242df956`,
and 9,291 bytes. It records zero observations, RGB/raster requests,
objectives, backward calls, optimizer updates, EMA updates, presentations,
and scientific work. V1 is consumed and permanently closed; V2 is a distinct
integrity replacement, not a V1 retry.

V1 fresh-module construction saved and restored the CPU RNG around
`torch.random.manual_seed(20260712)`. That call also scheduled accelerator RNG
seeding, so the runner correctly rejected the changed CUDA RNG state.

## Sole implementation delta

V2 replaces only that seed call with
`torch.random.default_generator.manual_seed(20260712)` while retaining the
same CPU RNG save/restore boundary. This seeds only the CPU default generator
used for fresh decoder, state-head, and predictor parameter draws. It neither
reads nor mutates a device RNG stream. The initialized V2 state dictionary
must be bitwise identical to V1 for the same bound N320 encoder state, and the
caller's CPU and every device RNG state must be exactly unchanged.

No other scientific or operational change is permitted. In particular V2
preserves exactly:

- the direct RGB-to-three-logit BEV architecture and sole causal bottleneck;
- N320 encoder-only migration, seed `20260712`, parameter draw order,
  initialization bytes, target hard sync, and EMA momentum `0.996`;
- all train and checkpoint-selection rows, endpoint populations, mappings,
  labels, roles, and input hashes;
- `G + J + C`, all loss weights and reductions, and the six-call isolation;
- AdamW groups, learning rates, clipping, precision, and weight decay;
- schedule seed `20260713`, order, prefix hashes, batch 16, microbatch 4;
- observations at 0/100/400/1000 and every comparator and threshold;
- the 1,000-update, 16,000-presentation, 60-minute, one-attempt hard caps;
- all custody, write-only, failure-receipt, no-retry, and downstream denials.

The frozen V1 science-contract SHA-256 is
`9cd985135ad9fef2ce324d4389b01d95f800d186724729626cbe895e1d8bdfb9`.
The frozen V1 model, objective, optimizer, schedule, and gate-threshold hashes
are respectively
`90a2d725ce3694235d5aca2ecdda8b9e0d38df0f11e29a4d571dd0e7c2d76c5b`,
`a3e48bb32f35d5d66c21572c9aa5fe5b5673833be768759a7875f5c07630e19a`,
`af379468031d4dc7c7bf26ec9e0a0d30ca29fc16a9b34c44e688994e41372715`,
`f156cc0274590be295bac0607790b61e2ed6aed9528a236bfb157cd5dd4beba2`,
and `18a62d22d0bd2b1b7b93e469d6a9d4954d517b7fcbe2961c64d7a675bc53f1b0`.

## Lifecycle and authority

V2 uses the distinct output root
`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_direct_egocentric_bev_state_jepa_probe_v2_integrity`.
The root must be absent before reservation and is consumed by any reservation
or partial attempt. V1 output is terminal evidence and is never a V2 input.

Before any generated input, checkpoint tensor, Torch runtime, or accelerator
access, V2 still requires a frozen recursive source manifest, independent
source-and-science review, and a separate authorization for exactly one fresh
attempt. A pass authorizes only later perception-gate requalification work.
It does not authorize G2, navigation, held-out or sealed evaluation,
production, promotion, deployment, retry, resume, repair, a second seed, or a
parameter/loss/schedule/timing successor.

No generated input, RGB, label, checkpoint, tensor, runtime output, trace,
accelerator, navigation, held-out, or sealed material was opened to prepare
this amendment.
