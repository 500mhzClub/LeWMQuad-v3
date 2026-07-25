# RGB JEPA encoder pretraining V3 integrity-replacement amendment

Date: 2026-07-25

## Decision

Authorize source preparation and independent review for exactly one fresh,
science-identical V3 integrity replacement of RGB JEPA Encoder Pretraining V2.
Execution still requires a new machine-checkable source review and a new
one-attempt authorization.

V2 passed its isolated hardware preflight, activated the GPU, loaded the
authorized N320 initialization, and stopped while hashing the newly
constructed Phase-A model before update zero:

`RuntimeError: self.dim() cannot be 0 to view Long as Byte (different element sizes)`.

The canonical state-dictionary hashing helper attempted
`tensor.view(torch.uint8)` on the scalar `Long` buffer
`appearance_projector.net.1.num_batches_tracked`. V2 therefore completed zero
updates and zero presentations, did not enter Phase B, emitted no scientific
result, and qualified no checkpoint.

The independently written terminal audit is
`docs/lewm_go2_rgb_jepa_encoder_pretraining_v2_integrity_replacement_terminal_audit_2026-07-25.json`
at commit `dd56463`, file SHA-256
`62df9c5e433dad4d3c052fffd96cbc51f19bf847455f4021c79e0326f209c438`,
content SHA-256
`43a629ef3144fed88bb1326abc973a6ba638007f559e78b8fb7bba1f3716d38a`,
5,649 bytes. V2 is sealed historical evidence and yields no scientific
evidence for or against the encoder-level JEPA mechanism.

## Sole implementation delta

In the canonical `tensor_state_dict_sha256` helper, replace the byte view:

`tensor.view(torch.uint8)`

with:

`tensor.reshape(-1).view(torch.uint8)`

The original tensor shape remains bound in the hash header. Flattening before
the byte reinterpretation makes zero-dimensional tensors valid and leaves the
byte sequence—and therefore the hash—unchanged for every previously supported
non-scalar contiguous tensor.

A focused regression must prove deterministic scalar-`Long` hashing and exact
equality with the old algorithm for non-scalar tensors. Before execution, a
GPU-hidden synthetic smoke must also construct and hash the exact Phase-A
model, serialize and rehash its snapshot, transfer only its online raw encoder
into Phase B, and verify the frozen Phase-B boundary. No generated input,
dataset payload, checkpoint, prior attempt root, or protected material may be
opened by that smoke.

No other implementation, fallback, model, data, training, evaluation, or
scientific change is authorized by this amendment.

## Exact scientific identity

V3 retains the complete scientific contract preregistered in
`docs/lewm_go2_rgb_jepa_encoder_pretraining_v1_preregistration_2026-07-25.md`
at commit `51dd74e3a74faab6e575d66761e31f0372285ead`, file SHA-256
`2008ab643faa19a410283ef0fe9ec57ad824ded4429789f65a0ff78ea73bd744`.
Its canonical science-contract SHA-256 must remain exactly
`760a922e8123934d54dd26292846af2e516e64d239cf92da315c44f6141a710a`.

In particular, V3 preserves exactly:

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
identical. Only external custody identities and the sole scalar-safe
serialization fix differ.

## Fresh custody boundary

V3 must use:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_jepa_encoder_pretraining_probe_v3_integrity_replacement`

That root must be absent before reservation. The sealed V1 and V2 roots are
historical evidence only and must not be opened by the V3 launcher or runner.

V3 requires new source-manifest, independent-review, and execution-
authorization receipts whose paths identify the V3 integrity replacement.
The launcher must validate those exact new receipts before hardware access.

Any V3 integrity, operational, Phase-A, or Phase-B failure is terminal and
authorizes no retry or resume. A V3 scientific pass still authorizes only a
separately preregistered qualification step, never direct G2, navigation,
held-out, sealed, production, promotion, or deployment access.
