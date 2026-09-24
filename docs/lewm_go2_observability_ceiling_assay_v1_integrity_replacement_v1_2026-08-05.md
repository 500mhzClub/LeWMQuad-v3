# Integrity replacement V1 for the observability-ceiling assay

Date: 2026-08-05
Replaces attempt: `go2_observability_ceiling_assay_v1_attempt_v1`
Replacement attempt: `go2_observability_ceiling_assay_v1_attempt_v2`
Governing preregistration:
`docs/lewm_go2_observability_ceiling_assay_v1_preregistration_2026-08-05.md`
plus `docs/lewm_go2_observability_ceiling_assay_v1_amendment_1_2026-08-05.md`

Status: **science-identical integrity replacement.** No hypothesis, arm, gate,
threshold, seed, split, scorer, decision rule, or diagnostic changes. Only the
runtime environment and the attempt label change.

---

## 1. The infrastructure failure

`attempt_v1` reached the V-JEPA 2.1 encoder construction and stopped with

```
RuntimeError: Missing dependencies: einops
```

The registered terminal is

```json
{
  "attempt_id": "go2_observability_ceiling_assay_v1_attempt_v1",
  "status": "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION",
  "error": "RuntimeError: Missing dependencies: einops",
  "citable_as_scientific_evidence": false
}
```

The V-JEPA hub entrypoint `vjepa2_1_vit_base_384` declares `einops` in its
`hubconf.py` dependency list, and `torch.hub` refuses to construct the model
when a declared dependency is absent from the interpreter. The GPU environment
`~/TinyQuadJEPA` did not have it.

The attempt **failed closed with no scientific decision**, wrote no result, and
emitted no Outcome. It is not retrofittable into a pass, a stop, or any
interpretation.

## 2. What had already happened when it stopped

The failure occurred *after* DINOv2 encoding of both roles and *before* any
model fit, any score, and any Outcome evaluation.

This is material and is recorded rather than glossed: the **evaluation successor
RGB was already opened** by `attempt_v1`. The one-way custody cost declared in
preregistration §4 has therefore been paid. The replacement does not pay it
twice and does not restore it. The V3 panel is spent for privileged-successor
purposes from `attempt_v1` onward, exactly as declared.

No collection byte was mutated. No checkpoint existed to mutate. No untouched,
sealed, held-out, or V4 material was touched.

## 3. The single change

`einops` version `0.8.2`, a pure-Python tensor-rearrangement library, was
installed into `~/TinyQuadJEPA`. It is a declared dependency of the frozen
V-JEPA hub entrypoint and is used only by that encoder's own source, which is
itself frozen and unmodified.

Nothing else changed:

- the same immutable collection at SHA-256
  `711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`;
- the same frozen DINOv2 and V-JEPA 2.1 weights at the same paths;
- the same arms, capacity ladder, split seed, model seeds, bootstrap seed and
  resample count, objective, scorer, complete-tie convention;
- the same `0.13` absolute gate, `0.05` validity threshold, and Outcome
  I/IV/III/II ordering;
- the same amendment-1 validity controls 2a and 2b;
- the same integrity gates and access-ledger expectations.

## 4. Why a new attempt label rather than a rerun

`attempt_v1` holds an immutable terminal recording the infrastructure failure.
The runner refuses to overwrite, resume, or repair an existing attempt. The
replacement therefore executes under the fresh label `attempt_v2`, and
`attempt_v1` is retained permanently as the audit record of the failure.

This mirrors the established precedent in this repository, where infrastructure
failures were replaced by separately registered, science-identical replacements
rather than by silent reruns.

## 5. What this does not authorize

No threshold relaxation, no additional attempt beyond `attempt_v2`, no
tuning, no retry of a *scientific* failure, no promotion, no planner
integration, and no access to untouched, sealed, held-out, or V4 material. If
`attempt_v2` fails scientifically, that is the registered result and no further
replacement follows.
