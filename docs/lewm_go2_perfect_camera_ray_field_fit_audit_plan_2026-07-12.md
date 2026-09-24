# Perfect camera-ray field fit audit

Date: 2026-07-12  
Status: implementation and synthetic dry run complete; exact 320-frame fit run
not yet executed

## Question

Before training another visual head, determine whether exact camera-ray
information can mechanically reproduce the current 64x64
UNKNOWN/FREE/OCCUPIED physical-v3 label on every registered train-fit frame.
This is an information-contract test, not a learned-model test.

## Exact meaning of a perfect field

The existing target is not generated from a conventional depth image alone.
The audit therefore keeps the two ray populations used by the target:

1. Five exact ground-support queries for every 0.05 m physical source cell:
   the center and four corners. A source cell is visibly FREE only when all
   five queries lie in the calibrated rectilinear frustum and no rendered box
   is hit before the ground point.
2. The registered stride-2 pinhole pixel lattice. Its exact nearest rendered
   box hits provide OCCUPIED surface witnesses.

The independent rasterizer aggregates those fields into rotated 0.10 m output
squares. It reports two arms:

- `ray_only`: prescribed ray evidence plus the collision FREE-to-UNKNOWN veto;
- `contract_assisted`: the same ray evidence plus the source zero-inflation
  physical-free mask used by the current target.

The split is mandatory. If only the assisted arm is exact, a privileged
physical prior is doing work that should not be attributed to the camera.

This audit does **not** claim that an ordinary finite-resolution pixel depth
map is sufficient. The five ground-support queries are generally off the
registered pixel lattice.

## Implementation boundary

- Independent NumPy geometry and rasterization:
  `lewm/benchmarks/go2_perfect_camera_ray_field_audit.py`
- Fit-only runner:
  `scripts/audit_go2_perfect_camera_ray_field_fit.py`
- Focused parity, determinism, semantic-arm, fail-closed, and CLI tests:
  `lewm/tests/test_go2_perfect_camera_ray_field_audit.py`

The rasterizer does not call the production physical-v3 label builder. The
exact runner reuses only the frozen N32 audit's allowlisted input reader to
obtain the already-bound 320 train labels and source geometry. It requires
exactly 320 unique fully supervised 64x64 frames and fails closed otherwise.

Neither mode reads RGB, model/checkpoint output, checkpoint-selection,
calibration, G2, held-out, runtime, or sealed payload. No GPU is used. Every
promotion license in the result remains false.

## Verified dry run

Command:

```bash
PYTHONPATH=.:lewm_worlds python3 \
  scripts/audit_go2_perfect_camera_ray_field_fit.py --dry-run
```

The synthetic train-scene dry run is deterministic and bit-exact against the
production label contract. The focused suite passes:

```bash
PYTHONPATH=.:lewm_worlds python3 -m pytest -q \
  lewm/tests/test_go2_perfect_camera_ray_field_audit.py
```

Current result: `9 passed`.

## Exact fit execution

After independent source review, run once with the already reviewed frozen
fit-input machine-manifest file hash:

```bash
PYTHONPATH=.:lewm_worlds python3 \
  scripts/audit_go2_perfect_camera_ray_field_fit.py \
  --run-exact-fit \
  --machine-manifest-sha256 <reviewed-file-sha256>
```

The immutable result path is:

```text
.generated/go2_perfect_camera_ray_field_fit_audit/v1/result.json
```

## Decision table

| Contract-assisted | Ray-only | Meaning | Next action |
| --- | --- | --- | --- |
| fail | any | Independent mechanics do not reproduce the target | Fix the audit or expose a target-contract inconsistency; do not train |
| pass | fail | Exact labels require the privileged physical-free prior | Redefine a deployment-observable target or explicitly prove how RGB supplies the missing fact |
| pass | pass | Prescribed perfect rays mechanically contain sufficient information | Build a learned camera-ray/first-surface head, then rerun the unchanged N32 fit ladder |

Even a two-arm pass licenses only the representation mechanism. It does not
license a model, generalization, G2, memory, exploration, runtime, or claims.

## Subsequent learned step

If both arms pass, the next head should predict a queryable camera first-
surface field from the shared JEPA tokens, not directly memorize a 64x64 map.
Its output must support the exact ground-query directions and the registered
pixel first-hit lattice. A fixed coarse depth vector with interpolation is not
accepted without a separate parity proof.

The learned sequence is then:

1. Micro-overfit the exact 320-frame train panel and pass every unchanged N32
   class, distance, family, and wrong-image gate.
2. Pass both registered seeds and both registered train-role holdouts.
3. Extract the ray head as a parallel consumer of the one shared JEPA encoder.
4. Only after the complete N32 ladder passes, train the shared model and use
   the one-shot G2 protocol.
5. Fuse learned physical evidence over time, derive body configuration only
   after fusion, and then train the frontier-viewpoint exploration head.

