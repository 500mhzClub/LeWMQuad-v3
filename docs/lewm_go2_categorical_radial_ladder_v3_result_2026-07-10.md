# Go2 categorical radial ladder v3 result

Date: 2026-07-10

Status: N=1, N=4, and N=16 passed; N32 construction licensed

This is a train-role-only implementation and capacity result. It does not pass
G2, demonstrate scene-disjoint generalization, select or calibrate a full
checkpoint, or license runtime/navigation use.

## Immutable artifacts

- frozen ladder manifest:
  `.generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json`;
- manifest file SHA-256:
  `967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12`;
- V3 full-ray amendment:
  `docs/lewm_go2_categorical_radial_ladder_v3_full_ray_amendment_2026-07-10.md`;
- amendment file SHA-256:
  `921fc48cf2a41924c720654c2d08fbd09ca6ce3ccc7c94ccb6600096a434fcbf`;
- authoritative V3 result:
  `.generated/go2_categorical_radial_micro_overfit/v3/seed_20260710_ladder_result.json`;
- result file SHA-256:
  `7a5f67bacb2e3df67421bcff13b15d1fa3e00d99f3b2af52c52b0b6ce14617a8`;
- result content SHA-256:
  `517313139077027176c471f829f57148684d3df0def6096ce7702d3bbba46ce1`.

The content hash recomputes exactly. The result source-binds the immutable V2
result and all V1/V2/V3 sources, reproduces the registered V2 initial-state
hash, and proves 97 common state tensors bitwise identical outside the replaced
radial block. It records all 4,096 direct radial input/output reachability pairs,
the exact 2,887,067 parameter count, no range wrap, fixed stage restarts, and
zero checkpoint-selection, calibration, non-train, or G2 access.

## Fixed-terminal results

| Frames | Balanced NLL | UNKNOWN recall | FREE recall | OCCUPIED recall | Wrong-view NLL delta | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.00003393 | 1.00000 | 1.00000 | 1.00000 | n/a | PASS |
| 4 | 0.00008444 | 1.00000 | 1.00000 | 1.00000 | 5.32886 | PASS |
| 16 | 0.00284617 | 0.99644 | 0.99984 | 1.00000 | 4.00051 | PASS |

Every stage consumed its full registered budget and used `1e-5` on its final
update. N=16 first passed the complete gate at step 900 and passed every
subsequent 100-step evaluation through step 2,000. Its FREE recall was 1.0 at
1-2 m and 2-3 m and 0.99973 at 3 m and beyond.

Relative to the frozen V2 N=16 endpoint, full-ray V3 changed:

- balanced NLL: `0.01151191 -> 0.00284617`;
- UNKNOWN recall: `0.98747 -> 0.99644`;
- OCCUPIED recall: `0.97007 -> 1.00000`;
- UNKNOWN/known weighted NLL: `0.01902187 -> 0.00535045`;
- FREE/occupied weighted NLL: `0.00400196 -> 0.00034190`.

The full-ray bundle therefore clears the measured UNKNOWN/known occlusion
boundary failure under the frozen data, schedule, budget, and controls. Because
V3 also adds nonlinear depth, GroupNorm/GELU stages, parameters, and compute,
this supports the registered six-block bundle; it does not identify receptive
field alone as the cause.

## Decision

The V3 ladder licenses construction and execution of the already-registered
N32 fit-panel diagnostic. It does not license a longer ladder run, second seed,
full-dataset training, checkpoint selection, calibration, G2, memory fusion,
or closed-loop evaluation. N32 must use its original faithful and conditional
ceiling optimizer branches, aggregate-plus-five-family gates, fixed terminal
three-evaluation rule, wrong-view controls, and train-only provenance. Only a
favorable seed-20260710 N32 and holdout result may license seed 20260711.
