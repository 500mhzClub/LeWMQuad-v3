# Swept-progress selection census V1 result

Date: 2026-07-28

## Verdict

`PASS_SELECTION_SCREEN`

The action-conditioned swept-progress target is informative on 399 of 495 exact
checkpoint-selection states, above the frozen minimum of 128, and every registered
family exceeds its minimum of 8. The target therefore earns one reviewed joint-
JEPA implementation and complete model-free preflight. This result does not itself
authorize RGB, GPU, training, promotion, navigation, G2, held-out, or sealed use.

## Exact result

- source commit: `31c6162b87152426c7f2f38eb13d36381908a6be`
- authorization commit: `f84c897`
- command: `/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python -I -B scripts/diagnose_go2_swept_progress_selection_v1.py`
- exit: `0`
- population: 495 states, 3,960 non-HOLD action rows, 8 scenes
- report content SHA-256:
  `0cd1972913a1c3edde3ac806f7dd6fb78082400b6556ab0f30ca5940303b25c8`

| Family | Informative | States | Required |
|---|---:|---:|---:|
| `large_enclosed_maze` | 64 | 64 | 8 |
| `local_composite_motifs` | 51 | 64 | 8 |
| `loop_alias_stress` | 61 | 64 | 8 |
| `medium_enclosed_maze` | 64 | 64 | 8 |
| `open_obstacle_field` | 26 | 64 | 8 |
| `rough_local_dynamics` | 22 | 64 | 8 |
| `small_enclosed_maze` | 47 | 47 | 8 |
| `visual_sensor_stress` | 64 | 64 | 8 |
| **Total** | **399** | **495** | **128** |

Of the 96 non-informative states, 20 had no positive safe-progress prefix and 76
had positive progress but identical action prefixes. Every non-HOLD action
participated in all 399 unequal-prefix states. Positive-prefix counts by action
were: arc left 455, arc right 452, backward 441, forward fast 431, forward medium
443, forward slow 452, yaw left 473, and yaw right 474.

The small-maze failure is resolved by the target rather than by a threshold
change: all 47 small-maze states contain positive, differing action prefixes.
Their safe progress is naturally short (primarily 0.1--0.5 m), which the rejected
1.45 m all-or-zero bridge could not represent.

## Access receipt

The pass opened the exact execution binding, raw manifest, pairs, endpoints,
audit, geometry contract, directional policy, and primitive registry once each,
plus exactly 8 render summaries, 8 source-frame JSONLs, and 8 scene manifests.
Schedule, RGB, labels/V4 outputs, source-authority documents, models, checkpoints,
runtime outputs, navigation, G2, held-out, sealed, and production open counts were
all zero. No GPU or training was used and no filesystem output was written.

## Next boundary

Implement exactly the direct N320 action-predicted-latent conditional-survival
mechanism frozen in the target decision. Do not introduce global mean pooling,
teacher-state reconstruction, a separately trained head/predictor, or a different
bin/horizon target. Before GPU use, materialize and validate the same target on
train and calibration and the frozen 16,000-presentation schedule against the
already frozen coverage floors.
