# Post-action projective-support selection admissibility census V1 result

Date: 2026-07-28

## Verdict

`STOP_SELECTION_SCREEN`

The corrected end-to-end admissible-prefix definition increased informative
checkpoint-selection states from 58 to 165 of 495, but the preregistered
per-family floor failed: `small_enclosed_maze` had 0 informative states against a
minimum of 8. No RGB, model, GPU, training, checkpoint, navigation, G2, held-out,
or sealed access is authorized or justified by this result.

## Execution

- corrected source commit: `7e79346253c707cba440cf1557649d3d6adca844`
- integrity-replacement authorization commit: `87fd25b`
- command: `/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python -I -B scripts/diagnose_go2_post_action_projective_support_selection_admissibility_v1.py`
- exit: `0`
- report schema:
  `lewm_go2_post_action_projective_support_selection_admissibility_census_v1`
- report content SHA-256:
  `66100602f72e8f8a9707ebc3b8d41ffc53a2e616dc06427c1bd745d933d96317`
- population: 495 states, 4,455 action rows, 8 scenes, 8 registered families

The original invocation stopped before data access because it required canonical
JSON without the exact binding file's terminal newline. The one integrity
replacement changed only that redundant representation check; the exact file
SHA-256 and content SHA-256 checks remained. The failed invocation opened only the
V4 execution binding and wrote nothing.

## Census

| Family | Original informative | Proposed informative | Required |
|---|---:|---:|---:|
| `large_enclosed_maze` | 0 | 22 | 8 |
| `local_composite_motifs` | 7 | 21 | 8 |
| `loop_alias_stress` | 3 | 19 | 8 |
| `medium_enclosed_maze` | 1 | 24 | 8 |
| `open_obstacle_field` | 21 | 35 | 8 |
| `rough_local_dynamics` | 25 | 32 | 8 |
| `small_enclosed_maze` | 0 | 0 | 8 |
| `visual_sensor_stress` | 1 | 12 | 8 |
| **Total** | **58** | **165** | **128** |

Of 495 states, 187 had a positive best admissible prefix; 165 also had action
variation and were informative. The transition was 58 original-and-proposed, 107
proposed-only, 0 original-only, and 330 neither. Rejections comprised 308 states
with no positive admissible prefix and 22 with a positive prefix but no action
difference.

The dominant failure is the fixed blind bridge, not immediate action feasibility.
Across all families, immediate primitives were feasible for 441--475 states per
action, while the 1.45 m blind bridge was feasible for only 108--136. In
`small_enclosed_maze`, every action had 0 feasible blind bridges across all 47
states. Only 7 small-maze states had any positive uncomposed remote prefix, so
removing the bridge alone would still miss the frozen per-family floor and would
not justify threshold shaving.

## Access receipt

The successful pass opened the exact execution binding, raw manifest, pairs,
endpoints, audit, geometry contract, directional policy, and primitive registry
once each, plus exactly 8 render summaries, 8 source-frame JSONLs, and 8 scene
manifests. Schedule, RGB, labels/V4 outputs, source-authority documents, models,
checkpoints, runtime outputs, navigation, G2, held-out, sealed, and production
open counts were all zero. It used no GPU or training and wrote no filesystem
output.

## Scientific consequence

Do not train the frozen fixed-horizon projective-support corridor V1 and do not
create another science-identical label adapter. The admissible composition fixed
the all-actions mask error, but the 1.45 m fixed bridge/remote horizon is
structurally incompatible with the registered small-maze family. The next probe
must use a materially different, scale-appropriate navigation target rather than
relaxing the observed family threshold after the result.
