# Chained full-history timing comparison preparation

The fresh chained-controller maze-02 simulation is still collecting. Its
launch SHA-256 is
`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`.
The original worker, PID 2994743 with creation time 1789194027.81, was observed
running with flushed decision timing through tick 999 during this preparation.
No terminal result or failure had been recorded at that observation. This is
progress evidence, not an audited navigation outcome.

`scripts/measured_plane_chained_full_history_timing_development.py` prepares
the comparison interface needed by a future replay of that completed history.
It reuses the existing complete-population timing accounting, physical command
endpoint validation and fixed-plus-final state checkpoints. It uses the
previously prepared chained single-pass decision normalization and checks that
retained state comes from the exact chained baseline or chained single-pass
controller, with the same exact chained visual-motion class. Motion and mission
types and fields remain unnormalized, including image caches and tracking
witnesses. The established eleven map/history/registration type-path
normalizations remain the only retained-state implementation normalization.

Five focused tests passed in 2.38 seconds. Synthetic accounting populations
include all 3,124 rows, the original failure region and the final state;
truncation, a missing interior observation, unequal decisions, unequal model
calls and a missing final state are rejected. Fresh actual controller objects
have matching initial retained state; a new tracking witness changes its
fingerprint. Unchained controller or motion substitutions are rejected.
The 3,124-row fixture is synthetic and is not an executed sensor replay.

| File | SHA-256 |
| --- | --- |
| `scripts/measured_plane_chained_full_history_timing_development.py` | `d63cd198978cc30f1fdaeb2693512bfe30ceedd13b55f0aed94b29700a7c2ccf` |
| `lewm/tests/test_measured_plane_chained_full_history_timing_development.py` | `aacfe640e477cb13f9a26e12519b285e4f8beb75696b1425e2d5dd25244b35ec` |

All 2,627 sources bound to the active native launch were independently
rehashed and remain unchanged. Neither new source nor test is in that live
binding. No profiling or full-history comparison was launched, and the active
controller was not changed. A future runner still needs actual completed-input
admission, full paired replay, raw packet/decision reconstruction, model and
state checks, bounded resource monitoring and output authentication. This
preparation establishes no additional speed improvement, navigation success,
real-time operation or hardware qualification.
