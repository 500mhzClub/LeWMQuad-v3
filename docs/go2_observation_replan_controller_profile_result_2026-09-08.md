# Observation-replan controller profile result

The complete recorded full-direct observation-replan trajectory reproduced
**all 40 decisions exactly**, with unchanged final-model state. This diagnostic
identifies controller costs; it adds no native execution or arrival evidence.

The instrumented controller calls consumed 15.294723 seconds in total, with
median 504.622 ms and maximum 685.141 ms per observation. cProfile overhead and
concurrent short-horizon training are included. These durations are not an
uninstrumented latency benchmark and do not establish a speedup.

| Function | Calls | Cumulative seconds |
| --- | ---: | ---: |
| Measured-floor mapper `observe` | 30 | 10.3466 |
| Joint visual floor map `observe` | 30 | 5.9528 |
| Retained-patch `classify_current` | 30 | 4.3824 |
| Joint visual surface memory `observe` | 30 | 4.1470 |
| Measured-sample bounds `insert` | 90 | 3.6792 |
| `deepcopy`, including recursive calls | 4,765,678 | 3.3104 |
| `observed_floor_cell_index` | 91 | 3.2316 |
| Surface index `insert` | 120 | 3.1568 |
| Visual-led motion `observe` | 30 | 1.9003 |
| Neural candidate `select` | 27 | 0.3112 |

**Cumulative times overlap; the rows must not be summed.** Candidate selection
includes input preparation, model inference and its initial scoring. All torch
module calls consumed 0.2395 cumulative seconds. This evidence points first to
map insertion, repeated floor-index construction and witness copying when
optimizing this controller. Any replacement still needs exact output and
failure-boundary checks, followed by uninstrumented full-loop measurement.
No optimization has been adopted in the short-horizon native probe.

Native acquisition is outside this profile. The completed predecessor readout
recorded direct-case acquisition median 121.571 ms, controller median 359.118 ms,
and complete-iteration median 500.742 ms; all 39 command iterations exceeded
100 ms. Improving mapping alone would not resolve the whole-loop deadline.

The diagnostic used one CPU process and one numerical thread, 1,096 bound
source paths, 74,500,079,616 bytes available RAM and 69,744,381,952 bytes free
artifact storage before launch. It completed in 28.494353 seconds after launch,
including replay and final verification. All original artifact/source bindings,
the assigned snapshot and URDF were reverified. The profile and every outcome
remain immutable.

Artifact root:
`go2_observation_replan_controller_profile_v1_attempt_001`

| Artifact | SHA-256 |
| --- | --- |
| Input probe result | `8cdfd800eda961de5b1a58a3b64fe8fd471c0f8c8165aae9c61f699c500010eb` |
| Launch | `4bff42da56cadcf0b56b41cd3206480ed11ad7d99094d0cf6cde277b7f6fb78c` |
| Profile | `d0032b9d318b50b533fa6feddfa1d4e82ba33fd6029bff058b156b619cbed352` |
| Result | `faac6f7e6be768cc566360e5591e3ca2c4dfa9cdddce8a4939129be6f879390f` |

This remains a reused development trajectory with zero independent maze
replication, zero new arrivals and no real-time or hardware qualification.
