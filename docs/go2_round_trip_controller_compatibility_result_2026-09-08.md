# Variable mission controller recorded compatibility result

The new `ObservedRoundTripController` reproduced all **311 recorded decisions**
from the completed causal-residual native pair: 254 JEPA observations and 57
direct observations. Every original decision field except the explicit controller
identity matched exactly. Commands, terminal policy, sensor/mapping receipts,
model forecasts, scoring and constraint evidence, and causal residual state were
all compared. There were no intervened commands or inferred unexecuted outcomes.

The replay used the original `[1.2, 0]` goal, 240-tick navigation budget and return
disabled. Both original failed native outcomes remain failed. This establishes
compatibility of the variable mission integration on those recordings; it is
not native round-trip execution, independent maze evaluation, or evidence that
persistent memory improves navigation.

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_round_trip_controller_compatibility_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `launch.json` | `aed68d3d365b763c94fb3bc45898053b36e4034997af3e48a7abdae5e4064efa` |
| `seed_2026091001_full_jepa_decisions.json` | `28ddfe9bc4b35ad27e00161597cb70dfb146cefaff7d0dbb03666744cbb7bef4` |
| `seed_2026091001_full_direct_decisions.json` | `a48e84fb66f181f2311a614bed5a2867179269e8c63609a7f57e0bb1ab275510` |
| `result.json` | `5c9d1d60e789329ccb7d2a035388e9fa315f7d702aa7d7dfa438c3b49164792f` |

The completed replay bound 1,382 sources and took 188.241 seconds after launch
with one CPU worker and one fresh model per case. Corrected model state hashes
remained `4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`
(JEPA) and `20c77ef052f7a5168ad60860f4d8d5c40de6a68d9b7f0a2367b8004192d046e5`
(direct); all parameter gradients remained absent. Output bindings were rehashed
after completion. Hardware after replay reported 81.809 GB available RAM and
58.136 GB free artifact space. Elapsed time is replay throughput, not an online
latency measurement.

The source and protocols bound by this attempt are frozen locally. Future
behavior changes require a new source/attempt identity. The next native work
must integrate the four prospective maze scenes, longer bounded acquisition,
both native arrival windows and physical return-edge verification, with an
explicit storage and hardware plan.
