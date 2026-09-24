# Trained observation-horizon native probe result

Both preselected first-seed models failed to reach the goal. They completed
fresh native collection and exact raw sensor/model/command replay, with all
strict physical-visibility checks passing and no hard measurement failures.
The trained 100-ms first forecast did not repair this integration mission.

| Outcome | Full JEPA | Full direct |
| --- | ---: | ---: |
| Verified arrivals | 0 | 0 |
| Minimum actual goal distance, m | 1.1077011731531885 | 1.1050350370169997 |
| Terminal actual goal distance, m | 1.1077011731531885 | 1.1055756786450857 |
| Command ticks / RGB-D frames | 38 / 39 | 38 / 39 |
| Physics samples | 2,650 | 2,650 |
| Right-turn selections | 17 | 16 |
| Right-arc selections | 8 | 9 |
| No-feasible-action selections | 1 | 1 |
| Complete-iteration median, ms | 490.699570 | 494.264361 |
| Complete-iteration maximum, ms | 534.669804 | 533.844989 |

Both stopped at tick 28 with
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`, followed by the
unchanged ten zero-command drain. There was no physical or acquisition stop.
Each made 16 view-acquisition and 10 waypoint selections, including the final
no-action decision. Maximum observed XY pose error was 0.0021068214643736064 m
for both. Final native XY displacements were
[0.09680598273669395, -0.09982409167830529] m for JEPA and
[0.09910756196985142, -0.10165343624087184] m for direct.

At tick 28 all six first-horizon surface checks passed for both models, but all
six nominal predicted chords failed the unchanged 0.45-m observed-cell radius.
Their minimum predicted distances were approximately 0.4353–0.4376 m for JEPA
and 0.4381–0.4388 m for direct, with nearest observed cell [11, -2]. This is a
predicted-clearance rejection, not a measured collision or sensor failure.

The new models shared 28 exact observation frames, with their first requested
command difference at tick 27: JEPA turned and direct continued its arc.
Against each model's predecessor cadence probe, the common prefixes were
18 frames for JEPA (first command difference 17) and 19 for direct (difference
18). Physical, public-policy, fast-gyro, RGB and observer/memory prefixes were
exact through the observation preceding each differing command. No unexecuted
outcome is inferred. The previous JEPA's long hold sequence disappeared, but
its replacement still failed this mission.

A read-only check of the saved forecast banks identifies a planning limitation:
the controller used only its first 100-ms forecast for nominal clearance even
though it had forecasts through 800 ms. The selected JEPA arcs at ticks 23–26
and direct arcs at ticks 22–27 had later predicted segments that violated the
same radius against cells already recorded as nearest witnesses at that tick.
The first JEPA example passed at 100 ms with 0.4959140512 m clearance but crossed
the nominal boundary at 700 ms; its predicted path minimum was 0.4425587471 m.
The first direct example passed at 100 ms with 0.4983481236 m but crossed at
700 ms, with path minimum 0.4425459655 m. Both examples use the already-observed
cell [11, -1].

This limited-witness diagnostic used the exact saved predictions, observed
poses and map transform. Result, decision-file and geometry-function hashes
were checked, and recomputed first-horizon distances matched the recorded
checks exactly. It does not reconstruct every occupied cell, approve another
action, claim a hypothetical collision or show that longer-horizon control
would succeed. It motivates a distinct prospective planner using all eight
predicted segments while still executing only one command per observation.

The two-scene phase took 76.956872 seconds, used 1,141 bound source paths and
378 artifacts, and reached maximum worker RSS 2,246,471,680 bytes. Preflight
measured 82,640,416,768 bytes available RAM and 69,670,760,448 bytes free artifact
storage. The readout verified 1,144 source paths. All model states remained
unchanged; no fitting, checkpoint selection or retry occurred. The mapping
optimization prototype was not included in this source closure or execution.

All 38 complete command iterations per case exceeded 100 ms while physics was
paused during computation. Acquisition medians alone were 118.065356 ms and
118.737331 ms. This is not real-time or hardware-qualified operation.

Artifact roots:
`go2_observation_horizon_goal_probe_v1_attempt_001` and
`go2_observation_horizon_goal_readout_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| Fit result | `45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418` |
| Fit readout result | `b7b9aac1b8c44ee924e8d940e782f016a0693be5149c91bcb5f6ed676ad30a25` |
| Probe launch | `2b9d173a6775109328de6fa09f98b6a04131305c32df7135eb9f5b0aa1ff0dfc` |
| Probe result | `f484a0c3b95cb2f3d0868dca1f1f40d11ba0d29c42e12cc9604061870a0d5608` |
| Readout launch | `e79d53e3be5825d64bacc21f86e56f160551f763bcd350e27dfa3e248f6984a0` |
| Readout result | `8bce2ae56ed150872b28add57a694ddc685ee6ff92ba77a16aa8aaf54c9a3cb7` |

There remain zero verified native arrivals and zero independent novel maze
missions. Predictive benefit, physical backtracking, matched reactive/nonpredictive,
planning and memory baselines, independent layouts, realistic sensing/timing
and bounded real-platform evidence remain outstanding.
