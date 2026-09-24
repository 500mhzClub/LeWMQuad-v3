# Verified simulated round trip on reused maze02

The fresh stop-conditioned trial completed with **`verified_round_trip: true`**.
The robot reached the outbound goal, physically retraced the loop-erased route
home, and passed both one-second arrival windows and the separate final stop.
All 4,750 complete controller decisions replayed exactly. Raw sensor
reconstruction, command auditing, strict physical visibility and unchanged model
weights passed; there were no hard measurement failures.

| Physical arrival window | Frame | Maximum goal distance, m | Maximum speed, m/s |
| --- | ---: | ---: | ---: |
| Outbound | 3062 | 0.0437775564 | 0.0441619732 |
| Return | 4739 | 0.0219954773 | 0.0322539125 |

Both windows satisfy the unchanged 0.06 m distance and 0.05 m/s speed limits.
There were 15 outbound and 11 return edge crossings, with no invalid crossings,
and zero recorded contact flags across 238,200 physics samples. The final ten
zero-command ticks and terminal quiet check passed. The previous failed run
remains preserved: its return window peaked at 0.0608677 m/s. The new mission
waits for a prior quiet interval under a zero request before counting dwell.

This is one **reused development layout**, using the corrected
`seed_2026091001_no_rgb_direct` action-conditioned predictor and RGB-D visual
perception. The no-RGB ablation applies to the learned predictor; the controller
still uses the cameras and depth. This outcome does not establish JEPA's
contribution or reliable navigation across independent mazes.

Collection took 4,207.51 seconds, and the complete run took 8,507.30 seconds
(141.79 minutes). Acquisition plus control had median 787.800 ms and p95
1,192.186 ms. All 4,750 observations exceeded the 100 ms command interval;
physics paused during computation. Sensing retains its ideal simulation
assumptions. Continuous real-time execution and hardware qualification are false.

The native runner ended successfully in session 97626, with no failure artifact.
Both collection and audit resource checks completed without a sampled breach.
No duplicate predecessor-prefix reconstruction was added.

Artifact root under the existing development artifact directory:
`go2_stop_conditioned_settling_maze02_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| `result.json` | `643b3d41e6e11398706ec8dd295a180356591d7e1ff7a82e95e272553a4a5918` |
| `no_rgb_direct_stop_conditioned_settling_maze_02_audit.json` | `aeac508e24c12929ad7f16dc373e648ea0281f6d9ed1cba5de8c6f2b0bb82ad6` |
| `no_rgb_direct_stop_conditioned_settling_maze_02_readout.json` | `70c7768f380274d4f62e9453cdd28cf3e6d28b9e19e4ac62ba59c87b14554404` |

The full task remains active. The independent layout-0 full-RGB JEPA case has
started with the original fixed model and controller; its matched reactive case
remains queued. Independent-layout reliability, matched training/planning/memory
comparisons, realistic sensing and timing, and bounded real-platform evidence
remain outstanding.
