# Measured-plane native pilot: completed audited negative result

The original measured-plane recovery run has completed collection, raw audit,
actual physical prefix comparison, and terminal accounting. It reached the
outbound goal on reused development maze 02 but did not complete a round trip.
This is simulation evidence, not independent-maze, real-time, or hardware
qualification. The full navigation goal remains incomplete.

## Frozen evidence

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_plane_dispatch_recovery_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `result.json` | `4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18` |
| `launch.json` | `93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb` |
| `no_rgb_direct_measured_plane_maze_02_audit.json` | `85cb5cea7ac2c585b3ec8fcf2b4e06007088e9ad2fdfafb093e2a3fa400a39ec` |
| `no_rgb_direct_measured_plane_maze_02_prefix_comparison.json` | `b35db19277cfcece528cee75a5fdf342e704a66a04b587be4081e04e5db069cb` |
| `no_rgb_direct_measured_plane_maze_02_worker_terminal.json` | `f30c17f11404d64fdd6a35f1169bdc923755a96570d1d48138c304d15e2fa4c6` |

Parent PID 2916106 (creation time 1789162140.42) and worker PID 2916239
(1789162190.15) both ended. Parent status is
`MEASURED_PLANE_DISPATCH_RECOVERY_V1_COMPLETE`; worker status is
`MEASURED_PLANE_MAZE02_COLLECTED_AND_RAW_AUDITED`. No `failure.json` was
present. The scientific failure is preserved in the completed result.

The completion check rehashed all 2,562 result-bound current source files and
the four terminal evidence files above, checked the exact result/worker record
equality and audit/prefix consistency, and verified both owners had ended.
This lightweight completion check did not rehash all 18,784 result-bound
artifacts. The parent completed its programmed artifact verification; the
dependent queue admission performs its own required authentication.

## Demonstrated outcome

- 3,124 observations, 3,123 completed commands, and 156,900 physics samples.
- Outbound arrival at frame 3062 passes the evaluator's one-second goal and
  quiet criterion: maximum distance 0.04377755639348901 m and maximum speed
  0.0441619731798587 m/s over that window.
- 15 outbound crossings over 10 distinct open edges; no invalid crossings.
  Zero return crossings and no physical retracing of the outbound route.
- Zero recorded native contact samples. This does not certify general
  collision clearance or future navigation safety.
- Terminal schedule reason `SENSOR_OR_MODEL_FAILURE`, with ten terminal zero
  commands. No physical or acquisition stop was reported.
- Full raw sensor reconstruction, auxiliary RGB reconstruction, model/command
  replay, command audit, and unchanged-model checks pass. Strict physical
  visibility passes with no hard measurement-failed frames.
- The actual 123-observation physical/public prefix and complete candidate
  decisions match the registered comparison at intervention frame 122. This
  validates that intervention, not the later return outcome.
- Observation/control median 1440.4902195 ms, p95 2996.3371336999994 ms as
  calculated in the worker readout; all 3,124 samples exceed 100 ms. Physics
  was paused during computation. No real-time claim follows.

The controller-level failure diagnosis remains the exhausted visual bridge
budget during the return turnaround; see
`go2_measured_plane_return_turn_trace_diagnosis_2026-09-12.json` and
`go2_measured_plane_return_anchor_probe_and_candidate_2026-09-12.md`.
Visibility passing does not mean feature tracking remained available.

## Queue transition and next decision

After the original parent ended, the existing waiters admitted and launched
both dependent jobs against the exact completed result SHA above:

| Job | Owner PID / creation time | Launch SHA-256 |
| --- | --- | --- |
| `go2_nominal_measured_plane_maze02_v1_attempt_001` | 2933558 / 1789172031.45 | `463ec812fb12e16f3b8543a28d69b9275774488ebc3759b6fd457837abc6d1e6` |
| `go2_measured_plane_single_pass_full_history_v1_attempt_001` | 2933637 / 1789172064.29 | `a38fccef6002ca980ca65a47ca9c77079cabc2ab0431a1e07484531098d999a7` |

Both live processes matched their launch PID, creation time, command, and boot
identity. All 2,577 nominal and 2,583 timing source bindings were rehashed and
matched. Nominal has a live native worker, PID 2933696, creation time
1789172078.87. The timing replay has written a complete comparison through
frame 13 with original and normalized-candidate decision equality, unchanged
public inputs, and one actual model call per controller. These are initial
progress observations, not completed outcomes or a full-history speed claim.
Available memory was 72.7 GiB and artifact storage had 543.4 GiB free at the
dispatch check.

The reactive native waiter remains behind nominal, and the chained controller
replay waiter remains behind the reserved timing replay. Do not launch
duplicates or bypass the existing waiters.

The next perception decision depends on the chained controller replay's
actual first changed command or terminal decision. The small fixed-pair
probe and prepared native-prefix helpers alone do not establish recovery.
Any candidate native execution must use authenticated replay evidence and
remain serialized with the existing nominal/reactive native queue. A new
perception candidate also needs matched perception across comparison arms
before claiming planning, JEPA, or memory benefit.
