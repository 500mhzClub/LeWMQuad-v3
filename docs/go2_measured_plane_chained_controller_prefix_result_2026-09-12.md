# Chained tracking replay passes the original return-turn failure frame

The completed replay consumed 3,114 original public observations, frames
0–3113, and stopped at the first changed command or terminal state. Every
complete original-controller decision reproduced the recorded native decision.
The candidate's first recorded chained anchor reacquisition was at frame 3103.

At observation 3113 the original controller stopped with
`SENSOR_OR_MODEL_FAILURE` / `same-episode current visual evidence required`
and requested `[0, 0, 0]`. Its retained mission receipt remained at frame 3112
in RETURN. The candidate admitted a current primary-camera visual pose using
reference frame 3112, had no failure or terminal state, advanced its RETURN
mission receipt to frame 3113, and selected `right_turn`, requesting
`[0, 0, -0.45]`. Thus the candidate remained active at the exact observation
where the original lost tracking.

The original made 3,098 actual model forward calls and the candidate 3,099.
The 3,098 common raw forecasts matched; the extra candidate call occurred at
the final observation, when the original stopped before forecasting. Both
fresh independent corrected models remained unchanged and without gradients.
Public input packets remained unchanged.

This is causal recorded-history evidence through the first intervention.
The changed command was not executed, no observation following that command
was consumed, and no new native trajectory, round trip or backtracking result
is established. The candidate observer was not separately replayed in an
additional observer-only experiment. The single-pass timing optimization is
not part of this candidate.

## Completed identities and verification

Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_plane_chained_controller_prefix_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `result.json` | `9703206c207fbc642164c1972a6eabdc70ebab3f1ced0c2df33c0ad74101eb72` |
| `launch.json` | `ddf403f8bd376f8db22a2e5fc88e3b45f816bf8091f9c73619fef1490f96fbc2` |
| `context_decisions.jsonl.gz` | `9901493831d49e20d91978d3cf5c5304bd05bb8b8eecd863a70868a482ccab4a` |
| `resource_monitor.jsonl` | `a0e802b8b6b76f5869a10ac4500dd085a5af04271fa4222e830e5aba8906c619` |
| `report.json` | `274de6417e69864b5ef961352b1c2bf9cabb687e125db00b4085e4da0e8392eb` |

Status: `MEASURED_PLANE_CHAINED_CONTROLLER_PREFIX_V1_COMPLETE`.
The original child PID 2963794, creation time 1789182926.58, ended without a
failure artifact. Wall time including output and input verification was
9,060.495 seconds. Corrected model state identity remains
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.

The original process reconstructed all saved comparisons and consumed public
packets and reauthenticated the original raw inputs before and after execution.
After completion, all 2,594 result-bound source hashes and all four artifact
hashes were independently checked, the launch/result source maps matched,
and both report copies matched. This independent hash check does not claim a
second replay or another full public-packet reconstruction.

The original chained waiter PID 2930187 was live in completed-child verification
when this note was written. Its completed result is still required. Do not
bypass it or restart the completed child. Once the original waiter completes
and ends, pass its actual result SHA to the already prepared native launcher's
preflight and then, after admission passes, execute the fresh candidate pilot.
The launcher and dynamic physical-prefix verification are documented in
`docs/go2_measured_plane_chained_native_launcher_preparation_2026-09-12.md`.

Reliable round trips, independent-maze replication, matched scientific
comparisons, realistic sensing/timing and bounded hardware evidence remain
unproven. The full goal remains active and incomplete.
