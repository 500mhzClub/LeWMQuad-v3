# Learned and nominal forecasts choose different first navigation commands

Completed `scripts/replay_go2_measured_plane_forecast_source_prefix_v1.py`
under `docs/go2_measured_plane_forecast_source_prefix_v1_2026-09-11.md`.
The exclusive root is `go2_measured_plane_forecast_source_prefix_v1_attempt_001`
on the navigation artifact volume. Session 40918 exited 0; the original replay
owner ended. Launch SHA-256:
`f06e93dcb7dd4bc81242a7274aec7751edde77a6c0d66e05356175d061060b76`.
Result SHA-256:
`0d080057b6fb4802e103623496e92e0bbb7be77474cfdbf9a9e6d08c500b8478`.

Both controllers consumed exactly four original public observations, frames
0–3. At the first navigation decision, frame 3, the forecast source changed
the command:

| Arm | Selected action | Requested `[vx, vy, yaw_rate]` | Feasible actions |
| --- | --- | --- | ---: |
| Frozen corrected no-RGB direct model | Left arc | `[0.16, 0, 0.45]` | 6 |
| Nominal requested-motion prediction | Forward | `[0.20, 0, 0]` | 6 |

Both were nonterminal and outbound. Complete visual, registered-floor, map
and mission receipts were identical. Both used the same waypoint score
contract and the same six allowed actions. The learned baseline reproduced
the complete recorded measured-plane decision after removing only its added
source provenance and root metadata. The different choice at this boundary
arises in scoring under a common feasible action set.

Actual forward-hook counts were **one learned model call and zero nominal
model calls**. Both independent assigned model states remained unchanged,
with no gradients. The nominal arm retained its explicit perfect requested
velocity-tracking assumption and constant contact reference. It still used
predictive planning and observed residual correction.

The replay stopped at that first changed command. No following public
observation was consumed, no changed command was dispatched, and no
counterfactual navigation outcome was inferred. This demonstrates that the
forecast source affects an actual controller choice; it does not establish
that the learned choice is better, that JEPA helps, that predictive planning
beats reactive control, or that persistent memory helps.

The current controller's waypoint pose utility already uses the first
100-ms forecast. Its 800-ms forecasts contribute contact risk and path
feasibility. A future one-versus-eight-step comparison must distinguish those
effects and must not be called a complete planning-on/off ablation. Relevant
source is `lewm/executed_waypoint_score_development.py` and
`lewm/executed_horizon_final_goal_development.py`.

Validation: **25 tests passed in 2.30 s**, session 2097, covering exact
source provenance, complete original decisions, mismatched evidence,
terminal/command boundaries, and complete saved-stream reconstruction.
Source preflight passed, session 34310, with 2,492 source bindings, 77.33 GB
available RAM and 591.06 GB artifact free space. The completed frozen runner
reconstructed every saved comparison against the reference and reauthenticated
the original raw inputs before and after execution. A separate read-only
ended-owner check revalidated all source/artifact bindings and reran complete
saved-output reconstruction, session 42538, exit 0. No model or observer was
rerun by that final check.

The separate fresh learned measured-plane native scene remained live at tick
872, parent PID 2916106 and worker PID 2916239, under launch
`93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb`.
No native terminal result or failure was present. Its physical outcome remains
pending. The nominal command has not been tested in fresh physical execution.
