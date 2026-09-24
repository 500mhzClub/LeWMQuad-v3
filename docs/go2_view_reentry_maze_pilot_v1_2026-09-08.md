# Prospective translating view-recovery maze pilot V1

This new native experiment tests the policy specified in
`docs/go2_view_reentry_maze_prefix_v1_2026-09-08.md` on reused development maze 0.
It requires a completed, hash-bound public-sensor prefix that matched the
predecessor observations/maps/mission/forecasts/constraints and all other decision
fields until the first changed command. That command must be an explicitly
recorded translating view-recovery exception, with no terminal or model failure.
Do not launch from a partial or failed prefix. Record its completed result in
`docs/go2_view_reentry_maze_prefix_result_2026-09-08.md` before native launch.

The collector and fresh-controller raw audit are exact copies of the completed
executed-waypoint pipeline apart from the separately named controller and status
labels. They retain the original paired public RGB-D and body sensors, physical
scene, pretrained PPO locomotion policy, six primitive bank, 100 ms replanning,
same corrected JEPA checkpoint, continuous 2 ms native physics, command slew,
observed map and mission state, shared 3000-tick outbound/return budget, observed
arrival dwell, native arrival verification, physical backtracking checks, final
drain, visibility/measurement gates, physical stops and persistence rules.
No native pose enters the policy and no model is trained or selected.

The worker collects fresh observations after every actually executed request.
It persists raw evidence, then replays it with a fresh controller/model and
audits commands independently. In addition, the exact predecessor/new native
physics and public-sensor prefix must agree through the observation preceding
the bound first changed command. Their preceding commands must match, and their
changed requests must equal the bound prefix requests. This comparison does not
claim that observations after the changed command match the old trajectory.

One CPU scene worker requires 32 GiB available RAM and a full 10 GiB collection
allowance plus 1 GiB persistence headroom above the unchanged 40 GiB free-space
reserve. This is the same allowance as the executed-waypoint and prepared
reactive baseline collectors. Assess hardware immediately before launch; do not
lower the envelope to bypass a failed resource admission. No concurrent native
scene is budgeted. Sources and all completed input identities are checked before
and after execution. Preserve partial failures, worker logs and resource records.
The exclusive output is `go2_view_reentry_maze_pilot_v1_attempt_001`.

`--preflight-only` performs completed-input/source checks and reports memory and
storage admission without creating output or a scene. Actual launch requires
`--prefix-result-sha256` with the completed prefix result hash. The reactive
baseline is ahead of this experiment in the native execution queue while storage
is constrained. No cache retirement authority is implied by this protocol.

A successful raw audit is not a successful mission. Report observed/native
arrivals, outbound and return edges, terminal reason, physical and acquisition
stops, complete/censored execution intervals, and timing even on navigation
failure. This contributes zero new independent layouts. Physics pauses during
computation; no real-time, sensor-latency or hardware qualification is established.
Matched baselines, JEPA/RGB/planning/memory attribution and independent maze
success remain necessary for the full navigation goal.
