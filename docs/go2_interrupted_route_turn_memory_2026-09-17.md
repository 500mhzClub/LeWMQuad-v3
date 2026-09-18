# Interrupted route-turn memory pilot

The completed polygon-floor batch produced two supervised round trips and two
JEPA goal-only outcomes. The final JEPA return applied no translating commands,
reversed turn direction 127 times and eventually lost visual tracking. Its first
route-turn interruption occurred at frame 2672, before clearance rejection.

The observed-map forecast probe replayed the actual mapping updates through
frame 4204, matched retained map counts and every sampled saved neural segment
clearance, and verified delivered noisy-depth digests. At 2640, 2668 and 2672,
both turns passed the reserve checks under neural, pose-command and command-
history forecasts. At 2876, the right turn failed under neural and pose-command
forecasts but passed command history; at 2880 and 4208 it failed under all three.
The later conflict is therefore not uniquely a neural-forecast effect. No
alternative actions were executed in this probe. Receipt: final batch mission's
`polygon_return_forecast_clearance_probe_v1.json`.

The new runtime remembers a local route turn only when measured heading changed
at least 0.025 rad in its requested direction before visual recovery interrupted
it. After recovery, the same local heading objective can use the other direction
if that candidate passes the existing forecast-clearance predicate. The alternate
attempt persists until the measured target heading, a changed local objective,
translating action, or another visual recovery. A blocked alternate requests hold.
Failures apply only within 0.20 m and 0.30 rad of the recorded objective, with at
most sixteen records and a reset on mission-leg change. Ordinary viewing tasks
and terminal positioning retain their existing behavior. Both directions may
fail; this change does not claim a guaranteed escape.

Three focused tests passed (session 81859). They exercise observed motion,
local/generation scope, clearance rejection, continued alternate turning,
priority of actual visual recovery, and measured-heading/translation release.
The saved return-decision prefix first differs at frame **2700**, immediately
after the first recovery: original right turn, proposed forecast-clear left turn.
The readout stops at that first changed action and proves no later trajectory.
The initial diagnostic invocation lacked a generation field on uncommitted plans;
the corrected invocation used the verified return-leg generation 1. Receipt:
`interrupted_route_turn_saved_activation_v1.json` in the final batch mission.

Run **one** prospective exposed-layout-1 JEPA pilot, with the same frozen readout,
six candidates, polygon mapper, tracking/view thresholds, reserve/coverage/
stopping/dispatch checks, depth noise, ideal gyro, CPU group, deadline and
4800-tick budget. Keep the original route-search implementation to isolate this
behavioral intervention; the separately verified axis-distance acceleration is
not included. Plan: `docs/go2_interrupted_route_turn_memory_plan_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_interrupted_route_turn_memory_development.py`.

The source and plan were frozen by `--prepare` (session 49929, exit zero). No
other experiment process was live; the output drive had 6.6 GiB available,
enough for one recording under the existing four-GiB launch threshold. Run
native execution sequentially, with no competing training or replay. Preserve
the full failed comparison and this pilot through diagnosis. Evaluate only after
the mission owner exits and finishes recording persistence. No hardware or
fresh-layout reliability claim follows from an exposed-maze pilot.

The pilot launched in session 79667, owner PID 4147558, verified live. Launch
metadata confirms `RouteTurnMemoryRuntime`, `ProjectedPolygonFloorRoutingMap`,
route-turn memory enabled, axis-routing acceleration disabled, and unchanged
JEPA readout state
`f372e75c1a5c4b3933beb9d59ee97158ce17a8a2b567a89c9be59b74cf8112a8`.
Root: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_interrupted_route_turn_memory_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_interrupted_route_turn_memory_launch.log`.
Detailed navigation progress is in the root's `worker.log`. No outcome yet.

## Completed pilot

The owner exited zero after persistence. Evaluation session 16995 exited zero:
**verified goal-and-home round trip in 353.66 simulated seconds, zero disallowed
contacts, no pipeline faults**. Goal/home arrival frames were 2180/3532; the
maximum physical distances throughout their one-second quiet dwells were
13.57/3.93 mm, with maximum 100-ms speeds 0.02767/0.03125 m/s. Both independent
arrival checks passed. All eleven unique return corridor edges reversed
outbound edges, with zero invalid graph transitions.

The new behavior was exercised on 22 plans at frames 2228–2312, all requesting
the alternative left turn after interruption of the right-turn route approach.
Seventeen of these plans were on time and committed; five were late. Every
selected alternate passed the existing forecast-clearance predicate. The next
plan, frame 2316, selected a translating left arc. This is prospective evidence
that the new memory behavior can execute and navigation can complete; the
single asynchronous exposed-maze outcome does not isolate a causal benefit or
prove repeatability or generalization.

There were 3535 camera pairs and 866 selections: 780 on time, 86 late (90.07%).
Live wall time before archival was 464.97 s. Actions were eight holds, 144 left
turns, 235 right turns, 118 right arcs, 111 left arcs and 250 forward plans.
The return included 30 left turns, 85 right turns, 35 left arcs, 61 right arcs,
115 forward plans and two holds. Full real-time execution remains unqualified.

On matched selected-action 700-ms windows, neural XY RMSE was 9.426 mm versus
7.355 mm for pose-command and 8.766 mm for command history. Neural yaw RMSE was
0.883 degrees versus 0.746 for command history. No JEPA forecast advantage is
established. Preserve this full recording as the exercised current success
reference. The next scientific test should use a fixed fresh development maze
and matched model treatments, rather than adding repetitions to this pilot.
