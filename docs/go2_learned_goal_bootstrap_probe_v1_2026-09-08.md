# Prospective learned-model goal bootstrap probe V1

This is a two-episode development integration probe, with a real downstream
goal and online learned action selection. It is not an independent-maze result,
model comparison, training attempt or navigation qualification. The completed
96-episode family retains its failed four-second action-necessity gate. None of
those outcomes is used to select a checkpoint or fit this controller.

Use the first preregistered seed 2026091101, full inputs, JEPA objective from the
completed independent-pulse parallel study. Its fixed checkpoint SHA-256 is
`59cec8efec26a2318bdfecb0315f84a17a244f7d03f4029d9f408cdcd3b6abc7`.
Authenticate the whole historical study and exact fit receipt before launch;
reload evaluation-only with the recorded data/schedule/configuration binding.
The short-pulse training has a known domain gap to these four-second plans and
moving replans. No probability-calibration or accuracy guarantee is inferred.

Run fresh physical instances of family_episode_052 then family_episode_039,
the cluster-00 mirrored panels with the first appearance. Their physical specs
are reused exactly; their old action assignments do not enter this controller.
Use a new exclusive root `go2_learned_goal_bootstrap_probe_v1_attempt_001`.
No retry, resume, model substitution or parameter change within the attempt.

After 1.5 seconds of recorded gait settling, acquire joint RGB-D continuity
from frame zero. Keep three zero command ticks for four real causal packets.
The instruction is goal (1.2,0) metres in the first observed body frame. Every
command requires current joint visual pose evidence; gyro only checks visual
rotation consistency. Transform the remaining goal into the current body
frame. The fixed existing six-candidate selector uses rollout outcomes,
terminal predicted distance reduction minus 1.2m times cumulative contact score,
with hold-first ties. Do not add orientation bonuses or adjust scores after
seeing outcomes. Execute the first five 100ms commands of the selected plan,
then predict again from the latest four observed packets. Keep every forecast.

Allow at most 240 navigation ticks, including arrival dwell. Observed arrival
requires distance at most .04m and ten zero-command intervals. Otherwise record
the model/sensor failure or budget exhaustion. Drain ten zero-command ticks
after a controller terminal, unless the unchanged native physical guard stops
execution. Any physical contact, body/speed/domain stop or acquisition failure
is preserved. Source and checkpoint hashes are checked before/after episodes.

Raw audit reconstructs sensors and native contacts, replays the observer and
actual checkpoint, compares every decision/forecast exactly, and checks exact
float64 requested commands and independently float32-clipped applied commands.
The native goal evaluator is separate from control. Verified goal-reaching
requires the observed candidate, complete terminal drain, no disallowed contact
or stop, and the last 501 physics samples all within .06m of the instructed
goal at speed at most .05m/s, with the last 500 requests zero. Short progress is
never arrival. Preserve strict per-ray results and the existing separate stable
interior/near-occlusion measurement gate. Failed measurement invalidates any
success claim, even if raw commands replay exactly.

Inspect hardware, competing processes and storage before launch. Use one fresh
native worker at a time so measured online-loop timing is not confounded by a
second simulation. The previous four-worker benchmark supports collection
throughput, not realistic simultaneous control latency. Reserve 4GiB output
plus the standing 40GiB free-space floor; require 32GiB available RAM. CPU and
OpenCV/BLAS/Torch thread count is one. Record resources during execution and
report acquisition, model/observer, and complete command-iteration wall times.
Physics is paused during computation; this is not a real-time or hardware run.

All new runner, controller, auditor, checkpoint helper, focused tests and this
protocol are source-bound before any native execution. A raw audit failure
stops later launches; scientific controller/physical failures are retained and
the second predetermined episode still runs if infrastructure is intact.
