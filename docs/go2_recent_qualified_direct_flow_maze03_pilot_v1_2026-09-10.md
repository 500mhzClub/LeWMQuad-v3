# Prospective isolated recent-reference native maze3 pilot V1

Execute one fresh reused-development maze3 episode with
RecentQualifiedDirectFlowController, full input and JEPA model
4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.
Keep direct-flow floor registration and the existing planner, model, mission,
memory, raw sensor gates, command budget, physics and evaluator. The only
controller intervention is retention of the immediately previous already
qualified visual reference under the frozen original missingness/rigid-fit
gates. Do not introduce the partial-floor-height intervention.

The completed prospective replay is
16c6917d2e4c2b728bd08a330290e141a93a500f9f28b7a3abd27e9d4f51926a,
root go2_recent_qualified_direct_flow_maze03_prefix_v1_attempt_001.
The original native input is
6be6aa6e60be4b1a9e3d9b79aa3b55e4becc3265fe208ce900c8e1962c3db3ea,
root go2_direct_flow_maze03_pilot_v1_attempt_001.

Require1207 complete replay observations,1193 complete normalized original
decisions,1203 exact raw forecast banks, one extra reference attempt qualified
at1193, and the first changed requested command at1206: originalzero becomes
leftturn[0,0,.45], with the candidate nonterminal. Reconstruct every saved
comparison. Keep actual changed visual/downstream state from1193 onward.

The physical prefix must contain61050 identical physics samples and1207
identical public observations. Every candidate decision through1206 must equal
the frozen prospective decision. Compare original state only before the extra
reference attempt. Record actual completion of the changed command. No old
following physical outcome may stand in for this new execution.

Preserve the original prepared native queue and its separate supervised
contact-horizon waiter. Preflight may verify inputs while they run. Actual
launch additionally requires the explicit SHA of the complete fixed contact
waiter result, all its source/output bindings, its original native verifier,
the exact queue-completion receipt, and absence of competing native runners
or workers. A negative scientific outcome may pass admission; a failed
integrity/raw audit or prefix must not be bypassed. No automatic retry.

Use one native scene, CPU deterministic model inference and one-thread BLAS/
OpenCV/Torch. Require32GiB available RAM and51GiB artifact free space:
40GiB reserve,10GiB collection allowance and1GiB persistence headroom.
Recheck capacity after predecessor authentication and immediately before
launch. Preserve3 warmup,3000 shared navigation and10 terminal drain ticks.
Commands commit100ms through50 physical2ms steps. Physics pauses during compute;
record full loop timing and do not claim real-time operation.

The fresh worker must persist raw artifacts, replay the complete controller
with a separately loaded identical model, audit sensors, commands, renderer
witnesses and strict visibility, compare the prospective physical prefix and
preserve all later outcomes. Native pose/topology remain evaluator-only.
Every failure is retained. Output ownership is exclusive:
go2_recent_qualified_direct_flow_maze03_pilot_v1_attempt_001,
case full_jepa_recent_qualified_direct_flow_maze_03.

No navigation, independent-layout, JEPA advantage, memory advantage, pose
uncertainty, physical-clearance, real-time, hardware or deployment qualification
follows from source tests, prefix agreement or completed execution alone.
