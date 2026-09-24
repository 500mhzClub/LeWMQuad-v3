# Auxiliary depth native goal probe V1

Execute two fresh CPU missions on reused development layout `family_episode_039`,
with the fixed corrected seed-2026091001 full-JEPA and full-direct models and
unchanged seed-2026091501 case order. Require complete training-only correction
admission, its completed readout, the passing robot-visible 20-frame sensor
capture, and the completed auxiliary controller prefix integration before launch.

The intervention adds the fixed 640-by-480 auxiliary depth camera at body mount
[0.35, 0, 0.08] m, pitched down 30 degrees, with 0.2-to-5-m public metric depth.
Both camera views render all 33 robot visual geometries in the verified complete
floor/wall/robot draw order. Primary camera calibration and the observer are
unchanged. Capture primary then auxiliary at the same paused physical sample;
pair their measured clocks and primary RGB identity before controller admission.
Retain native depth, valid masks, RGB and evaluator-only segmentation/pose audits.
Never expose native pose or segmentation to the controller. This ideal sequential
acquisition has zero simulated delivery latency; wall time is separately measured.

Auxiliary depth contributes measured floor and occupied geometry, preserving all
sample returns. Whole-foot floor coverage uses complete measured patches or grid
coverage. Other/unknown primary and auxiliary evidence remains collision checked.
The model sees its original input contract. Preserve training-only XY correction,
the eight-step planner, all nominal segments with 0.45-m clearance, the first-step
articulated veto, route/view state, and failure latching. Execute one 100-ms command
per observation with the original actuator slew, 240 navigation ticks and ten
terminal zero commands. Preserve observed 0.04-m arrival and the original native
0.06-m quiet/contact goal gate.

Freeze the complete sources and dependencies before the exclusive
`go2_auxiliary_depth_goal_probe_v1_attempt_001` output. Use two independent native
processes with one numerical thread each, within the passing four-process native
scaling benchmark and current visible-robot capture evidence. Require 32 GiB
available RAM and an 8-GiB output allowance above the 40-GiB storage reserve;
record hardware capacity and monitor live use and full command-iteration timing.

Require exact first-four primary and auxiliary public packets and first-900
native physics samples against the successful visible-robot prefix. Replay every
new decision using a fresh assigned corrected model. Keep the original native
goal, actuator, setup, stop and primary sensor auditor functions. Validate the
complete 35-node raster before applying the original static precision check.
Audit each auxiliary metric image against its recorded native camera pose and
physical static geometry; body occlusion cannot qualify under that static-surface
audit. All returns still remain in controller geometry. Require both cameras'
measurement gates for a counted native success. Reverify complete model state,
sources and artifacts after execution. Preserve all failures without retry/resume.

This is a development integration probe with zero independent maze layouts.
Physics is paused during computation. Independent-maze missions, physical
backtracking, matched reactive/nonpredictive and planning/memory comparisons,
realistic sensing/timing and bounded hardware evidence remain necessary.
