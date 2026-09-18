# Fresh reactive execution with current sensing, memory and settling

This prospective maze0 pilot tests the separately implemented
ReactiveFloorTransportController. It shares the learned controller's paired
primary/auxiliary RGBD, measured joint/gyro observations, visual tracking,
floor registration/transport, persistent map/contact evidence and settled
outbound/return mission. Its existing observed-route reactive rule replaces
learned candidate prediction, predicted feasibility and scores; it has no
high-level world model or forecast residual. The pretrained low-level gait
remains unchanged. This is a matched method comparison on reused maze0, not an
isolated ranking ablation or independent-layout reliability result.

Require completed reactive prefix result
71a5ecd8486d6d8354762c5dc249307cc7d1bf7dc57fe6d1f2df76372d5aaba9,
and the actual completed --learned-result-sha256 from
go2_measured_floor_transport_maze_pilot_v1_attempt_001. Admit the latter using
its complete raw audits and prospective intervention prefix, preserving any
negative physical/strict-visibility outcome. No success-based model/controller
selection. Require every saved reactive prefix decision to agree with the
four-observation report: three original zero commands, then forward[.2,0,0]
at frame3 instead of the learned left arc[.16,0,.45]. Neither policy is terminal
at that observation. No later old observation supplies a reactive outcome.

After the currently ordered learned maze0 execution/readout and fixed learned
layouts1,2,3 cohort, inspect current hardware and absence of competing native
scenes. Run scripts/run_go2_reactive_floor_transport_maze_pilot_v1.py with the
actual --learned-result-sha256 and --preflight-only, then without the flag if
admitted. Exclusive root:go2_reactive_floor_transport_maze_pilot_v1_attempt_001.
No execution of this pilot is implied merely by preparing these sources.

One fresh CPU scene, one OpenCV/PyTorch/BLAS thread, original robot geometry,
appearance/physics seed, gains, gait and friction. Original3000navigation ticks
shared outbound/return, three warmup observations, ten terminal zero commands,
50physics steps per100ms decision, physical guards and stop rules. Retain
receipt-inclusive timing and all raw RGB/depth/body/gyro/physics/commands plus
renderer acquisition witnesses. Admission requires32GiBavailableRAM and the
original10GiBcollection+1GiBpersistence above40GiBreserve. These are capacity
checks, not enforced OS quotas. Refresh resources after input validation and
monitor throughout. Physics remains paused during computation.

The new collector differs from the learned collector only in constructing the
reactive controller without a model, the reactive command-role/status labels
and explicit no-prediction metadata. The new full raw audit reconstructs the
same complete packets and every reactive decision with a fresh reactive
controller, retaining original physical outcomes, actual-command/slew checks,
sensor geometry and strict visibility tests. The command-audit implementation
differs only in the exact reactive role label. No high-level model is loaded
for collection or replay, so no high-level model-state assertion is substituted
for the complete reactive command replay.

Compare this fresh native prefix directly against the completed current learned
native episode: all900physics samples before the first changed command, all
four paired public observations and shared pose/map/settling receipts must
match exactly. Both native definitions use the current mission wording, so
this comparison permits no receipt normalization. All four full reactive
decisions must match the completed prospective replay; original and candidate
requests must match their actual tapes. Only the first three old commands must
agree; following physics belongs to the new reactive execution. A physical
stop partway through the new command is an actual outcome, not a reason to
invent a completed command or reuse its old outcome.

Retain collection and audit artifacts before prefix validation, preserving
failures without retry. A raw-valid navigation failure is a complete negative
scientific result. An infrastructure or prefix mismatch fails the attempt and
retains its evidence. Do not relabel old reactive or learned episodes.
Final evaluation requires both settled arrivals, physical retracing and strict
sensor/guard checks. Report actual terminal events and physical candidates
separately from verified success. Completion of this pilot does not establish
JEPA superiority, a memory advantage, independent-layout reliability, real-time
operation, calibrated pose uncertainty or hardware/deployment qualification.
