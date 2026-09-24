# Residual planner memory comparator and original-process status

The prefix has now completed and its saved boundary has been independently
checked. Result `f802f1c14263878520e4d9221128bd037442c2ed9fdae62de2ba589eb1efaf49`;
original handle 82947 exited 0. See
`go2_residual_current_observation_planning_prefix_result_2026-09-10.md` for the
completed evidence. The pending replay statements below record its earlier
launch state and are superseded by this completion.

This goal turn made progress by adapting the already implemented current-paired
planning-map comparator to the complete residual-anchored continuation selector.
The original observation, contact, localization, model-history and mission
state remain. This is a planning-grid persistence comparison, not a fully
memoryless controller or a new navigation result.

The 2026-09-09 memory scope audit is historical: it was superseded by the
current-observation planning implementation, causal prefix, native maze0 pilot
and readout. Those results were found during source discovery this turn and
must not be treated as missing work. Original native result
`dca1757aa1358ca48bf5d40240337aafb43391d0306b376309e1b3239da89780` and paired
readout `3d11666a31d229355bafbd50f622c73ef0dd7acf2882654eb07fe1b8e7747214`
report a tracking failure, strict visibility failure and no round trip. Reusing
that map does not convert its failed experiment into positive evidence.

An initial duplicate implementation created during this turn was removed before
any experiment launch. Its development tests had 17 passes and one failure in a
test intercepting the wrong inherited waypoint module. Only the four files just
created for that duplicate were removed; no existing implementation, scientific
artifact or failed native result was changed. The final implementation reuses
`CurrentObservationPlanningMap` directly and adds the residual selector wrapper.

Final tests 46219 exited 0: 23 passed in 7.24s. They cover the inherited
observation/mission/residual pipeline, retained articulated contact queries,
the full residual selector's shared current planning cells, scope metadata and
causal-prefix rejection of changed sensor/mission/residual/forecast evidence.
The checked predecessor current-map tests also cover reobserved old cells,
immutable view bounds, missing/stale data and waypoint changes. No scene ran.

Source/input preflight 40020 exited 0, with 1755 source bindings. Hardware:
16 physical / 32 logical CPUs, 3.4% busy, 72,736,116,736 available RAM bytes,
657,111,564,288 free artifact bytes, 21,356,310,528 free workspace bytes, idle
GPUs. The single CPU replay requires 32GiB available RAM and 40+1GiB artifact
headroom. Preflight checked original artifact bindings but did not substitute
for the full original completion verifier.

Submitted exactly one original replay process: handle 82947, PID 2649125,
creation 1789025076.66, running
`scripts/replay_go2_residual_current_observation_planning_prefix_v1.py`.
At the latest observation it remains live in full original input authentication;
no prefix result is claimed. Preserve this process rather than launching another
attempt. The exclusive output is
`go2_residual_current_observation_planning_prefix_v1_attempt_001` under the fixed
RecoveryStorage navigation artifact root.

The replay uses the completed residual-anchored maze2 result
`818a598ca6336866cf5f4768c11edaf67c8c1ca60896c305f93f69fd0ed5230c`, launch
`c4681e31baaf5dc1c8fa368854ddbcf19866090787741b946c4d37b9a3a477b3`, and two fresh
loads of original corrected model state
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
The new expanded-model assignments in the native queue are unchanged. The
prefix will reconstruct complete original decisions, compare retained contact
and observation state, and stop at its first changed command or terminal.
It must not consume later packets on a changed trajectory or infer unexecuted
physical outcomes. Native use remains a separate later experiment.

Original native parent 2636286 / worker 2637549 remain live, with original
creation times 1789017984.16 / 1789018454.4. The 31st collection stopped at
observation 2794 on floor-registration correction limits, without an arrival.
Its raw audit, physical-prefix comparison and top-level result remain pending.
There are still 30 completed audited native episodes and zero verified round trips.

The original six-model waiter 14784 / PID2641948 remains live. Verification
20532 exited 0: all 1890 source bindings remain unchanged, launch
`6e96b6bc78f08f8dd7b5af3ad25e48b78ca915a5c3b5ede37efe0bc8a7be9b5e`.
It owns the next native launch; no duplicate waiter, model assignment change or
new independent-layout scene was started.

Next: finish and independently inspect the original prefix process, including
any failure; continue polling native 25801 and waiter 14784; then bind the
available planner, reactive and planning-memory treatments into the independent
population protocol. Reliable independent navigation, planning/JEPA/memory
comparisons, realistic sensing, timing and bounded hardware evidence remain open.
