# Closed maze2 collection: prolonged holding after a single feasibility recovery

The residual pilot's collection is closed, but its raw audit and final physical
prefix comparison remain pending. Collection SHA-256
f7a946687f108c354a7c6f4e5155530875ca0da7bcd7e57f9564ff89c23fa5b4:
3014paired observations,3013completed commands,151400physics samples and10terminal
zero intervals. First terminal at observation3003 is MISSION_TICK_BUDGET_EXHAUSTED.
No physical or acquisition stop, no observed arrival, final reported goal
distance3.847554391690519m. This is not yet a verified native outcome.

The closed-stream diagnosis96580 exited0. Root:
go2_residual_maze02_stagnation_v1_attempt_001. Result
97ddebca3610af75f0f7a84bb8a74df2e7fb36dbdcfc4a58a1a94a122d0496b5;
launch0a16a068525cb2a0beb7fb803aa2fc45a2448096955a48acf43d16c4ea1c3ece;
1,675sources. The collection, native launch and compressed stream were checked
before and after analysis. Stream SHA-256
d4aecc3d9dde5f3d1cf42b6c8bbbea5c5b360406cd5565f0c0b709700014172d.
No model, evaluator pose, action rescoring or new execution was used.

Of3000active selections,2643are hold,147left turn,91right turn,75left arc and
44right arc. Forward is never selected. There are2646active zero requests,
including the3warmup observations. Only one residual-feasibility fallback is
attempted, at463. The longest continuous selected-hold run is554–733 inclusive,
180observations; many shorter hold runs are separated by turns.

At463 the corrected right arc is the only eligible action, exactly as the
prospective prefix predicted. At503, right arc has an original nominally clear
path, but its saved utility -0.0010909 is below hold -0.0000854. By1000,2000and
2990, all three translating actions fail the original nominal path gate, while
hold and turns remain feasible. Right-arc minimum observed nominal clearance is
0.4487909m,0.4479375m and0.4486197m at those samples, against the unchanged0.45m
radius. Surface checks pass for all six actions at each sampled frame.

The reported observed position stays near[0.932,0.154]m and goal distance near
3.846m from1000through2990. The planner continues to report an observed-floor
frontier route and a waypoint about0.36m ahead. The existing correction fallback
only runs when the original action is None; a feasible hold therefore suppresses
any reconsideration of corrected translation feasibility. The late snapshots
show geometric vetoes in addition to the contact-cost effect at503.

Next hypothesis: test a separately named hold-feasibility successor that retains
the original hold unless a candidate with strictly higher original utility passes
the existing corrected-first-interval geometry, all eight nominal segments,
original/corrected surface checks and original phase restrictions. Preserve raw
forecasts, residual targets, later points, yaw/contact predictions, model weights
and original no-action fallback. Do not force a translation, lower clearance or
contact thresholds, add an arbitrary hold timeout, or claim that such a candidate
is currently feasible. Implement and test the hypothesis, then replay the full
closed original prefix to the first changed command only after its raw audit is
admitted. Fresh physical continuation remains necessary.

Diagnosis sources:

- `scripts/diagnose_go2_residual_maze02_stagnation_v1.py`:
  916ed7461a23ad4eadb8ec8e92822229729fa3dbd0eee6f0e02a40a7766bdb63
- `docs/go2_residual_maze02_stagnation_v1_2026-09-09.md`:
  27ce38e26a78ea6e63be046fd7cbf1cb3ea7e572b3b0d7f50398aa944dfeca68

Latest measured artifact free space78,166,843,392bytes is below the supervised
three-case launch allowance78,383,153,152bytes. The earlier supervised preflight
was valid at its measurement time. Refresh and resolve capacity before launching
that cohort; do not bypass its frozen resource gate. The immediate tracking
pilot's smaller admission still fits. No deletion or storage mutation was done.

## Raw audit completed; final native admission pending

The native worker has now written its completed raw audit:
5928241136736e148bf64cdd91294774da1255074b6c10aea80309037f1c4fd8.
Raw primary/auxiliary sensor reconstruction, full model/command replay,
actual-command audit and unchanged model state all pass. Strict physical
visibility passes with zero hard measurement failures. No observed/native
arrival, no physical stop and no verified round trip. The physical evaluator
records only the open crossing from cell[-1,0]to[0,0], at sample9966, with no
invalid crossing. No return traversal; terminal native quiet does not pass.

The audited selected-action counts reproduce2643holds,147left turns,91right
turns,75left arcs and44right arcs. This supports the stagnation finding, while
the fresh physical-prefix comparison and final source/artifact/result admission
remain pending. Do not run the hold prefix or paired readout until the final
native result and its exact SHA-256 are available and admitted by their runners.

Physical-prefix comparison subsequently completed with SHA-256
c65b1d2ea83cd87b013054c8304341a3209b3129c0f7f35e760ddc8a6dc57321:
464 common observations and the complete pre-command physical/public prefix
match; all candidate decisions equal the prospective replay. The root's final
source/artifact checks and terminal result remain pending at this entry.

Final native result subsequently completed,78591exit0:
55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466.
Full result and scope:docs/go2_residual_first_interval_maze_pilot_result_2026-09-09.md.
The closed stagnation findings are now backed by completed raw audit, physical
prefix and final source/artifact verification. No arrival or round trip.
