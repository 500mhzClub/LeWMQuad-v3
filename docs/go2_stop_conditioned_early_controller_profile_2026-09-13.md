# Current controller: early-decision costs

The current stop-conditioned controller replayed actual frames 0–12 of its
freshly collected maze02 trial. All 13 complete decisions matched the recorded
decisions exactly. Frames 0–2 supplied the original warmup; only frames 3–12
were profiled. The unchanged assigned model was
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.
The diagnostic completed in session 81998 without creating a native scene.

The ten profiled calls totalled 7.287 seconds. Selected inclusive costs were:

| Existing operation | Mean profiled ms per decision |
| --- | ---: |
| Map update (`LaterResolvedFloorMap.observe`) | 263.319 |
| Visual tracking (`DualCameraVisualMotion.observe`) | 159.667 |
| Measured floor candidate extraction, four calls | 124.945 |
| New/updated measured sample bounds, eight insertions | 67.867 |
| Assigned model forward | 7.613 |

These are inclusive function costs and overlap. In particular, candidate
extraction contributes to tracking and floor registration; the table must not
be summed. Python profiling exaggerates the cost of frequently called Python
helpers. This was a shared-host early-history diagnostic, not an uninstrumented
benchmark, a late-memory profile or a full acquisition/control-loop measurement.

The candidate-extraction callers still build full camera floor-cell grids to
select a smaller stride-four population. Two calls use the original dense grid
inside visual tracking and two use the existing density-routed implementation
inside floor registration. This is a concrete place to investigate reuse or
reduced work. Any successor must retain the actual candidate population and
measurement gates; this profile alone does not prove a replacement equivalent.

Receipt construction and copying also remain visible, but the completed older
deferred-copy experiment saved only 0.158% of total navigation controller time
on its own trajectory. The current profile does not justify repeating that
optimization or estimating its gain from instrumented call counts.

Model inference is a small part of this measured path. Faster inference alone,
or the separately measured 7–15 ms saving in auxiliary packet reads, cannot make
the existing loop meet 100 ms. Perception and map processing deserve the next
latency work. No running or queued controller has been modified.

Raw profile: `go2_stop_conditioned_early_controller_profile_2026-09-13.prof`.
Structured results, selected input identities, per-frame equality records and
top function costs: `go2_stop_conditioned_early_controller_profile_2026-09-13.json`.
Source: `scripts/profile_stop_conditioned_early_decisions_development.py`.
This diagnostic does not replace the pending complete raw audit or add an
independent-maze, continuous-execution or hardware result.
