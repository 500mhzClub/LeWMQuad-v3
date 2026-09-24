# Matched surface-memory native result — 2026-09-08

Persistent surface evidence changed an actual controller decision under identical
sensor history and model predictions, but the four-case experiment reached zero
goals. On the second layout it replaced the current-frame variant's eventual
contact with an earlier terminal stop, 0.958 m from the goal. This is insufficient
navigation: the controller still cannot find or execute an alternative route.

The [prospective protocol](go2_surface_memory_goal_probe_v1_2026-09-08.md) fixed
the final full-JEPA model, original six actions, 1.2 m goal, five-tick commitment,
240-tick navigation budget, corner observer and native guard/audit rules. Both
variants updated the same surface memory. At each replanning decision they
filtered candidates with possible intersections between the predicted half-second
robot footprint and either current-frame or persistent surface voxels, then kept
the highest original learned utility. No-hit queries remained unknown. All-blocked
decisions latched a terminal stop and ten zero-command intervals. No source, model,
cost or threshold changed during this attempt.

| Layout / surface query | Outcome | Terminal goal distance | Minimum goal distance | Model selections |
|---|---|---:|---:|---:|
| 052 / current frame | Visual-pose failure | 0.975310 m | 0.972100 m | 6 |
| 052 / persistent | Visual-pose failure | 0.975310 m | 0.972100 m | 6 |
| 039 / persistent | All candidates have possible intersections | 0.957754 m | 0.957748 m | 6 |
| 039 / current frame | Disallowed panel contact | 0.892337 m | 0.890668 m | 9 |

The fixed execution order was current 052, persistent 052, persistent 039,
current 039. Every case used a fresh process, native scene, model load, observer
and memory. All four completed raw sensor reconstruction and exact observer/
memory/model/command replay. The model state stayed unchanged. There were 163
frames, 10,990 physics samples and 27 model selections. Native execution plus
auditing took 399.946 s. No failed case was retried or resumed.

On layout 052 the variants made the same commands: persistent memory filtered
four candidate instances, but never the selected action. Both lost visual pose
at the original point and recorded the frame-36 near-occlusion depth failure in
the terminal drain. The hard measurement failure remains in both results. Neither
had physical contact; both completed all ten zero intervals.

On layout 039 the first command and selection divergence was tick 28. The paired
readout verified exact equality of every complete public packet and observer
evidence record through that tick, plus identical six-candidate model predictions
and original utilities. Current-frame queries found no possible intersections
and selected `right_turn`. Persistent queries found possible head intersections
for hold, forward, left arc and both turns; right arc intersected the front-left
hip. Forward also intersected that hip. These witnesses came from earlier depth
frames, not native scene geometry or future contact labels. All six candidates
were filtered, so the persistent controller stopped and completed ten zero
intervals. Its terminal distance was 65.417 mm farther from the goal than the
current-frame variant's contact termination. Both layout-039 cases passed strict
sampled visibility and hard depth-measurement checks.

Persistent filtering changed one of its twelve model decisions to no action;
current-frame filtering changed none of its fifteen decisions. Persistent variants
filtered four and eight candidate instances on layouts 052 and 039 respectively;
current variants filtered zero. Fewer executed commands and earlier stopping mean
that contact-free termination does not establish improved full-mission safety or
goal-reaching. This is one known mirrored layout pair and one optimization seed,
not an independent-maze or statistical-benefit result.

| Case | Complete iteration median / maximum (ms) | Iterations over 100 ms |
|---|---:|---:|
| Current 052 | 222.819 / 272.749 | 38 / 39 |
| Persistent 052 | 224.968 / 272.870 | 38 / 39 |
| Persistent 039 | 211.711 / 263.488 | 37 / 38 |
| Current 039 | 235.048 / 471.592 | 43 / 44 |

These include warmup and applicable drain intervals. Physics was paused during
computation. Added memory processing did not solve the original latency problem.
Before launch the machine had 16 physical / 32 logical CPUs, 0.3% CPU use,
82,575,474,688 bytes available RAM, idle GPUs and 96,316,747,776 artifact bytes
free; no substantial competing Python job was present. Runs were serial to avoid
contention in complete-loop timing. The launch repeated resource/reserve checks,
and resource monitoring continued throughout all four cases.

Five focused controller tests passed in 1.68 s. They covered unchanged utility/
prediction ranking after exclusion, all-blocked zero output, unmodified ranking
without intersections, undefined-yaw rejection, sensor-failure latching and the
unchanged external native-goal/actuator audits. Complete raw native replay then
tested the actual model and sensor-to-command implementation on every case.

Artifacts are under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`.
Native root: `go2_surface_memory_goal_probe_v1_attempt_001` (902 frozen sources;
782 bound artifacts totaling 265,549,328 bytes). Paired-readout root:
`go2_surface_memory_goal_readout_v1_attempt_001` (904 frozen sources).

| Identity | SHA-256 |
|---|---|
| Native launch | `055f7eafac1e275bc579d7e98ff830da2ada750b20e9b639b3385177d790ecb9` |
| Native result | `730e18a0345b8cd49b432ad5c53ed08cb7e80434e15a7ddfdacc454524afc74e` |
| Paired-readout launch | `5648869434ec5d96862e6d305620614dd1f8a8cb5b393193b4f5a5b8b061d2bf` |
| Paired-readout result | `bc9c54e43774d8083e39da0a80b558fd145efdb06f231a65cc5d1ab1dad142f2` |
| Final full-JEPA snapshot | `bcb8874e2adf89053463206267a4ccb90380909c324e303734a59b038f5b1821` |

All bound source, native/input, robot-URDF and artifact identities were verified
before and after native execution and readout. The
[paired-readout protocol](go2_surface_memory_goal_readout_v1_2026-09-08.md) retains
the full first-divergence witness. Original attempts remain frozen.

The next change must provide route selection and recovery, not just another
contact-cost or veto threshold. The present action bank has hold, forward, two
forward arcs and two in-place turns; its frozen command validator forbids negative
forward velocity. It therefore supplies no backward candidate when every local
forecast intersects a remembered wall. Reverse motion would require a separately
declared command domain, physical response/guard verification and compatible
transition data; it cannot be inserted as an allegedly validated old action.
Observed floor/free-space coverage and waypoint selection should guide an earlier
detour, with physical backtracking toward retained poses when exploration fails.
The current history is only a route proposal. Acquisition profiling, the recurring
near-depth failure and visual support loss also remain unresolved. Preserve this
matched failure while developing those capabilities for the full active objective.
