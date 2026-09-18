# Inactive routing-memory comparison preparation

The current eight-run new-maze learned/reactive study remains unchanged. This
separate preparation addresses the outstanding causal role of spatial routing
memory; it is not selected by any native launcher.

`lewm/current_pair_routing_memory_development.py` wraps the existing map's set
updates. It captures current floor, coarse obstacles and fine obstacles from the
same paired geometry computation, including reobserved cells already present in
history. Fine-obstacle updates from both cameras are unioned. The accumulated
map remains available and unchanged; immutable current sets accompany each
snapshot through the existing process boundary.

Two future arms can use this same capture implementation. `persistent` uses the
accumulated routing snapshot. `latest_mapped_pair` uses only the latest completed
mapping observation for route construction, fine goal fallback, route preference,
frontier-view interpretation and waypoint lookahead. It does not pretend that the
latest mapped pair always equals the planning camera frame. Existing freshness
checks and clocks remain in force. Missing captured evidence cannot silently
fall back to accumulated routing cells.

Both arms retain the full accumulated map for action prediction/clearance,
current-body obstacle dispatch, visual/gyro tracking and floor anchors, model
temporal history, residual history, mission/settling, scan state and frontier
visit/exclusion state. Thus this is a routing-map memory ablation, not a fully
memoryless controller or an ablation of safety-related obstacle history. Planning
records identify the scope and current/retained cell counts.

Five focused tests passed in 2.14 seconds: capture of reobserved cells and both
camera updates, clearing only the current observation, immutable view/process
serialization, persistent/current routing and lookahead with unchanged action
clearance input, and rejection of unavailable current evidence.

Seven actual saved camera pairs from new-maze learned layout 0 (frames 0, 4,
100, 104, 1000, 1004 and 2900) were then used for a map-operation comparison.
Every original accumulated snapshot field matched exactly. Each captured current
floor/coarse/fine set matched independent recomputation from that single pair.
The harness supplied saved observation-derived registered poses through a
comparison-only pose-reader binding; it did not reconstruct the registration
evidence envelope or load native physics. This is sparse implementation evidence,
not an online timing qualification, complete mapping replay or navigation result.
Result: `go2_post_repeatability_transfer_learned_native_layout00_4800_v1_attempt_001/current_pair_map_capture_equivalence_v1.json`.

After completing all eight current study assignments, specify the memory study's
fixed layouts, controller and dispatch order. Both arms require new native trials
with the captured map, so timing overhead is represented in both. No memory
benefit, reliable navigation or hardware capability is established by this source
preparation.

Prepared `scripts/run_go2_routing_memory_scope_development.py` with explicit
`--scope persistent|latest_mapped_pair` and `--layout-index 0|1|2|3`. Both arms
use the same captured-map initializer and the unchanged learned controller,
with routing scope as the sole treatment. No native memory trial has launched.
All eight writer configurations were exercised from the original base launch
fields without scene construction or model loading. Original learned launch
fields and source identities matched before adding memory-treatment metadata
and its two source files. Runtime method resolution reaches the scope wrapper
for routing, waypoint lookahead and action-selection evidence.

The stronger reactive comparison runs first. Specify the memory trial order
before its first dispatch and compare both newly executed arms; the previous
uncaptured-map successes are context, not a runtime-identical control.

## Fixed eight-run routing-memory experiment

Fix the roster now, while the first stronger-reactive pair is still running:
all four existing post-repeatability layouts, the unchanged learned
`HeadingRecoveryRuntime`, and both prepared routing scopes. After all four
stronger-reactive runs finish, dispatch persistent layouts 0/1, latest-pair
layouts 0/1, latest-pair layouts 2/3, then persistent layouts 2/3. Wait for both
owner exits and archives between pairs. Keep the same source, seeds, camera
configuration, mapping capture, CPU groups, 4,800-tick budget and arrival
criteria in all eight runs. Finish the entire roster without tuning or outcome
substitution. These are development revisits and do not add independent mazes.

Primary outcomes are independently verified goal and round-trip arrivals and
disallowed contacts. Inspect trajectories and the recorded routing cell counts
to establish that the treatment actually changed available routing evidence.
Preserve timing, failures and other retained-state limitations. Interpret the
result only as a test of accumulated spatial evidence for routing; both arms
retain accumulated obstacle evidence for predictive clearance and the other
state explicitly listed above. It cannot establish a fully memoryless baseline
or isolate the learned model's internal temporal memory. Eligible depth
retirement can precede later batches under the existing user policy.

The stronger reactive study is complete: 3/4 goals and 0/4 round trips, with
all failures retained. Before the first memory batch, both native owners and
the completed diagnostic replay are gone. Available RAM is about 76 GiB and
RecoveryStorage has about 32 GiB free after reviewed old-depth retirement.
The 32 logical CPUs remain available; both GPUs reported 0% compute use, with
about 1.84 GB used of 34.21 GB on the discrete GPU. Retain the established
two-owner CPU execution profile; its full-budget runs used about 24 GiB peak
RSS each, with no process swaps. No GPU execution change is introduced.

The first persistent-memory pair launched around 21:34 local log time:
layout 0 session 57598 / PID 3475570; layout 1 session 39745 / PID 3475590.
Both owners are live and progressed past 200 camera frames. Actual launch
records identify the fixed JEPA assignment, persistent routing scope and
current-pair capture enabled. Results remain pending. The next batch is
latest-mapped-pair layouts 0/1 after both complete archives and owner exits.

## First persistent-memory pair completed

Both owners exited 0, and both round trips passed independent physics checks.
Layout 0: 3,106 pairs, goal/home frames 2048/3104, physical dwell ranges
7.464–9.508 mm and 11.716–17.737 mm, path 23.545 m, maximum pose error
7.795 mm, 761/769 plans on time. Layout 1: 3,525 pairs, frames 2577/3523,
dwell ranges 13.937–16.776 mm and 19.192–24.035 mm, path 27.687 m, maximum
pose error 11.131 mm, 856/874 plans on time. All arrivals met quiet-motion and
zero-request checks; both runs had zero disallowed contacts.

Every selected plan recorded persistent routing scope and equality between
routing and retained floor/fine-obstacle cell counts. Evaluations, summaries,
scope diagnostics and resource records are saved in each root. Owner times
were 442.13/520.99 seconds; maximum RSS 17,193,608/19,264,208 KiB, zero swaps.
These are the fresh baseline outcomes for the paired memory experiment, not
evidence of memory causality before the other arm executes.

A separate reference-refresh tracking replay started on layout 0's released
CPU group after that native owner exited. It changes no memory-study source;
finish this bounded replay before the next paired native dispatch.

The separate tracking replay finished and all native owners were absent, with
about 28 GiB disk and 76 GiB RAM available. The fixed latest-mapped-pair
layout-0/1 batch launched around 21:45 local log time: session 65946 /
PID 3479355 and session 58415 / PID 3479390. Actual launch records identify
the reduced routing scope and both owners are live. These runs retain the
original tracker and accumulated action-clearance evidence. Next are
latest-mapped-pair layouts 2/3, then persistent layouts 2/3, unchanged.

## First reduced-routing pair completed

Both owners have exited and their sensor archives are complete. Layout 0 exited
0 after exhausting the 4,800-tick mission budget: 4,805 pairs, no arrivals,
zero disallowed contacts, maximum position error 8.690 mm, and 1,188/1,200
plans on time. Owner elapsed time was 666.51 seconds, maximum RSS 25,324,128
KiB, zero swaps. Layout 1 exited 1 after 984 pairs with a physical guard stop
and no arrivals. The final measured speed was 0.301485 m/s against the fixed
0.3 m/s guard; this was the only sample exceeding that threshold after camera
0. It remained inside the XY domain and recorded zero disallowed contacts.
Separate nonfoot-ground guard rows were not saved, so the speed condition is
confirmed without claiming to have excluded every other guard condition.
Maximum position error was 6.530 mm; owner elapsed time 161.74 seconds,
maximum RSS 7,249,736 KiB, zero swaps. Both failures remain in the population.

The two matched launch pairs differ only in owner and the two explicit routing
scope labels; all other recorded settings and source identities match. Reduced
routing contained fewer floor and fine-obstacle cells on 1,199/1,200 layout-0
plans and 244/245 layout-1 plans. Accumulated action-clearance evidence remains
present. Independent evaluations, summaries, resource records, paired comparison
JSON and trajectory figures are saved; both figures were visually inspected.
Comparison roots are `go2_routing_memory_comparison_layout00_v1_attempt_001`
and `go2_routing_memory_comparison_layout01_v1_attempt_001`.

Before the next batch, both prior native owners were absent, CPU was 1.8% busy,
about 76 GiB RAM and 24 GiB artifact-volume space were available, and both GPUs
were idle. Keep the same two CPU groups and capture mode. Latest-mapped-pair
layouts 2/3 launched around 21:57 local log time, sessions 69568/22504 and
PIDs 3483227/3483246. Persistent layouts 2/3 follow after both owners finish
and archive. No controller, tracker, seed, threshold or roster change was made.

The reduced-routing layout-0 stall has public sensor and dispatch evidence:
primary depth had 283,176 valid pixels at frame 300, but zero at sampled frames
400, 2000 and 4800. Auxiliary depth retained 157,030/148,929/133,373 valid
pixels at those latter frames. In the final 180 seconds all 451 selected plans
requested a left turn for an additional view (446 on time), while all 9,022
actual dispatch requests were zero: 8,476 latched vetoes, 446 unavailable
observations and 100 missing-on-time-plan intervals. All 446 unavailable
requests had no obstacle observation rather than merely an aged one. This
identifies the immediate failure to execute recovery; it does not prove how
a different routing history would have prevented entry into that state.
The two diagnostic JSON files are retained with the full sensor recording.

Both latest-pair layouts 2/3 subsequently passed camera frame 800 with live
owners and the assigned scope/model recorded in their launch files. Neither
had an arrival at that observation point; their outcomes remain pending.

## Second reduced-routing pair: simulation phase complete

Layouts 2/3 both exhausted the 4,800-tick mission budget with 4,805 pairs,
no observed arrival, no recorded disallowed contact and no pipeline fault.
At this observation point the owners are still archiving, so independent
physical evaluation and owner resource results remain pending.

The saved planning evidence confirms smaller routing floor/fine-obstacle sets
on 771/772 selected layout-2 plans and 1,073/1,074 layout-3 plans, retaining
accumulated action clearance. Layout 2 recorded 428 `VIEW_BUDGET_EXHAUSTED`
planning entries and layout 3 recorded 126. These entries return before action
selection in the existing planner, explaining much of the dispatch-level
`NO_ON_TIME_PLAN` count; that count must not be presented simply as slow
computation. Actual selected plans were on time in 764/772 and 1,050/1,074
cases. Ten/twelve frontier-view events completed, respectively. Routing-scope
diagnostics are saved in both roots. Neither controller settings nor view
budgets were changed in response to these outcomes.

Both archives completed and both owners subsequently exited 0. Independent
evaluation confirms no arrivals, zero disallowed contacts and all 4,805
registered poses per run. Layout-2 median/maximum position errors were
4.995/10.588 mm, path 9.168 m and closest goal distance 2.590 m. Layout-3
errors were 7.977/16.017 mm, path 14.466 m and closest goal distance 1.278 m.
Owner elapsed times were 671.44/711.24 seconds, maximum RSS
25,154,432/25,314,724 KiB, zero swaps. Evaluations, summaries and resource
records are saved. The reduced-routing arm is complete: 0/4 goals or round
trips, with three budget exhaustion outcomes and one physical guard stop.

Before the final persistent pair, both preceding owners were absent, CPU was
2.9% busy, about 76 GiB RAM and 18 GiB artifact-volume space were available,
and both GPUs were idle. Persistent layouts 2/3 launched in sessions
40803/37814 using the original CPU groups, source and settings. These are
the last two assignments of the fixed memory roster. No tracking refresh
trial has launched.

## Completed eight-run result

All eight owners have exited and archives/evaluations are complete. Persistent
routing achieved 4/4 verified goals and 4/4 round trips; latest-pair routing
achieved 0/4 of either. All eight recorded zero disallowed contacts. The reduced
arm's physical-speed stop remains a failure in the population.

Final persistent layout 2: 3,105 pairs, verified goal/home frames 1956/3103,
physical dwell distances 16.174–23.103 mm / 20.148–21.363 mm, maximum
position error 8.867 mm, path 23.458 m, 751/767 selected plans on time.
Layout 3: 3,027 pairs, frames 1968/3024, dwell distances 9.571–11.188 mm /
13.615–14.050 mm, maximum position error 13.673 mm, path 21.134 m,
725/749 selected plans on time. All four arrivals passed quiet-motion and
zero-request checks. Both owners exited 0 after 441.71/452.08 seconds,
maximum RSS 17,177,968/16,909,852 KiB, zero swaps.

Every paired launch differs only in owner and routing-scope labels; all other
recorded settings and source identities match. All four paired trajectory
figures were visually inspected. Combined results and individual failure,
resource and routing-scope evidence are saved in
`go2_routing_memory_four_layout_summary_v1_attempt_001/result.json`.

This intervention supports the value of accumulated spatial routing evidence
for this controller on the four development revisits. Both conditions retain
accumulated action-clearance obstacles and other controller state. It does not
establish a fully memoryless comparison, learned internal-memory contribution,
JEPA-specific benefit, broad reliability or hardware readiness. Since reduced
routing never reached a goal, it also does not isolate memory needed specifically
for the return journey. The prepared tracking-refresh experiment can now run
without changing any completed memory assignment.
