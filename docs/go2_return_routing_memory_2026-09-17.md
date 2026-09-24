# Return-leg routing-memory comparison

**Complete: persistent routing returned home on both new mazes; latest-pair
return routing reached both goals but returned home on neither. Zero contacts
and zero tracking/pipeline failures in all four runs.** The return-only scope
was verified from actual selected plans, and no local turn-memory selections
occurred in any arm. All models and frozen source assignments matched.

| Maze (figure label) | Return routing | Goal verified | Round trip | Total simulated seconds | Goal-to-home interval |
| --- | --- | --- | --- | ---: | --- |
| 1 | Persistent | Yes | Yes | 224.52 | 58.40 s |
| 1 | Latest mapped pair | Yes | No: budget exhausted | 481.02 | No home after 319.90 s |
| 2 | Persistent | Yes | Yes | 154.74 | 53.50 s |
| 2 | Latest mapped pair | Yes | No: budget exhausted | 481.00 | No home after 367.90 s |

Goal-to-home intervals use recorded mission sensor timestamps. Failure intervals
end at the final mission observation and are censored, not completion times.
Figure labels 1/2 correspond to inventory indices 0/1 used in the run records.
This supports accumulated routing geometry for returning through previously
observed corridors in this controller. It does not establish JEPA superiority,
learned internal-memory benefit, fully memoryless navigation, broad reliability
or deployment readiness. Outbound paths and asynchronous timing differed.

Complete result: `docs/go2_return_routing_memory_result_2026-09-17.json`.
Physical trajectory figure: `docs/go2_return_routing_memory_trajectories_2026-09-17.png`
(also SVG). The four panels were visually inspected; persistent returns retrace
observed corridors, while reduced-memory returns enter different corridors and
stop short of home. Reader: `scripts/read_go2_return_routing_memory_development.py`.
All four full sensor recordings remain retained. All jobs and evaluations ended.

The earlier eight-run routing-memory study is complete: persistent routing
achieved four verified round trips, latest-mapped-pair routing achieved none,
on four development revisits with an older controller. Its result is at
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_routing_memory_four_layout_summary_v1_attempt_001/result.json`.
That supports accumulated spatial evidence for exploration in that controller,
but cannot isolate the return leg because the reduced arm never reached a goal.
The recent local-turn-memory ablation addresses a different mechanism.

This fixed four-run experiment gives both arms persistent outbound routing and
the same frozen JEPA model, completed sparse-corner tracker, polygon map,
six candidate primitives and selected-turn-memory correction. One arm retains
persistent routing on return; the other uses only the latest completed mapped
camera pair for return route finding. Both retain accumulated obstacle evidence
for predictive clearance, current obstacle dispatch, tracking references,
model history, local turn memory, mission state and frontier-visit state.
This is neither a fully memoryless controller nor a learned internal-memory test.

The existing per-plan mission generation fixes the scope throughout each plan;
existing command checks discard plans if the mission changes before dispatch.
Two focused tests passed, including the concurrent mission-publication case
and preservation of the original accumulated snapshot. Every selected plan
records scope, generation and current/retained cell counts for evaluation.

Two new same-family geometries were selected before execution, excluding the
explicit 103-layout registry by the existing topology/embedding criteria.
Construction seed 2026091781 accepted its first two candidates into distinct
new topology/embedding groups. No runtime outcomes informed selection.

Fixed order:

| Assignment | Layout | Return routing |
| --- | --- | --- |
| 1 | 0 | Persistent |
| 2 | 0 | Latest mapped pair |
| 3 | 1 | Latest mapped pair |
| 4 | 1 | Persistent |

Plan: `docs/go2_return_routing_memory_plan_2026-09-17.json`.
Inventory: `docs/go2_return_routing_memory_layout_inventory_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_return_routing_memory_development.py`.
All native jobs run sequentially on CPU 8–15/24–31, with the established
4800-tick budget, 300-ms planning deadline, 20-ms extra wait, ideal gyro and
2-mm synthetic depth noise. Physics pauses during computation. No new training,
hardware execution or real-time qualification is included.

Before the first launch, CPU was 99% idle, about 71 GiB RAM was available and
both GPUs reported 0% use. After reviewed depth retirement 13.18 GiB was free
on the artifact volume. Independent temporal tracking and current native timing
make this a sequential run schedule; no heavy analysis runs during navigation.
Evaluate each completed run before the next. Keep each new pair full through
paired analysis, every new failure and unresolved input; retire eligible
completed success depth under the existing policy if recording space is needed.

Report all assigned outcomes. A run that never reaches the goal does not
exercise the return-only intervention. Compare goal/home physical dwell,
contacts, physical backtracking, treatment activation and timing. Differing
asynchronous outbound trajectories limit causal interpretation; one execution
per condition per layout cannot establish broad reliability or JEPA superiority.

## Assignment 1 complete: persistent return, layout 0

Native session 86983 (owner PID 4184487) and evaluator session 68785 exited
zero. Independent checks verify a **224.52-second simulated round trip**,
zero contacts and zero pipeline faults. Both quiet-arrival checks passed at
goal/home frames 1656/2240; maximum physical dwell distances were 16.50/3.45 mm.
All seven unique return edges reversed outbound edges (nine outbound), with
zero invalid graph transitions. Tracking covered 2243 camera pairs.
517 of 542 plans were on time (95.39%); live navigation took 296.07 wall seconds.
All 400 outbound and 142 return selected plans used persistent routing, as
assigned. The paired latest-pair return run is next. No return-memory benefit
can yet be inferred from this control alone.

## Assignment 2 complete: latest-pair return, layout 0

Native session 27451 (owner PID 4185337) and evaluator session 17160 exited
zero. The run **reached the goal but exhausted its budget without returning
home**, at 481.02 simulated seconds. Independent physical evaluation passed
the goal's quiet dwell at frame 1604, maximum distance 12.33 mm. There were
zero contacts, zero pipeline faults and 4805 tracked camera pairs.
1140 of 1196 plans were on time (95.32%); live navigation took 619.94 wall seconds.

Scope verification passed on every plan: 397 outbound plans retained persistent
routing; all 799 return plans used reduced floor and fine-obstacle sets from
the latest mapped pair. Accumulated predictive-clearance evidence remained.
No local turn-memory selection occurred. Of six unique return edges, only
three reversed outbound edges (seven outbound); graph transitions remained
valid. In the last 100 simulated seconds, 248 of 251 plans selected hold,
all with an observed-floor route to a frontier. The failure is preserved in full.

The first pair supports the value of accumulated routing geometry for the
return leg in this controller. Both arms physically reached the goal, so the
intervention was exercised rather than prevented by outbound failure. Timing
and outbound paths differed; this single pair is not a broad causal or
reliability claim. The reversed-order pair on layout 1 remains to be completed.

## Assignment 3 complete: latest-pair return, layout 1

Native session 80119 (owner PID 4186775) and evaluator session 32712 exited
zero. Independent evaluation verifies goal arrival at frame 1124, maximum
physical quiet-dwell distance 22.17 mm, but **no return home before budget
exhaustion at 481.00 simulated seconds**. Zero contacts and pipeline faults;
4805 tracked camera pairs. 409 of 438 selected plans were on time (93.38%).
Live navigation took 625.58 wall seconds.

All 262 outbound selected plans used persistent routing. All 176 selected
return plans used reduced floor and fine-obstacle sets, preserving accumulated
predictive clearance. No local turn-memory selection occurred. Two of four
unique return edges reversed outbound edges (six outbound), with zero invalid
graph transitions. The planner recorded 743 `VIEW_BUDGET_EXHAUSTED` entries;
these return without selecting an action. They explain much of the dispatch
`NO_ON_TIME_PLAN` count and must not be described as computation deadline misses.
The failed return and its sensor recording remain full. Assignment 4 is the
last frozen run: the persistent control on this same second maze.

## Assignment 4 complete: persistent return, layout 1

Native session 80925 (owner PID 4188246) and evaluator session 52069 exited
zero. Independent evaluation verifies a **154.74-second simulated round trip**,
zero contacts and zero pipeline faults. Goal/home quiet arrivals passed at
frames 1005/1540, maximum physical dwell distances 10.21/17.59 mm. Tracking
covered 1543 camera pairs. All six return edges reversed outbound edges, with
zero invalid graph transitions. 355 of 379 plans were on time (93.67%);
live navigation took 201.61 wall seconds. All 248 outbound and 131 return
selected plans used persistent routing. No local turn-memory selection occurred.

The second pair reproduces the first pair's outcome under reversed execution
order. Both reduced-return runs reached their goals, confirming that their
intervention was exercised. The recent tracking/controller improvements now
have successful JEPA-driven navigation on two additional prospective layouts,
but the source differs from the earlier four-layout comparison and the studies
should retain their separate identities. The next remaining goal questions
concern reliability under realistic sensing/timing and what learned prediction
adds relative to simpler matched controllers, rather than repeating this
completed memory comparison.
