# Next execution stage: uncertain online memory, observed beacons and actual return

Whole-task integration update: the [corrected four-trial pilot](go2_whole_task_navigation_sampling_correction_development_v1_result_2026-09-05.md)
is COMPLETE with full400-decision audit PASS, but0/4 task success. Every trial
fails the first local visual-change arrival predicate; no marker discovery or
return occurs, and the memory contrast is not exercised. No contacts or sensor
faults occur. The original pre-navigation infrastructure failure is preserved.
Full1,026 tests across89 files pass; all runs are terminal and source-bound.
The next actionable stage is [observed geometry feeding this continuous mission](go2_observed_geometry_whole_task_next_steps_2026-09-05.md),
not another isolated memory probe or a lowered novelty threshold.

Physical-marker update: [six-case actual RGB acquisition](go2_marker_beacon_development_v1_result_2026-09-05.md)
and corrected full audit are complete. The positive case registers the declared
pattern once; absent, occluded and three distractor cases register none. This
provides stage 2's initial interface, not discovery during exploration or robust
semantic recognition. Next implement stage 3's continuous controller with the
existing episodic memory and this observer. No return has been physically executed.

Implementation update: [episodic route memory and event replay](go2_episodic_route_memory_development_v1_result_2026-09-05.md)
complete the first software foundation: provisional visit/attempt history,
acquired multiview context, ambiguous retrieval and fresh return-candidate gating.
Thirty new tests and four recorded-trajectory interface replays pass; no physical
return or beacon detection has been executed. Next prioritize stages 2 and 3
below, using this component in a continuous live development episode. Actual
place association, return reliability and memory benefit remain unverified.

The task-acquisition factorial is complete with full audit. Useful local operators
exist, but every arm still succeeds2/4: one remaining fixture has a false arrival
candidate and another has insufficient turning clearance. Preserve those failures.
The next stage must move beyond another isolated two-leg or threshold study.

The full objective remains reliable RGB-plus-deployment-valid-sensor JEPA maze
exploration, initially hidden beacon discovery and return through online memory,
with matched scientific comparisons, independent layouts and real-platform evidence.
This document specifies the next implementation work, not a completed milestone
or authority to use protected benchmarks, hardware or old frozen output roots.

## 1. Implement a usable hypothesis memory without inventing place identities

The existing ObservedExploration component requires externally associated place
identities and qualified arrivals. It cannot be connected by simply casting the
current ExitCandidate/ARRIVAL_CANDIDATE objects into those stronger types.

Add a separate episodic hypothesis memory. Allocate observation/visit-event IDs
online, bind them to actual image/body timestamps and descriptors, and retain
the observed departure bearing, attempted movement and provisional terminal
evidence. These IDs identify records, not true physical cells or recognized
places. Preserve alternative association candidates and explicit unknown states.
No simulator pose, maze cell, beacon coordinate, map or teacher target enters it.

First support an outward visit history and candidate return-to-predecessor intents.
The opposite of an outward bearing may suggest what to observe next, but is not
an automatically verified reverse edge. Require a fresh observed exit before
executing a return attempt and keep actual return evidence distinct from intent.
Loop/revisit association may remain uncertain and may guide bounded development;
never promote a highest-scoring hypothesis into ground truth merely to enable
routing. Record ambiguous associations, failures, contradictions and recovery.

Tests must distinguish event identity from place identity, reject stale/rewritten
observations, preserve failed attempts, avoid implicit reverse-edge qualification,
and represent repeated-looking places without unconditional image-similarity
merges. Evaluate any descriptor thresholds on declared development observations;
do not call uncalibrated scores probabilities or claim final place recognition.

## 2. Detect an actually rendered beacon from current observations

Add a physical visible marker to fresh development scenes, initially occluded
from the starting camera. Use its current RGB appearance, or another explicitly
declared sensor, to detect identity. A declared simple marker detector is an
integration baseline, not an appearance-general semantic detector. Never count
simulator proximity, a scene-graph beacon ID, projected known coordinates or a
teacher's visibility flag as a runtime detection.

Validate actual captured positive, absent, occluded and distractor observations,
including timestamp/availability and duplicate detections. The simulator may
verify visibility/proximity only in evaluation. Preserve beacon collision geometry
and camera calibration; do not let adding a marker silently remove obstacles.

## 3. Run a genuinely whole-task development episode

Build fresh small connected mazes with a choice of branches, an initially hidden
physical beacon and a return requirement. Fix layouts, initial conditions, policy
variants, budgets and metrics before execution. A separately declared room width
or initialization domain can isolate integration work, but is development-only
and does not qualify the failed1.2 m domain or replace later independent tests.

The runtime sequence must remain continuous: observe, choose an observed frontier,
execute bounded local commands, update visit hypotheses, detect the beacon, and
attempt remembered return. No mid-episode reset, oracle route, injected cell
association or goal image from a future destination. A starting observation may
serve as the home reference because it was actually acquired at the start.

Compare the same executor and sensor configuration with persistent route memory
versus a declared local-only baseline. Keep the minimum task state/home reference
available to both and state exactly which history the memory ablation removes.
Do not claim a JEPA contribution from this memory comparison. The direct,
supervised-recurrent and JEPA controllers must eventually share the same memory,
available observations, candidate actions and physical safeguards.

Primary task outcomes must use actual detected beacon evidence and evaluation-only
physical return, contacts/interventions, path length, time and stalls. Report
false beacon claims, false home claims, false/missed local arrivals, association
errors and unfinished returns. Controller MISSION_COMPLETE alone is insufficient.
Local provisional visits need not become certified before testing this uncertain
prototype, but none may be reported as certified afterwards without evidence.

## 4. Address the measured observation limits alongside integration

The present body-span-plus-command-distance rule lacks an observed opening
position. The floor-extension ray is not a portal or clearance measurement, and
a successful crossing does not guarantee a useful turning pose. Develop observed
opening/near-field geometry with explicit uncertainty and measured motion, not
another fixed extra-distance margin fitted to the failed corner.

An additional RGB/depth/range configuration is legitimate only as a separately
declared, deployment-realizable sensor arm. Bind actual sensor frame/calibration,
measurement/availability times, validity/noise/missingness and policy-only I/O;
do not rebrand analytic simulator wall access as a sensor. Extra sensing is not
a JEPA improvement and must be compared at its own sensing/compute cost. Actual
hardware availability and calibration remain unverified until measured.

Keep the existing narrow-maze collision and premature-arrival traces as regression
evidence. A new wider-domain episode may demonstrate memory/beacon integration,
not solve those failures. Do not repeatedly postpone the full-task prototype
until every local observation is perfectly qualified, and do not erase uncertainty
to make the prototype appear complete.

## 5. Preserve the final scientific comparisons

After task integration, train/evaluate supported scan, alignment, repositioning
and traversal command families with matched data/optimization budgets. Separate
predictive training from online rollout, memory, extra sensors and action coverage.
Current frozen heads score one transition and do not establish multi-step planning.
Require actual task outcomes against strong non-learned and supervised controls;
a scientifically credible negative JEPA result is valid, but not a substitute
for the requested navigation system.

Independent novel layouts, appearances/dynamics, training-seed variation, maze-level
uncertainty, compute/latency and bounded supervised real-Go2 evaluation remain
required. Protected legacy benchmarks remain inaccessible. Do not mark the
overall goal complete on unit tests, local panels or a single whole-task demo.
