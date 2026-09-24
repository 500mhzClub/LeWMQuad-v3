# Fresh-maze replication: complete development result

All 20 prospectively assigned missions are evaluated: **14 verified round trips,
zero disallowed contacts, six navigation failures**. JEPA completed 2/4 fresh
mazes; each other controller completed 3/4. This experiment does not establish
reliable maze transfer or a JEPA advantage.

Subsequent [recovery-limited survey experiments](go2_recovery_limited_initial_survey_2026-09-16.md)
completed verified round trips with both neural models on the exposed layout-0
failure. Prompt cancellation alone failed. These are separate development
interventions and do not alter the fixed outcomes below.

## Complete matched outcomes

Successful times are simulated seconds, including outward and return travel.
Layout numbers below match the zero-based inventory. Every failure is included.

| Controller | Layout 0 | Layout 1 | Layout 2 | Layout 3 | Round trips |
| --- | --- | --- | --- | --- | --- |
| JEPA | Tracking failure | Budget exhausted, 480.90 s | 174.02 s | 237.20 s | 2/4 |
| Supervised rollout | Tracking failure | 248.02 s | 162.08 s | 230.06 s | 3/4 |
| Pose-command prediction | Tracking failure | 379.36 s | 180.00 s | 194.86 s | 3/4 |
| Instantaneous ranking | Tracking failure | 249.62 s | 165.68 s | 210.76 s | 3/4 |
| Reactive feedback | Tracking failure | 385.28 s | 207.66 s | 204.90 s | 3/4 |

Each successful arrival passed the existing physical-radius and one-second
quiet-motion checks at both the outward goal and home. All 14 successful
returns traversed previously observed outward corridors in reverse, with no
invalid maze-graph transitions. This demonstrates physical backtracking in the
implemented controller; it does not isolate learned memory or prove a causal
memory advantage.

Supervised was faster than JEPA on both jointly successful layouts, by 11.94 s
and 7.14 s, and succeeded on layout 1 where JEPA failed. Supervised was only
1.60 s and 3.60 s faster than instantaneous ranking on layouts 1 and 2, then
19.30 s slower on layout 3. Thus predictive ranking did not improve completion
count and did not consistently improve completion time. Instantaneous ranking
retains predictive clearance guards. Reactive feedback changes the controller
package and disables forecast-based selection; it is not an otherwise identical
one-term ablation. Pose-command was slower than supervised on layouts 1 and 2
but fastest on layout 3. None of these four-layout, single-execution comparisons
supports a statistical algorithm-superiority claim.

## Failures and useful diagnoses

All five arms failed layout 0 during the initial panorama, before any requested
translation. Each completed three views and repeatedly sought the next heading
near 135 degrees while weak-view recovery directed it back to a supported view.
The immediate common limitation is the visual-survey/recovery interaction.
Full failure recordings remain available. No view was marked complete without
observation, and no tracking-loss stop was bypassed.

JEPA alone exhausted the layout-1 budget, with intact tracking, no arrivals,
zero contacts and 1176/1200 plans on time. It spent 965/1200 plans requesting
a frontier standoff view. Post-hoc inspection of frames 600–4299 found 104 new
alternative-turn latches: every preferred turn cleared the nominal footprint
but missed the additional predicted reserve, by a median 2.65 mm. The selected
turn opposed the preferred turn in 455/925 plans. Recorded registered-position
error was small (global median 1.90 mm, maximum 5.43 mm), making a large position
drift an unlikely explanation. These observations narrow the problem toward
prediction/clearance/view-control interaction; they do not establish which
alternative action would have succeeded or justify reducing the reserve.

Reactive feedback had prolonged goal-area turning on layout 1 but eventually
completed. Both neural controllers took longer settling at the outward goal on
layout 3. JEPA's post-hoc goal interval contained 61 short forward pulses and
23 holds, with 83/84 plans on time. `NO_ON_TIME_PLAN` request counts must not be
treated as deadline-miss counts: this label also covers the deliberate quiet
tail after a 100-ms terminal translation pulse expires.

## Prediction and timing evidence

The fitted pose-command forecast had lower planar endpoint RMSE than the
recorded neural forecast on the same executed windows in **all 20 recordings**.
The saved readout includes action-group metrics and yaw errors. This is a
700-ms executed-action comparison with overlapping windows; different arms
produce different trajectories. It does not evaluate unexecuted candidate
outcomes or establish a counterfactual navigation advantage. The unused neural
reference in pose-command and reactive runs is the supervised model.

Of 9593 recorded plans, 9312 were on time (97.1%). Per-run rates ranged from
95.8% to 100%. The requested publication delay was 20 ms throughout; actual
delay was 20 ms for 9592 plans and 22 ms for one supervised layout-3 plan.
The same camera drawing body ran with interpreter-lock release, with the
per-run fixed-pose pixel and unchanged-physics receipts. The earlier severe
scheduling interference did not recur in this batch. This remains measured
simulation timing, not a real-time or hardware qualification.

## Scope and next scientific work

Models, controller sources, six candidate futures, 0.8-s horizon, 300-ms deadline,
noise settings and assignments were frozen before the first new-maze outcome.
No retries or replacements were used. The four new maze topologies and grid
embeddings differ from the explicit 88-layout registry, but remain in the same
maze family. There is one execution per arm/layout and one neural training seed.
Inputs include RGB, synthetic depth with 2-mm noise, ideal gyro and command
history. Broader environment-type testing remains deferred. Realistic sensing,
multiple training seeds and bounded physical-platform evidence remain open.

The next priorities are the shared startup-survey tracking failure and the
JEPA-specific reserve-boundary turn loop. Use the retained failures to isolate
those mechanisms, then evaluate any repair prospectively with matched controls.
Navigation success alone is insufficient evidence that the learned visual
world model improves decisions; prediction accuracy and visual dependence also
remain scientific questions.

The user's larger-candidate idea remains a follow-up, not a change to this
completed comparison. A modest comparison such as 6 versus 12 versus 24
candidates could add finer speeds and turn rates while holding horizon and
safety requirements fixed. Measure physical completion, contacts and full-loop
deadline misses together. More candidates may help coarse action coverage, but
may also expose forecast errors or increase delay. No expanded-candidate test
has been implemented or run.

## Records and retention

- Frozen assignments and source bindings:
  `docs/go2_nogil_navigation_replication_plan_2026-09-16.json`.
- Frozen layout inventory:
  `docs/go2_nogil_navigation_replication_layout_inventory_2026-09-16.json`.
- Per-run narrative and diagnostic details:
  `docs/go2_nogil_navigation_replication_2026-09-16.md`.
- Aggregate directory under the existing navigation development artifact root:
  `go2_nogil_replication_readout_v1_attempt_001/`.
- `result.json` contains every assigned outcome;
  `complete_scientific_readout_v1.json` contains timing, executed-window forecast
  comparisons and all successful physical backtracking readouts.

All failure records, results, physics, RGB, commands, forecasts and source/timing
evidence remain. Full depth is retained for all six fresh-maze failures. The
first fresh-maze success was retained through completion of this comparison,
then its depth pin was superseded by the later JEPA survey-repair reference
under the standing retention policy. Diagnosed redundant success depth was retired under
the existing retention policy; consult each root's `depth_retention.json`
before historical sensor replay. The broad navigation goal remains incomplete.
