# Fixed repeatability and JEPA transfer batch

The current controller completed one exposed-maze round trip in 306.14 simulated
seconds using the frozen supervised model. That success does not establish
repeatability, JEPA transfer, or the benefit of the interrupted-view rule, which
never activated. Run four repetitions in the fixed order JEPA, supervised,
supervised, JEPA on the same exposed layout 1. Keep all four outcomes, including
tracking failures and budget exhaustion. Do not tune the controller between runs
or add repetitions to obtain favorable results.

All four use `InterruptedViewRuntime`, the frozen model snapshots, six actions,
4800-tick budget, 2-mm depth noise, ideal gyro, the same CPU group and the same
planning deadlines. Native jobs run sequentially; no heavy analysis competes
with a timed mission. The recording volume had 46 GiB free before preparation.
Keep this batch's full depth until analysis and keep all failures thereafter.

Primary outcome: independently verified physical goal-and-home arrival with no
disallowed contact. Also measure physical backtracking, tracking failures,
planning deadlines, actual view-interruption events and executed-window forecast
errors. This is development on an exposed maze, with two repetitions per model
and one training seed; it cannot establish fresh-layout reliability or deployment.

Launcher: `scripts/run_go2_view_replan_repeatability_development.py`.
Plan: `docs/go2_view_replan_repeatability_plan_2026-09-17.json`.
For each assignment, run native execution, wait for owner exit and archival,
then evaluate before starting the next assignment. Record results below.

## Outcomes

Assignment 1 (JEPA) launched in session 33378, owner PID 4104865. Launch metadata
identifies `InterruptedViewRuntime` and the frozen JEPA checkpoint
`3790b3bdef97fbb3c340bc43d6d1fbccc6295fb0da80f015f11d064296303978`.
Owner exited zero and full archival completed before evaluation (session 24030,
also exited zero). **Verified goal-and-home round trip in 374.98 simulated
seconds, zero disallowed contacts.** Both one-second quiet dwells passed:
maximum physical distances were 17.29 mm at the goal and 24.53 mm at home.
852/924 plans were on time (92.2%). The interrupted-view rule activated three
times, unlike the preceding supervised success. This shows activation within a
successful JEPA mission, not causal efficacy against an otherwise identical run.
All eleven return corridor edges reversed outbound edges; no pipeline faults
were recorded. Eight coverage patches were actually observed after requested
views. All three interrupted frontier visits targeted cell [8, 0].

Assignment 2 (supervised) launched in session 89648, owner PID 4107271.
Owner exited zero; evaluation session 74079 also exited zero after full archival.
**Verified round trip in 264.44 simulated seconds, zero disallowed contacts.**
The goal/home quiet dwells passed at maximum physical distances 20.14/16.64 mm.
594/641 plans were on time (92.7%). No interrupted-view event occurred.
Assignment 3 (supervised) launched in session 48824, owner PID 4108922.
Owner exited zero after full archival; evaluation session 31199 exited zero.
**Budget exhausted at 480.90 simulated seconds, no arrivals, zero contacts.**
804/1200 plans were on time (67.0%); no pipeline faults occurred. There were
1143 holds, 28 left turns, 28 right turns and one right arc. All 951 plans from
frame 1000 onward held, including 560 on-time plans. Timing alone therefore
does not explain this failure. The translation coverage guard rejected 692
proposals, no coverage patch completed, and two coverage views were interrupted.
The final coverage target remained [-6, 8], with a remote viewing route; its
last selected right arc was replaced by hold because it added five unknown
footprint cells. Pure turns failed nominal forecast clearance in that final
plan. This is a saved-state diagnosis, not proof of a safe escape or the original
entry cause. Keep the full failure and diagnose the coverage/viewpoint conflict
after completing the fixed batch.

Assignment 4 (JEPA) launched in session 75582, owner PID 4111714. Owner exited
one after tracking failure and full archival. Evaluation session 22904 exited
zero. **No arrivals, zero contacts, tracking failure after 469 acquired frames.**
115/116 plans were on time (99.1%). Three frontier views were interrupted, all
targeting cell [8, 0]; the rule therefore did not prevent tracking failure in
this repetition. Two coverage patches resolved. Sensor replay completed using
`scripts/replay_go2_view_replan_repeatability_tracking_failure_development.py`.
All 465 recorded accepted raw poses matched array-exactly, including modes and
selected references; the tracker failed again at frame 465. All eight active
references (457–464) were rejected. Reference 464 failed the original gyro
consensus match-count/strict-majority condition; the others had insufficient
rigid-pose matches. No old-view reference was eligible; all ten diagnostic
camera fits against the five stored views also failed for insufficient matches.
No native state or altered tracking threshold was used, and the failed tracker
was not restarted. The shared replay receipt retains its predecessor schema
name, `cache_trial_tracking_failure_replay.v1`; the wrapper and output root
identify this fourth batch assignment unambiguously.

At frame 456, both selected-feature counts fell below 48 ([10, 43]), triggering
recovery at sensor stamp 47.10 s. The cancellation was published at 47.29 s.
Plans 456, 460 and 464 were on time and requested a right turn toward the saved
stronger view. Actual requested commands changed from left turn to zero at
47.30 s, then remained zero through prefix mismatch until the right-turn request
started at 47.80 s. Tracking failed at sensor stamp 48.00 s, only 0.20 s after
the reversal began. The later pipeline-failure zero request occurred at 48.30 s.
Before recovery, plans
436–452 selected left turns along an observed-floor frontier route, rather than
an active viewpoint-heading request. The final accepted feature witness was
[0, 40]; it is not a fresh feature count at the failed frame. This points to
visual-support warning/recovery timing as a next diagnosis, not evidence that
loosening pose acceptance or adding more viewpoint exclusions would solve it.

## Complete batch result

All four fixed assignments are complete and evaluated. **Two verified round
trips and two failures; zero contacts. Each model succeeded once and failed
once.** No further repetitions belong to this batch.

| Assignment | Model | Outcome | Simulated seconds | On-time plans |
| --- | --- | --- | ---: | ---: |
| 1 | JEPA | Verified round trip | 374.98 | 852/924 |
| 2 | Supervised | Verified round trip | 264.44 | 594/641 |
| 3 | Supervised | No goal; budget exhausted | 480.90 | 804/1200 |
| 4 | JEPA | No goal; tracking failure | Partial recording, 469 frames | 115/116 |

This is evidence of possible end-to-end navigation with both models, but also
direct evidence that the current controller is not repeatable on even this one
exposed maze. The successful runs are not a new independent-layout comparison.
No JEPA advantage or causal benefit of interrupted-view replanning is established.

The fitted pose-command predictor had lower planar endpoint RMSE than the
recorded neural forecast on executed 700-ms windows in all four recordings;
the command-history predictor had lower yaw RMSE in all four. Neural versus
pose-command planar errors in assignment order were 11.06/6.57, 11.25/6.95,
5.13/4.35 and 10.38/5.66 mm. These are overlapping, trajectory-conditional
windows, including holds in the stalled run. They do not evaluate the outcomes
of unexecuted candidates or alternative navigation policies.

Complete machine-readable readout:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_view_replan_repeatability_readout_v1_attempt_001/result.json`.
Reader: `scripts/read_go2_view_replan_repeatability_development.py`.
Keep both full failures and the successful comparison recordings while failure
diagnosis is active. Diagnose the tracking rejection and the coverage/viewpoint
stall before another large navigation batch. Separate future development fixes
from later fresh-layout comparisons, model-training improvements and causal
predictive/memory controls. Realistic sensing, full-loop timing and hardware
validation remain outstanding.
