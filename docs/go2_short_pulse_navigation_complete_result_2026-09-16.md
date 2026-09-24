# Complete short-pulse navigation comparison

The raw pulse-trained neural planners did not complete a round trip on either
new development maze. Reactive control completed both, and the pose/command
predictor completed one. This experiment does not establish a JEPA advantage
or reliable navigation using the current raw learned forecasts. The broader
navigation goal remains incomplete.

All fourteen fixed missions have physical outcomes and actual-treatment
evaluations. There are two independent layouts, one execution per method per
layout and one neural training seed (2026091001), not fourteen independent
environments. Original failures remain part of the results.

| Method | Round trips | Outbound arrivals | Missions with contact |
| --- | ---: | ---: | ---: |
| JEPA | 0/2 | 1/2 | 0 |
| Direct | 0/2 | 0/2 | 1 |
| Supervised rollout | 0/2 | 1/2 | 0 |
| Pose/command predictor | 1/2 | 1/2 | 0 |
| Command-history predictor | 0/2 | 1/2 | 0 |
| Instantaneous ranking with predictive guards | 0/2 | 1/2 | 0 |
| Reactive | 2/2 | 2/2 | 0 |

Total: **3/14 round trips, 7/14 outbound arrivals, one contact failure**.
Neural methods account for 0/6 round trips, 2/6 outbound arrivals and that
contact failure. Maze 0 had two round trips and six outbound arrivals; maze 1
had one round trip and one outbound arrival, both reactive.

The principal failure mechanisms are concrete, but their individual causal
effects have not all been tested in new closed-loop missions:

- The direct maze-0 collision occurred near a wall end. The stored visible
  surface map overstated clearance to the physical wall volume by about
  66 mm. Position tracking error at contact was only about 1.2 mm. Full raw
  depth remains for this geometry diagnosis.
- Instantaneous ranking on maze 1 became stuck during its opening panorama.
  Noisy wall returns and 1-cm cell quantization reduced mapped clearance;
  predicted drift during the common zero-command prefix reduced it further
  to 0.4491 m, below the 0.45-m rule, despite actual clearance of 0.4715 m.
  This was a retained predictive-guard failure, not a main-ranking failure.
- The other predictive maze-1 runs completed their opening scans and continued
  frontier exploration, but never reached the goal-cell routing phase.
  Initial scans took 98.8–178.0 s versus reactive's 32.8 s. Planning service
  medians were about 106–113 ms versus reactive's 28.5 ms; 280–541 of their
  1,200 plans were late. Repeated stale-observation vetoes also interrupted
  otherwise timely plans. Faster planning alone is therefore not an adequate
  diagnosis, and better prediction error alone did not deliver navigation.

All latched command stops in those five predictive maze-1 runs began with
stale obstacle observations. For direct, 538/790 initial stale triggers
preceded the next obstacle result by only 2–6 ms; for supervised, 375/614 did.
A separate set of results shared the dispatch timestamp. In the deadline
readout of reactive, command-history, pose/command, supervised and direct,
none had a newer obstacle stage logged strictly before the stale request.
Same-timestamp records cannot
establish delivery before dispatch. These measurements motivate testing the
equivalent faster obstacle implementation, without relaxing the freshness rule.

On each neural method's own executed maze-1 windows, the unused pose/command
forecast had lower XY error than the neural forecast:

| Executed neural method | Matched 700-ms windows | Neural XY RMSE | Pose/command XY RMSE |
| --- | ---: | ---: | ---: |
| JEPA | 176 | 26.735 mm | 11.188 mm |
| Direct | 99 | 20.888 mm | 9.006 mm |
| Supervised rollout | 165 | 20.154 mm | 10.664 mm |

These windows overlap and come from different executed trajectories. They
are not independent trials, a matched cross-method prediction population,
or evidence of alternative-policy navigation outcomes.

The comparison uses original mission coordinates and raw nominal-composed
neural forecasts, with no external neural XY correction. This differs from
the earlier 35/36 successful neural-RGB cohort, which retained an external
pose-dependent correction and used different mazes with different observed
timings. That earlier success
cannot be carried over to the current raw-model treatment. The command-history
fit used the matched new training population; the pose/command fit is an older
practical comparator with different training data.

Reactive uses different forecast guards, terminal feedback and compute, so
its success does not isolate predictive ranking. Instantaneous ranking keeps
predictive guards and is not a full prediction-off control. Both reactive
missions really used measured-heading alignment and 100-ms terminal forward
pulses; corrective treatment records preserve this fact alongside the original
incorrect launch flag. The underlying mission, mapping and physical arrival
evaluation were shared.

This is development simulation with synthetic 2-mm depth noise and ideal gyro,
not sealed final evaluation, calibrated real sensing, wall-clock qualification
or hardware deployment evidence. The remaining geometry and initial-survey
failures also prevent a reliability claim. Diagnosed depth recordings have
been retired under the retention policy; results, failures, RGB/body/commands,
poses, physics, models and other evidence remain.

The next fixed experiment is an original-versus-packed obstacle observer pair
on exposed maze 1 using the same supervised model and controller, with no
concurrent spare-core analysis. It tests actual command continuity, progress,
arrivals and contacts. Exact replay equivalence or lower processing time is
not sufficient for success. See
[the experiment plan](go2_obstacle_grouping_navigation_2026-09-16.md).

Authoritative aggregate:
`go2_short_pulse_navigation_complete_v1_attempt_001/result.json` under the
navigation artifact root. The complete trajectory figures are in
`go2_short_pulse_navigation_maze00_complete_v1_attempt_001` and
`go2_short_pulse_navigation_maze01_complete_v1_attempt_001`. Per-assignment
details and diagnoses are recorded in
[the study journal](go2_short_pulse_navigation_2026-09-16.md).
