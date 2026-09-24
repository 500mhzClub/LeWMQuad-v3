# Independent-maze comparison with shared startup recovery

**Interrupted after four evaluated attempts because of a shared runtime defect.
Do not launch assignments 5–20 under this plan.** Two runs succeeded, one lost
tracking and one was interrupted after a live execution stall. The sixteen
unexecuted assignments are not failures. Original sources and all four outcomes
remain unchanged; any corrected cohort must be reported separately.

The earlier 20-run study completed 14 round trips, with all five controllers
losing tracking during the initial survey on one maze. Prompt cancellation of
pre-recovery commands plus deferral of an interrupted initial survey subsequently
completed exposed-maze JEPA/supervised missions and a two-maze JEPA transfer.
The four-run heading-release comparison was unexercised; retain the original
heading-release rule. Its outcomes do not justify disabling that rule.

Freeze four new development mazes before any navigation, distinct from the
explicit 94-layout registry by the existing topology/grid checks. Use construction
seed 2026091631, physics seeds 2026099600+i and appearance seeds 2026099700+i.
No selection or replacement based on navigation outcomes. This is the same maze
family; broader environment types remain deferred as requested.

Each maze receives the same five arms as the completed 20-run comparison:
JEPA, same-data supervised rollout, fitted pose-command prediction,
instantaneous ranking with predictive guards, and reactive feedback without
forecast selection. Apply the same startup repair to every arm. Rotate execution
order by maze using the existing fixed schedule. Preserve every failure; do not
repair the controller, change models or add retries during this batch.

The primary outcome is a physically verified goal-and-home round trip without
contact. Report all five per-maze outcomes, quiet arrivals, physical backtracking,
tracking failures, completion time, planning deadlines, survey deferral and
forecast errors. JEPA versus supervised addresses training objective; supervised
versus instantaneous addresses predictive ranking with guards retained; reactive
feedback is a wider controller-package comparison. Pose-command compares learned
prediction with a simpler fitted motion model. Persistent routing memory stays
enabled throughout; this batch cannot isolate its contribution.

Keep the original frozen model seed 2026091001, training data, six candidates,
0.8-second horizon, 400-ms planning cadence, 300-ms deadline, 20-ms extra wait,
4800-tick budget, 2-mm depth noise, ideal gyro and arrival/clearance thresholds.
These are measured-simulation experiments; neither sensing nor full-loop timing
is deployment validated. One execution per arm/maze and one training seed cannot
establish broad reliability or multi-seed JEPA superiority.

Hardware inspection before preparing this batch found no competing navigation
or training job, 64 GiB RAM available, CPU effectively idle and both GPUs idle.
The discrete GPU had about 1.84 GB of 34.21 GB VRAM occupied. Artifact storage
had about 7.1 GiB free and the workspace 1.5 GiB. Reuse the measured CPU/software
renderer setup; no new GPU conversion or parallel simulation benchmark is needed
for this unchanged workload. Run native missions sequentially with the existing
per-maze CPU groups, and no heavy concurrent analysis, because scheduling affects
the scientific outcomes. Evaluate and retire eligible diagnosed redundant depth
between runs under the standing policy; retain active failures and references.
Each launch requires at least 4 GiB recording headroom. Source/test preparation
is small and does not compete with a timed mission.

Launcher: `scripts/run_go2_recovery_repaired_navigation_replication_development.py`.
Prepare once with `--prepare`, then execute assignments 1–20 in order and evaluate
each with `--assignment N --evaluate` before proceeding. The plan and inventory
are fixed before the first run. This is a new cohort, not a replacement for the
original 20 outcomes or a retest of the four heading-release executions.

Preparation accepted four distinct layouts from five structural candidates;
one candidate repeated a prior topology and was rejected before navigation.
All five runtime compositions include both startup-recovery components, and
the three focused initial-survey tests passed. No model training was performed.

Assignment 1, JEPA on layout 0, passed both physical quiet arrivals in 175.66
simulated seconds with zero contacts and 416/431 plans on time. Its return
reversed all seven unique outbound corridor edges, with no invalid transitions.
The initial survey completed all nine views without deferral; no recovery plans
were needed. This is a successful independent-maze execution, not an exercised
test of the startup repair or evidence of superiority over the pending baselines.
Retain this first success in full as the new cohort's depth reference.
The aggregate readout is under `go2_recovery_repaired_replication_readout_v1_attempt_001`.

Assignment 2, supervised rollout on layout 0, also passed both physical quiet
arrivals and reversed the same seven outbound corridor edges home. It completed
in 223.58 simulated seconds with zero contacts and 538/549 plans on time. Its
initial survey likewise completed all nine views without deferral or recovery
plans. JEPA finished this first matched pair 47.92 simulated seconds sooner.
This is one execution per model on one maze, not a general JEPA advantage.

The supervised trace requested nonzero motion for 155.88 seconds versus JEPA's
157.64, despite its longer mission. It selected hold in 109 plans versus 15 for
JEPA. These observations locate much of the time difference in zero-command
intervals but do not uniquely explain its cause. On each model's own recorded
trajectory, fitted pose-command XY RMSE remained lower than the neural forecast:
7.22 versus 11.42 mm for JEPA, and 6.09 versus 9.71 mm for supervised. These are
different executed-window populations and do not isolate model accuracy across
the two navigation policies.

After its evaluation and diagnosis, assignment 2's redundant depth was retired:
4468 leaves, 1,334,792,192 allocated bytes reclaimed. All 4548 non-depth files and
40 JSON identities were preserved. Full sensor replay for that run is unavailable;
the first JEPA success remains full. Eighteen fixed assignments remain, beginning
with pose-command prediction on layout 0.

Assignment 3, pose-command on layout 0, failed with `measured visual pose
unavailable` after 751 acquired camera frames. It had no goal arrival and zero
contacts; 182/186 plans were on time. This was after a completed nine-view
initial survey, not the earlier startup-sweep conflict. The last three plans
requested visual recovery, but tracking still failed. Preserve its full recording
for later diagnosis; no retry or controller change is made within this cohort.
Three assignments are evaluated: two successes and one tracking failure.

Assignment 4, instantaneous ranking, stopped advancing after 1565 camera frames
with no goal arrival and zero contacts. Before interruption it had 383/389 plans
on time and 45 visual-recovery plans. Progress logging remained unchanged for
247 seconds while the process was live and its threads waited. SIGINT to the
owner allowed its exception/finalization path to persist the complete acquired
recording. The traceback places the main thread in `request()` waiting for
`self.lock`. The recorded `KeyboardInterrupt()` and subsequent clock-closed
worker errors describe our cleanup, not independent tracking failures.

The publisher in `visual_recovery_dispatch_hold_development.py` calls the
blocking measured simulation clock while holding that same request lock.
The simulator cannot advance to release the clock if it is waiting for the lock.
A focused test reproduces this dependency in the original method and shows it
absent with `MeasuredRecoveryPublicationMixin`. Direct live worker stack capture
was unavailable, so the exact worker stack at the native stall was not observed.
The new method waits before acquiring the request lock, snapshots publication
time after reacquiring it, and preserves cancellation and duplicate-trigger
handling. Three regression tests passed, including correct timestamp recording
when the clock advances during lock acquisition.

All four outcomes and the interruption rationale are recorded in the aggregate
directory's `execution_interruption_v1.json`. The interrupted run retains its
full sensor recording, intervention receipt and failed debugger outputs. The
cohort ends here rather than mixing the publication fix into pending arms.
One new exposed-maze technical-validation run will test the corrected publisher;
it cannot replace the interrupted outcome or establish a five-arm ranking.

End the temporary full-depth pin on assignment 1 now that this cohort has been
interrupted and its completed-success analyses are finished. Its redundant depth
is retired under the standing policy, preserving every non-depth artifact and
outcome. The original exposed/fresh-maze repair references remain full, as do
both failures from this cohort. This recording is not a training input and has
no pending raw-depth replay. See its `depth_retention.json` for the inventory;
full historical sensor replay is unavailable.
