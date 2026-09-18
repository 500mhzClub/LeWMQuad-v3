# Measured frontier visits

The fine-stored-obstacle run reached an exploration frontier, then repeatedly
turned toward a waypoint whose distance eventually fell to 1.5 mm. The
frontier remained eligible even after arrival. This experiment gives that
event an explicit completion rule while retaining the existing visual tracker.

Within 10 cm of a selected frontier, the controller requests a view toward
an unknown four-neighbour cell. It prefers neighbours outside the known
inflated obstacles, then those nearer the mission goal. The learned model
scores hold/turn candidates using its predicted yaw and contact outcomes.
Completion requires measured heading within 0.10 rad and a map observation
at or after that alignment. Nearby floor cells within 10 cm of the visited
target are then excluded from exploration target selection, but remain
traversable. A route to an observed goal takes precedence over frontier
exclusion. Return-phase planning does not use those exclusions. Actual
translation-veto recovery retains priority over a frontier visit.

Three focused tests passed in 1.81 s: measured heading plus subsequent map,
goal routing through excluded frontier cells, and the existing fine stored
obstacle checks. The six-cell visual-support rule is unchanged. This is a
bounded development exploration heuristic, not a completeness proof.

Prospective native experiment:
`go2_frontier_visit_native_layout00_v1_attempt_001`, session 33040.
The model, action bank, current obstacle checks, finer stored geometry and
1,800-tick mission budget remain the same. Collection completed all 1,805
frames and 9,025 policy steps over 181.04 simulated seconds, with no contacts
or tracking failure. There were 437 on-time and 13 late plans. The process
exited successfully and all 1,805 camera pairs were saved. No goal arrival
occurred; this remains a negative navigation result.

Two frontier visits completed, at map frames 452 and 472, excluding 12 unique
nearby cells from exploration target selection. Minimum observed goal distance
was 1.456 m; final distance was 2.196 m. The robot later became stationary
under current obstacle vetoes: 269 initial obstacle vetoes and 5,138 latched
command-window vetoes. The first current obstacle veto was at 71.3 simulated
seconds, with 0.44777 m nominal clearance to the nearest 1 cm cell.

The last translating request at 68.18 s was a left arc. Its current obstacle
observation contained no obstacle cells, so both immediate and stopping-margin
checks passed. The later turn revealed an obstacle. A public-sensor/model
replay is checking whether the stored map available before that translation
could have rejected its predicted path. The tracker support rule remains
unchanged; completing frontier visits avoided the previous frame-341 failure
on this different trajectory, which does not establish general tracker
robustness.

Post-estimation native comparison of all 1,805 poses gave median/max position
error 3.05/8.86 mm and maximum XY displacement 1.485 m. Result SHA-256:
`59f1a9452a66fcab4c85755e689b9ede33946aa066ef886db9f329de427a964c`.

Public-sensor/model replay of frames 648–672 reproduced recorded candidate
utilities within 1e-7 and checked each predicted path against the stored fine
map available at that observation. At frame 656, start clearance was 0.45186 m.
The selected left arc predicted a footprint collision; hold and both turns
were clear over the candidate interval. At frame 660, no candidate interval
was clear under its already committed prefix. Earlier frames 648/652 already
had nominal map overlap. This does not imply a counterfactual successful run,
but identifies a command-selection gap: the route uses memory while candidate
paths were not checked against it. Evidence: `memory_forecast_clearance_diagnostic.json`
in the frontier-visit artifact directory.

The next runtime filters all eight predicted 100 ms segments, including the
committed prefix and zero-command tail, against stored 1 cm cell squares with
the same 0.45 m radius. It selects the best remaining original utility and
preserves view-mode action restrictions. If none is clear, it requests hold
and explicitly records that no candidate was clear. This is a nominal learned
forecast check, not a bound on actual motion; current-view execution checks
remain active. A focused test passed in 1.72 s, covering hidden-obstacle path
rejection, view-action restriction and a collision in the common prefix.

Prospective experiment now running:
`go2_memory_forecast_clearance_native_layout00_v1_attempt_001`, session 35934.
That experiment stopped after 289 acquired frames and 1,443 requested policy
steps, with visual failure at frame 284. All camera pairs were saved and the
process exited with failure. There were 67 on-time plans out of 70; the memory
filter did not change any selected action before termination, so this run
does not establish its benefit. Final observed goal distance was 1.582 m;
one frontier view was pending and none completed. The obstacle-process
clock-closed error occurred during shutdown after the primary visual failure.

Exception-local replay found a candidate with 13/23 inliers (56.52%), six
reference and current image cells, and 2.67 mm translation. It fails the 60%
fraction rule, unlike the earlier five-cell failure. Evidence:
`go2_memory_forecast_clearance_native_layout00_v1_attempt_001/consensus_failure_diagnostic.json`.
