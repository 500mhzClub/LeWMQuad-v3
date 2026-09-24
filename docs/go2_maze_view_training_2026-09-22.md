# Broader native maze views for the frozen-feature motion readout

Latest status, September 23: attempt 003 completed feature extraction and both
matched 440-update fits. Total extraction/fitting time was 12,593.27 seconds;
the optimizer stage took approximately 28 seconds. The fixed-final checkpoint
SHA-256 values are `97b3972cf6e1283d8825e7e900c0d504ef88ce3f495690d8b8ac284cee2fb19f`
for `old_data` and `aa853c6f16d7f78e01ade2bb91c1dfceebda2b1a12b0657a223b184fe2005fb8`
for `maze_data`. The original maze-00 diagnostic started automatically in the
same process (PID 203565, cores 0--3). The separately prepared two-maze
evaluation started as PID 245575 on cores 4--7, with 56 GiB available RAM and
41 GiB free disk checked before launch. Both evaluations subsequently completed
and their processes exited. The original diagnostic saved all 128 rows in
568.65 wall seconds; the separately prepared two-maze evaluation saved all 240
rows in 510.21 wall seconds. Historical launch/recovery notes below describe
the earlier states.

### Completed original navigation-trajectory diagnostic

Each cell reports XY RMSE in mm / yaw RMSE in degrees, with 32 windows per
group/horizon. These are the same exposed development windows fixed before
fitting. Complete action-blind scores and all predictions are retained in
attempt 003's `maze00_evaluation/result.json`.

| Group | Horizon | Feature source | Initial mixed head | Existing-data control | Maze-data head | Command history |
|---|---:|---|---:|---:|---:|---:|
| Translation | 500 ms | Actual future | 53.20 / 5.44 | 55.98 / 6.26 | 49.38 / 4.81 | 6.13 / 0.99 |
| Translation | 500 ms | Action-conditioned | 42.49 / 4.91 | 49.85 / 5.88 | 40.36 / 4.55 | 6.13 / 0.99 |
| Translation | 700 ms | Actual future | 78.28 / 7.47 | 79.23 / 8.34 | 72.87 / 6.25 | 8.65 / 1.06 |
| Translation | 700 ms | Action-conditioned | 67.17 / 7.02 | 73.33 / 7.98 | 64.82 / 6.01 | 8.65 / 1.06 |
| Turn | 500 ms | Actual future | 17.29 / 5.92 | 18.19 / 7.10 | 17.44 / 4.46 | 6.00 / 0.59 |
| Turn | 500 ms | Action-conditioned | 16.82 / 6.34 | 16.05 / 7.60 | 17.96 / 4.84 | 6.00 / 0.59 |
| Turn | 700 ms | Actual future | 19.96 / 10.35 | 20.59 / 11.47 | 20.09 / 6.86 | 6.72 / 0.77 |
| Turn | 700 ms | Action-conditioned | 18.36 / 10.24 | 16.96 / 11.60 | 18.39 / 6.95 | 6.72 / 0.77 |

On this trajectory the maze-data intervention provides modest translation
improvement (700-ms action XY: 67.17 to 64.82 mm, about 3.5%) and larger turn-yaw
improvement (10.24 to 6.95 degrees), while turn XY does not improve. Command
history remains substantially more accurate. The separate fixed-tape mazes
show much larger actual-future decoding gains, documented in
`go2_maze_view_prospective_transfer_2026-09-23.md`. This difference limits a
general navigation-transfer claim. No head is promoted, and no new navigation
success or JEPA-objective advantage is claimed.

Collection is complete on sixteen fixed training contexts from four new mazes.
This follows the completed diagnostic in which the same readout decoded larger
training translations with 11.19-mm error but maze translations with 77.38-mm
error even from actual future images. Removing visual change worsened its
training error to 105.34 mm. The next intervention tests training-view coverage;
the encoder and action-conditioned predictor remain fixed.

The four training mazes exclude the explicit 109-layout source registry,
including all four current prospective navigation mazes. For each training
maze, four initial contexts face open passages in the cardinal directions.
Selection uses geometry alone, preferring dead ends, corridors and junctions;
it does not use prediction errors or navigation outcomes. Geometry, appearance,
heading and motion coverage change together, so this does not isolate one of
those factors.

Each context executes the same 17-second training tape: quiet history,
left/right reversals, two seconds forward, one second of each arc, more
reversals and final settling. These are training excitations, not navigation
trials. All commands, native physics/contact traces, primary and auxiliary RGB,
and failures are retained. Unused depth arrays are not written. Labels cover
100--800 ms, using all contact-free departure contexts from frame 10 onward
with the complete eight-horizon future available.

The existing native collector, low-level gait, camera wrappers and physical
checks are reused without changing their files. Source-specific pack checks
verified the actual robot spawn position and quaternion for all sixteen views.
The first case completed 171 frames in 37.95 wall seconds before persistence,
without contacts or a physical stop; top-level recording data occupied 14.71
MiB. A second worker was then started. The two workers use CPU cores 0--3 and
4--7; the active navigation cohort retains cores 8--15 and 24--31. Pre-launch
available RAM was about 39 GiB, and the collection filesystem had about 42 GiB
free. Collection has a 2-GiB free-space floor. This preserves the more limited
workspace space for the current cohort.

Collection root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_maze_view_training_v1_attempt_001`.
Its `plan.json` records the exact layouts, contexts, tape and source identities.
Physical stops retain their complete outcomes and contact-free prefixes;
there are no replacement attempts or outcome-based context substitutions.

Both collection workers and the result reader exited zero. All sixteen
recordings completed, each with 171 camera pairs, zero contacts and zero
physical stops. The dataset has 2,448 departure contexts and 19,584 overlapping
motion windows, not that many independent examples or environments.

| Horizon | Windows | XY displacement p90 / maximum | Windows with XY >=50 mm | Maximum absolute yaw |
|---|---:|---:|---:|---:|
| 500 ms | 2448 | 89.10 / 104.68 mm | 582 | 13.53 degrees |
| 700 ms | 2448 | 121.18 / 143.08 mm | 653 | 18.88 degrees |
| 800 ms | 2448 | 135.20 / 162.21 mm | 682 | 21.33 degrees |

All eight horizons are in `motion_support.json`. The sample file SHA-256 is
`58f3da4bd989777ee5765e41fd1dc62a6cc1c96c747ce07752a914c831f16999`.

The sixteen starting cells comprise four degree-one dead ends, eight
degree-two passages and four degree-three junctions. The initially faced
neighbour is degree one in three recordings, degree two in ten, and degree
four in three. Six have straight continuation beyond that neighbour.
Inspection of sampled recorded RGB confirms the native textured-wall/floor
rendering, but these short cardinal-view tapes do not exhaust the maze-view
distribution. A negative fit result would not rule out broader data coverage.

The matched fitting/evaluation workflow has been launched on CPU cores 0--3.
Each arm receives 440 updates of 64 examples. Both share 32 examples from the
5,966-context original-plus-heading pool per batch. The other 32 come from that
old pool for the control and from the 2,448 new maze contexts for the treatment.
Both halves have matched 100--800-ms horizon schedules. Architecture, initial
deployed mixed-data checkpoint, target normalization and optimizer are identical.
The encoder and predictor remain frozen; pooled FP16 features are held in RAM.
There are 9,662 image paths before byte-identical image deduplication.

The diagnostic evaluation reuses the exact preceding 128 target rows and
selected frames, verified byte-identical before fitting: 32 translations and
32 turns at 500/700 ms. It compares actual-future, action-conditioned and
action-blind features plus the fixed references. Evaluation follows fitting
automatically if the fit succeeds. These are exposed development diagnostics;
any promotion still needs new prospective navigation evidence.

The first fitting launch stopped before encoding or any optimizer updates:
the binding helper rejected the closure introduced by `torch.no_grad` around
the reused encoder. Its failure, prepared inputs and source remain intact in
`go2_maze_view_readout_v1_attempt_001`. The corrected runner binds the original
undecorated encoder and restores `torch.no_grad`. It uses the separate root
`go2_maze_view_readout_v1_attempt_002`; samples, image paths, training schedule
and evaluation targets were verified byte-identical to attempt 001.
The active runner is `scripts/run_go2_maze_view_readout_development.py`.
The corrected launch passed startup and encoded its first 32 of 9,102 unique
images in 44.3 seconds. At that initial rate CPU extraction is about 3.5 hours;
the fitting updates and fixed evaluation follow. No fit or transfer outcome is
available yet.
No new head is automatically promoted, and no current-cohort setting changes.

September 23 interruption recovery: attempt 002's tool session and recorded
process disappeared without a result, checkpoint, optimizer-progress file or
exception record. The cause is unknown. Its last verified encoding count was
3840/9102; RAM-only features were lost. The original attempt remains intact with
an `interruption_record.json`. A separate attempt 003 has been prepared and
launched through `scripts/run_go2_maze_view_readout_recovery_development.py`.
Samples, image paths, schedules and transfer targets are byte-identical to
attempt 002; model initialization, optimizer, budgets and CPU allocation are
unchanged. It repeats encoding and fitting from the fixed starting head, not
from a partial checkpoint. Fitting had produced no saved updates. The new
process has an independent session and persistent `worker.log` in its output
root, retaining progress if the monitoring connection is lost. No scientific
fit or transfer result is available yet.
