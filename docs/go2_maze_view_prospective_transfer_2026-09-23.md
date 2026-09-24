# Prospective motion-readout transfer on two new mazes

Latest status, September 23: neural evaluation is COMPLETE on all 240 fixed
rows and 360 images, in 510.21 wall seconds. PID 245575 exited after saving
`evaluation/result.json`. Both trained heads completed the same 440-update
budget; there was no checkpoint selection. The original maze-00 diagnostic
also completed all 128 rows. The historical preparation notes below describe
earlier states and are superseded by these completed results.

### Completed actual-future decoder transfer

Each cell reports XY RMSE in mm / yaw RMSE in degrees. Each translation row
has 16 departure windows; each turn row has 32. Hold and all-window metrics,
every prediction/error, plan identity and checkpoint hashes are retained
in `evaluation/result.json` and `command_references/result.json`.

| Maze | Horizon | Departure group | Initial mixed head | Existing-data control | Maze-data head | Command history |
|---|---:|---|---:|---:|---:|---:|
| 0 | 500 ms | Translation | 58.29 / 4.52 | 61.45 / 4.83 | 17.92 / 1.11 | 9.36 / 1.32 |
| 0 | 700 ms | Translation | 91.60 / 7.83 | 94.99 / 8.12 | 27.02 / 1.67 | 9.92 / 1.27 |
| 1 | 500 ms | Translation | 61.95 / 4.62 | 65.30 / 4.79 | 20.90 / 1.26 | 9.14 / 1.33 |
| 1 | 700 ms | Translation | 94.91 / 7.60 | 98.81 / 7.11 | 27.17 / 1.13 | 9.68 / 1.25 |
| 0 | 500 ms | Turn | 14.03 / 7.36 | 15.87 / 7.19 | 5.71 / 1.78 | 4.73 / 2.07 |
| 0 | 700 ms | Turn | 16.64 / 11.71 | 20.66 / 11.69 | 7.11 / 2.13 | 6.64 / 2.13 |
| 1 | 500 ms | Turn | 13.56 / 7.61 | 14.82 / 7.23 | 4.46 / 2.34 | 4.57 / 2.04 |
| 1 | 700 ms | Turn | 14.12 / 11.59 | 16.54 / 11.24 | 5.97 / 2.24 | 6.43 / 2.10 |

Maze-data training reduces 700-ms translation XY RMSE by about 70.5% and 71.4%
relative to initialization on the two mazes, while further existing-data
training worsens it. This supports a training-coverage effect for the frozen
encoder's motion readout under this collection distribution. It does not
isolate geometry, appearance, heading or motion coverage, which changed together.
Command-history translation remains better. Improvement is not uniform:
maze-0 hold-departure XY error rises from 8.59 to 10.30 mm at 500 ms and from
9.16 to 10.14 mm at 700 ms. Only two independent mazes, the matching fixed-tape
distribution, and use of actual future images limit the claim. No online
prediction, JEPA-objective advantage or navigation improvement is established.

The original exposed navigation-trajectory diagnostic shows much smaller
transfer: at 700 ms, actual-future translation XY RMSE changes from 78.28 to
72.87 mm, and action-conditioned XY RMSE from 67.17 to 64.82 mm, versus command
history 8.65 mm. These outcomes warrant separating action-conditioned prediction
from actual-future decoding on the newly collected scenes before attributing
the gain to a world model. No head has been promoted to navigation.

### Follow-up predictor/readout decomposition

The completed decoder results motivate a diagnostic on the same 240 windows,
with no row selection or new recordings. This follow-up is post hoc relative
to those decoder results. It compares the frozen action-conditioned and
no-future-action predictors against actual future and persistence (unchanged
current) features, using each of the three fixed readouts. It reports both
latent L1/MSE and XY/yaw errors per maze, horizon and departure group. Existing
command-history, integrated-command and zero-motion references are retained.
No encoder, predictor or readout is trained or promoted by this diagnostic.

Predictor inputs are the native RGB contexts at -1000/-500/0 ms, causal public
applied-command history and the fixed requested tape projected through the
same platform limiter used online. Projection matches future recorded applied
commands exactly in every window; future measured commands only check alignment
and never become model inputs. Actual future features are used for evaluation
and the explicitly labelled observed-future comparison. The observed-future
readout output is checked against the completed decoder evaluation.

Preparation completed with 240 rows and 376 unique RGB paths. The diagnostic
was launched as PID 248224 on CPU cores 0--3 after both earlier evaluators had
exited, with 61.9 GiB available RAM and 40.8 GiB free disk. Dense tokens stay in
RAM; no tensor cache or depth recording is added. Its source is
`scripts/evaluate_go2_maze_view_predictor_transfer_development.py`; plan, inputs,
process record and `worker.log` are under `predictor_diagnostic` in the existing
collection root. The diagnostic completed all 240 rows in 720.08 wall seconds;
the process exited and its actual-future outputs reproduce the preceding
decoder evaluation exactly (maximum absolute difference 0). This diagnoses
component transfer and does not establish navigation success or a causal
JEPA-objective advantage.

Each cell below is XY RMSE in mm / yaw RMSE in degrees. The observed, predicted,
blind and persistence columns all use the same maze-data readout.

| Maze | Horizon | Group | Observed future | Action prediction | No future action | Persistence | Command history |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 500 ms | Translation | 17.92 / 1.11 | 30.60 / 3.31 | 47.05 / 6.44 | 68.12 / 6.60 | 9.36 / 1.32 |
| 0 | 700 ms | Translation | 27.02 / 1.67 | 46.62 / 4.56 | 78.65 / 9.98 | 103.42 / 10.26 | 9.92 / 1.27 |
| 1 | 500 ms | Translation | 20.90 / 1.26 | 35.16 / 3.75 | 48.05 / 6.53 | 68.81 / 6.51 | 9.14 / 1.33 |
| 1 | 700 ms | Translation | 27.17 / 1.13 | 52.15 / 4.20 | 81.22 / 10.03 | 103.96 / 10.17 | 9.68 / 1.25 |
| 0 | 500 ms | Turn | 5.71 / 1.78 | 9.15 / 2.92 | 17.18 / 9.98 | 6.98 / 10.25 | 4.73 / 2.07 |
| 0 | 700 ms | Turn | 7.11 / 2.13 | 9.59 / 3.50 | 21.52 / 14.46 | 8.85 / 15.31 | 6.64 / 2.13 |
| 1 | 500 ms | Turn | 4.46 / 2.34 | 9.55 / 3.20 | 16.11 / 9.98 | 7.11 / 10.27 | 4.57 / 2.04 |
| 1 | 700 ms | Turn | 5.97 / 2.24 | 9.43 / 3.89 | 18.78 / 14.52 | 9.32 / 15.33 | 6.43 / 2.10 |

With predicted rather than observed futures, the readout intervention reduces
700-ms translation error from initial/control 83.63/89.64 to 46.62 mm on maze 0,
and from 90.92/94.94 to 52.15 mm on maze 1. Useful information therefore survives
action-conditioned prediction, but an additional gap remains relative to the
27-mm actual-future decoder and approximately 10-mm command-history reference.
Action prediction also has lower all-window latent MSE than action-blind and
persistence at both horizons on both mazes:

| Maze | Horizon | Action MSE | No-future-action MSE | Persistence MSE |
|---|---:|---:|---:|---:|
| 0 | 500 ms | 0.48853 | 0.58020 | 0.69779 |
| 0 | 700 ms | 0.50566 | 0.61236 | 0.73543 |
| 1 | 500 ms | 0.48677 | 0.58415 | 0.70260 |
| 1 | 700 ms | 0.50667 | 0.61374 | 0.73893 |

This is evidence for future-action conditioning within this diagnostic, not an
ablation of JEPA's training objective. Gains are not universal: persistence has
lower turn XY error, but much worse turn yaw; action-blind hold XY is lower on
maze 1 at 500 ms. All hold/all-window results, all three readouts, latent L1,
and every row remain in `predictor_diagnostic/result.json`.
The next intervention is a fixed readout comparison in closed-loop development
navigation. Existing navigation layouts are now exposed, so this follow-up
must not be counted as another independent prospective cohort.

The matched old-data/maze-data readout fit is still extracting features. Before
its results, fix a small separate evaluation to distinguish improved transfer
from improvement only on the previously examined maze-00 trajectory. This is
an evaluation-only component experiment, not navigation or a sealed benchmark.

Two mazes are generated with construction seed 2026092301, disjoint from the
explicit 113-layout registry: 109 earlier layouts including the completed
four-maze cohort, plus all four new training mazes. Each maze has four
geometry-selected cardinal views using the training collection's selection
rule. All eight contexts execute its unchanged 17-second motion tape. Neither
prediction errors nor fit outcomes choose the contexts, recordings or windows.
No failed case is replaced. All RGB, commands, physics and failures are retained;
unused depth arrays are not written. The independent geometry count is two,
not eight or the number of overlapping windows.

Compare the initial mixed-data head and both fixed-final-step heads from
`go2_maze_view_readout_v1_attempt_003`. Use actual current/future frozen V-JEPA
features at 500 and 700 ms for departure frames 10,20,...,150 in every context.
Report every missing/contact-prefix exclusion and physical stop. Targets use
the same full-body-frame translation and relative-body yaw convention as
training. Report XY/yaw RMSE per maze, horizon and departure command group.
Windows can cross command transitions. There is no fitting, checkpoint
selection or automatic promotion using this population. The fixed tape and
view-selection rule match training; this tests new geometry/appearance under
that collection distribution, not unrestricted visual generalization.

Collection root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_maze_view_transfer_v1_attempt_001`.
The prepared `plan.json` predates fit completion and records the layouts, tape,
target selection and fit-plan identity. Preparation completed successfully.
Two detached collection workers use cores 4--7 and 8--11, separate from the
existing four-core fit process. At launch, about 58 GiB RAM and 42 GiB disk
were available. Recording has a 1-GiB free-space floor. Initial worker PIDs
are 226258 and 226259; inspect live processes and logs for current status.

Sources:
`lewm/maze_view_transfer_layouts_development.py`,
`scripts/collect_go2_maze_view_transfer_development.py`, and
`scripts/evaluate_go2_maze_view_transfer_development.py`.
After all recordings finish, the evaluator's `--prepare` fixes and persists
targets while fit results remain unavailable. Run evaluation only after both
trained checkpoints exist. The original fixed maze-00 evaluation is unchanged.

Collection and target preparation are now complete. Both workers completed all
four assigned cases and exited. All eight recordings have 171 camera pairs,
zero disallowed contacts and zero physical stops: 1368 camera pairs total.
The fixed selection produced 240 evaluation rows with no exclusions and
360 unique image paths. At each horizon there are 64 turn, 32 translation and
24 hold-departure windows. These counts reflect departure commands and do not
imply motion stayed in the same category throughout each window.

`collection_result.json`, `transfer_targets.json` and `evaluation_plan.json`
are saved in the collection root. Preparation exited zero while the matched
fit was still extracting features, before any optimizer or fit outcome.
The evaluation itself is pending both fixed-final-step checkpoints; no
transfer-performance result is claimed yet.

The frozen command-history, command-integrated and zero-motion references are
now complete on the same 240 targets. This supplemental analysis was added
before the readout fit produced results. It preserves the original target file,
neural evaluator source and evaluation plan. The command-history reference uses
the existing short-pulse command-only checkpoint, four causal public
applied-command histories and the collection plan's fixed future requested
commands. Future measured commands and physics are not predictor inputs.
Every saved command-history prediction was compared with the navigation
runtime's `command_predictions` implementation; maximum absolute difference
was 5.551115123125783e-17 over all 240 rows. Departure timestamps match the
camera observations. The original neural evaluator's source binding still
matches. Neither reference model nor readout was fitted by this analysis.

| Maze | Horizon | Translation-departure windows | Command-history XY / yaw RMSE | Integrated-command XY / yaw RMSE | Zero-motion XY / yaw RMSE |
|---|---:|---:|---:|---:|---:|
| 0 | 500 ms | 16 | 9.36 mm / 1.32 deg | 26.87 mm / 3.04 deg | 73.10 mm / 6.49 deg |
| 0 | 700 ms | 16 | 9.92 mm / 1.27 deg | 28.64 mm / 2.96 deg | 108.18 mm / 10.15 deg |
| 1 | 500 ms | 16 | 9.14 mm / 1.33 deg | 26.86 mm / 3.04 deg | 73.10 mm / 6.50 deg |
| 1 | 700 ms | 16 | 9.68 mm / 1.25 deg | 28.81 mm / 2.94 deg | 108.12 mm / 10.16 deg |

On all 60 windows per maze at 700 ms, command-history XY/yaw RMSE is
7.99 mm / 1.70 degrees and 7.88 mm / 1.67 degrees respectively. All groups,
including turns and holds, are retained in `command_references/result.json`
under the collection root, along with every prediction and error. The source
is `scripts/evaluate_go2_maze_view_transfer_references_development.py`.
These are descriptive component references: actual-future image decoding and
command forecasting have different inputs, and neither establishes navigation
performance. Neural transfer results remain pending.
