# Complete corrected readout running; native queue preserved

Update:the readout subsequently completed and was independently reconstructed.
See docs/go2_all_phase_corrected_prediction_readout_result_2026-09-10.md.
Handle45959 closed0; preserve its completed artifacts and do not relaunch it.

This goal turn made progress: a separate complete corrected-prediction readout
was implemented, tested and started while the original native scene continues.
It changes no queued assignment, coefficient, model weight or controller. The
full navigation goal remains active, with30 completed audited episodes and zero
verified round trips. No corrected readout result is claimed yet.

Original process handle45959/PID2643209, creation1789021692.3, is live:
scripts/read_go2_all_phase_corrected_predictions_v1.py.
The latest direct process observation showed3m07s elapsed,99.3% of one CPU and
1,476,792,320 RSS bytes (ps1442180KiB). It is in prerequisite full correction/
model/data admission, before its exclusive output launch. No launch.json,
result.json or failure.json exists yet under
go2_all_phase_corrected_prediction_readout_v1_attempt_001. Preserve this process;
do not infer failure from its initially quiet output or start another copy.

Exact required inputs are completed full-fit result
44c4cd65812b021b29cfb0aff33e2058cfaaa71dcc627d36eced30dfac17ea35 and correction
1b36dc77ca51d342e45d73da027ebdbdbd1263ab5be4142766948fd8258dd460. Complete
correction admission reconstructs all30 head coefficient records and all18 fit/ledger/raw
score/snapshot identities before creating output. All36 corrected model/role
arrays are written before any corrected scoring. Then every60 trained-head/
role pair is scored with the original score function and target-count-weighted
aggregation. Original primary scores must reproduce exactly. Saved corrections
must reproduce; raw yaw/contact, clocks, masks, zero padding, populations,
non-position scores and censoring remain unchanged.

Output will retain all per-model before/after metrics, descriptive three-seed
summaries and fixed corrected comparisons (full supervised/direct minus full
JEPA, and no_rgb minus full within condition). No outcome-based selection.
Optimization-seed variation is not independent-maze replication. Training
scores are resubstitution; geometry transfer is development prediction evidence.
Undefined yaw may legitimately differ between models and is retained rather
than rejected as a target-population change. Within each model the correction
must leave all non-position scores/counts exactly unchanged.

Tests83329 closed0:19 passed in1.94s, including altered outputs/metadata/padding,
missing or duplicated model/role/stratum records, assigned seeds/heads, exact
target denominators, analytic three-seed summaries and fixed comparison direction.
Source/hardware preflight93093 closed0:1169 readout sources verified and all1890
queued-native sources unchanged; output absent. Available RAM74,774,224,896 bytes,
artifact free659,951,849,472 bytes, workspace21,356,711,936 bytes;16 physical/32
logical CPUs,3.3% busy, GPU0%. One numerical thread/process,8GiB available RAM and
512MiB output allowance over40GiB reserve. Original input collections and all
source/input/output bindings are checked again after scoring. No RGB/future
tensor materialization, neural forward pass, optimizer or native scene is added
by this readout's scoring phase.

Frozen sources for this running process:
- scripts/read_go2_all_phase_corrected_predictions_v1.py:
  4279ed97ababb5ba4ca877287f4a4d52f0df274e7bea5e013ea9089ceb5fbb08
- lewm/all_phase_corrected_readout_development.py:
  2f2b2792a9428e8dd0c96361d9a265208638ac9f0876bb23d005de46d29a35b1
- lewm/tests/test_all_phase_corrected_readout_development.py:
  fe2cc5b9475a9e5baa9e0d7cb6b2876cdde5e0074fd06d780e951358164a6b7e
- docs/go2_all_phase_corrected_prediction_readout_v1_2026-09-10.md:
  a6cb686cd373431e21067dfbc8f1db0785aae3f75aff3be1a5546c32e187c507

Native25801/PID2636286 and worker2637549 remain alive; six-case waiter14784/
PID2641948 is alive and waiting on that exact original owner. Neither has a
terminal result/failure yet. The native waiter owns the next launch; preserve
its1890-source closure. Detailed completed-model identities and queue contract
are in docs/go2_all_phase_models_and_native_queue_progress_2026-09-10.md.

Next:poll these original handles, verify the readout's eventual exact result,
all36 arrays/60 head-role pairs and complete summaries, and interpret all outcomes
without changing the fixed six native cases. Authenticate the current native
pilot once terminal, then preserve the original automatic six-case execution.
Independent-maze navigation, matched planning/memory controls, realistic sensing,
real-time feasibility and bounded hardware evidence remain open goal requirements.
