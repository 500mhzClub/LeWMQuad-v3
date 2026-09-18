# Expanded study and fit preparation

The combined study is implemented and passed actual-data validation. It has
4800 expanded training slots plus all456 unchanged original geometry-transfer
slots. Available populations are4010 training (1664 family/2346 switch) and420
transfer (348 family/72 switch). Transfer global indices4800..5255 map back to
the exact original912-slot indices. Every original transfer row remains exact.
Training and transfer batches route to separate original/admitted readers;
mixed-role or invalid batches fail before either reader is called.

Completed preparation result:
docs/go2_all_phase_study_stream_preparation_result_2026-09-10.json,
SHA c2d632f79588e057d7dbf0777ed26fe4c17f89f13f3a9a9efd8585d12f01a9dd.
Checker15692/PID2631005 exited0. The result binds1130 sources, all420 transfer
sample identities and14 actual training examples: offsets0,1,2,3,4,36,39 for
both sources. Each example matches its completed input/target tensor witnesses
and exact reader scopes. Training and inference inputs agree. All420 transfer
inputs match their original admission; the known command plans remain exact.
Synthetic zero-prediction score accounting equals the original transfer scorer
after index translation. This is an accounting check, not a model prediction
or performance result. Materialization/checking took9.902825506869704s, excluding
the full original authentication before and after. Sample cache usage was zero.
The original queue/contact waiter's1825-source union remained unchanged.

The input admission explicitly verifies the bound scope correction
8fcfdc678b721a84ed53c2545ec55284e2185d7b1e6bc5ca0dc26da4832e2dca
alongside input result ef48950b7987eaf9310ba8124a00c2e6e13c9b84b7cda6a362ac7ddb6ecd63fb.
Original erroneous inherited metadata remains preserved. Full predecessor
collection, target and input authentication executes before and after use;
the corrected-input helper alone is not a substitute for that authentication.

New source files bound in the completed preparation:
- lewm/all_phase_training_view_development.py
- scripts/all_phase_study_stream_development.py
- scripts/all_phase_study_inputs_development.py
- scripts/check_go2_all_phase_study_stream_preparation_v1.py
- lewm/tests/test_all_phase_study_stream_development.py

Separate schedule and fit helpers are now implemented:
- lewm/all_phase_training_schedule_development.py
- lewm/all_phase_training_fit_development.py
- lewm/tests/test_all_phase_training_schedule_development.py
- lewm/tests/test_all_phase_training_fit_development.py

Their source hashes and actual three-seed schedule checks are recorded in
docs/go2_all_phase_training_schedules_preparation_2026-09-10.json.
The original seeds2026091001/2026091401/2026091402 each retain1200 updates,
six examples per batch and7200 draws. Every training trial keeps its original
weight:75 family draws or50 switch draws. Shuffled within-trial cycles cover
every available context before repetition. The complete per-source draw pools
are shuffled and interleaved in random-order family/switch batch pairs, yielding
600 batches per source. All4010 available contexts appear at least once for
every seed; context multiplicities range1..10. No transfer context enters a
training batch. Target values and prediction errors are not used for sampling;
the existing availability masks remain fixed. These are still120 original
training recordings, not4010 independent episodes. Actual command switches in
the recordings remain at their original control phase; reindexing does not
create new switch phases.

Actual schedule hashes:
- 2026091001:5e2ba3b966c3b2b2437d5e3f9c30c2bd1fcfc7291b476237974db55915174d2a
- 2026091401:ab2f636f9810cb2008a02644771d286ea2093a734a8f23f87d12ff50d8bb80a2
- 2026091402:b5a36aa719a2baf685e3fb18aa50fa09f8ec38c86de1bca86b5bfae876e2f66b

The separate fit helper retains the original trainer, loss, normalization,
width32, learning rate0.001, EMA0.99 and full/no-RGB variants. It uses the exact
new schedule and command-plan checks and supports complete4010/420 prediction
populations. The original XY loss already scales by0.06m. A benchmark mode is
bounded to20 updates; no benchmark or successful optimizer update was run here.
Final prediction rejects untrained/incomplete fits. Future runner admission
must bind the selected input variant, model, schedule and complete checkpoint
ledger; this helper does not itself define the experiment/native roster.

Tests:27 new study tests passed independently; the combined study and original
input/fit suite passed67 tests (78949,14.73s). Ten schedule tests passed15075
in6.63s and eight fit-admission tests passed4572 in6.26s:85 focused tests total.
Fit tests construct fresh trainers only to verify rejection before input reads
or optimizer updates. Actual study construction91374 exited0; actual schedule
check71575 exited0. No fitted model or checkpoint resulted.

Hardware before the full preparation:76,738,330,624 available RAM bytes,
668,582,301,696 artifact free bytes,32 logical/16 physical CPUs,3.4% sampled CPU
utilization and zero GPU utilization. The read-only preparation used one CPU
thread and no sample cache, leaving32GiB allowance for the original native work.
This does not select fitting concurrency: the full cache has7,359,601,120 tensor
bytes before Python/process overhead, and useful throughput still needs an
actual representative benchmark.

Next: freeze the prospective matched18-fit roster (three original seeds,
full/no-RGB, direct/supervised_rollout/JEPA), training and benchmark output
ownership, and subsequent native assignments before fitting. Build and verify
the launcher/accounting, benchmark useful CPU/process/GPU options as applicable,
and select concurrency from measured throughput/equivalence and headroom. Preserve
all completed source bindings; use new successors for execution revisions.

The original queue37343/PID2551088 and contact waiter14895/PID2571800 still own
native launches. Recent-reference maze1 parent2623245/worker2624197 remain live
in raw audit at the last check. Collection ended without arrival; the final
pilot/audit/prefix/worker-terminal results were absent. Completed raw-audited
count remains28, zero verified round trips. The separately prepared isolated
recent-reference direct-flow maze3 pilot stays behind authenticated completion
of the original queue and contact waiter. Storage is not blocking this work:
the artifact volume has approximately623GiB free; workspace has20GiB free.

Final native-state update: the original worker has now written the maze1 raw
audit5cb5c958dfb9ed6c181a157ab2282d9d44173fd5285f9c72a7816c7d66daa47e
and prefix comparison6f9bd7b85f0198c56ddf0ea1dc71b1a0b3d65c14559be08eacb982d7d550f1e8.
The audit reports raw sensor, model replay, command, unchanged-model and strict
visibility passes; hard failed frames[],1556 frames/3112 captures witnessed,
no arrivals/crossings/round trip. The prefix reports646 frames/33000 physics
samples/642 model forecasts exact,635 complete original decisions exact,
first reference/observed decision difference635, first changed command645,
and the left-turn intervention completed. Candidate decisions match prospective
replay after the reference change. Following physical outcomes are not inferred.
Worker-terminal and final pilot result remain absent; full original pilot/queue
authentication is pending, so completed authenticated count stays28 for now.
