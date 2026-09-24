# Expanded-data fitting progress

The previous goal turn completed study/input/schedule preparation. This turn
implemented and launched the actual full-cache fitting benchmark, preserving
the full goal and original native owners. No full scientific fit has completed.

Frozen protocol:docs/go2_all_phase_matched_fits_v1_2026-09-10.md,
SHA9794db0b1ef01a4158f21648f38ffae4d6d8cc6f1f740243cfcd7dbefb4631d5.
It fixes18 fresh matched fits, original three seeds, full/no_rgb variants,
direct/supervised_rollout/jepa conditions,1200x6 schedule and original width32
CPU optimizer/loss settings. Primary subsequent native model remains
seed_2026091001_full_jepa; matched training and sensor-ablation models are fixed
before observing new fitted results. This runner creates no native scene.

New runner and worker sources, now bound by the live benchmark:
- scripts/fit_go2_all_phase_models_v1.py:
  4fc59031972a05f944f572b29b4b6bef2c0034e4e659b71da96809a637689ab7
- scripts/all_phase_fit_execution_development.py:
  03b083bbdd8edcf5fdae093c5589ee27bde4bb38069c1f3ae2cfcf4fef9451ff
- scripts/all_phase_fit_worker_development.py:
  ec5587142d7a0ba4a82f0319fe32a1c08ea0f8c844ad7a0667878dd4629406c9

Preflight25228/PID2632827 closed0 with1144 sources, full original admission,
unchanged original owner source union and sufficient capacity. Available RAM
81,303,887,872 bytes; artifact free668,562,804,736 bytes;16 physical/32 logical
CPUs,3.3% sampled busy, zero GPU utilization. Three-worker requirement64GiB
includes30GiB training,32GiB native and2GiB parent allowances. Those allowances
are launch checks, not kernel reservations; actual worker RSS is monitored.

Actual benchmark root go2_all_phase_fit_benchmark_v1_attempt_001.
Parent handle15290/PID2633175 is live. Launch
961f7fadf5f955f3fecc26c1a9494e71a31abe81575a1c330ebaf4762a53be8e.
Sources1144. Preserve the original process; do not restart or edit its closure.
Benchmark runs three serial subprocess assignments, then the same three in
parallel. Each assignment warms the complete4010-sample private cache once and
performs two fresh20-update JEPA benchmark fits. All six models and every ledger
must match across phases before choosing one versus three full-fit workers.
The fixed full-fit roster is distributed across the chosen persistent workers,
so each process reuses its private cache across its fresh models.

First assignment serial_0/PID2633590 completed successfully. It materialized all
4010 samples, cache7,359,601,120 tensor bytes, warm212.4372640750371s, total
worker286.15192261105403s, peak RSS9,176,956,928 bytes (within10GiB allowance).
Its two20-update benchmark fits completed:
- seed2026091810: initialaf5f32eb00f1058ffd0df2724101cc5f70577a33cc10331854e50706ea84a96f,
  finalb318aaa53c7a1c14cf9292d40e336e7c84e8d90e068a15588696e5df83668466,
  ledger4a883e8ccdfb3bdc49784277dd82f4a83c2082e52423ebcbec6f446d61862c76,
  fit2.0038021630607545s.
- seed2026091811: initial09e359889d98d041959db2e21a77026a1dcbbad2fb608984cf76e57d4bf332d5,
  final7a87d6dc7d01d71a910c4815d726ddaa7eea1079efadcbad5c985462c94bec33,
  ledger1b04c22b78a8d607d6c6b8a960755afbb29a21b9362460828defc525014a289b,
  fit1.899473096942529s.
These are benchmark-only weights, never native candidates or full fitted models.
The parent advanced to serial_1/PID2634114. Parallel equality/speedup and final
benchmark authentication remain pending. Do not select fitting concurrency yet.

Tests83066 closed0:27 launcher/fit focused tests,6.49s. A separate post-fit
admission reader and ten ledger tests are prepared;46847 closed0,2.03s:
- scripts/all_phase_model_admission_development.py:
  fdb39475dc7eab8ce820988b4011425293013cf9f552c781e771902eb8f084f3
- lewm/tests/test_all_phase_model_admission_development.py:
  ba182b461db592837246cf55ae22baeec40ab91d9e2c795a0fd7d192128ab7be
These two later files are not part of the live benchmark closure. The admission
reader reconstructs all21600 exact sample/treatment receipts, complete scores,
evaluation-only snapshots and selected subprocess assignments before a named
model reload. It has not yet been exercised on completed full fits.

Next actions:
1. Poll the original benchmark15290/PID2633175; inspect all six subprocess
   terminals, serial/parallel results, exact equality, full-cache memory and
   measured speedup. Authenticate its final result and preserve any failure.
2. If successful, launch the same frozen runner with --phase fits and
   --benchmark-result-sha256 set to the observed verified result SHA. Output is
   go2_all_phase_matched_fits_v1_attempt_001. No retry/resume or source change.
3. After full fits, run complete all-phase model admission. Derive any new
   training-only correction from these4010 training contexts and their fixed
   schedule; do not reuse the old408-context correction for new model weights.
   Preserve selection discipline and require a separately bound native protocol.
4. Preserve original contact waiter14895/PID2571800 and its native parent2633510.
   Queue37343/PID2551088 has completed0, result1473b2f801991698b41ddd8d97af627e2b1d34621e1cdc6c48886afb03141111.
   Contact launchf61bbd19f01bde603945de9e2feff6c787bd2c62d87942a7a41967e498052096.
   The separately prepared isolated recent-reference direct-flow maze3 native
   remains behind authenticated contact-waiter completion.

Native progress:29 completed raw-audited episodes, zero verified round trips.
Recent-reference maze1 completed with valid raw/strict-visibility evidence but
no arrival/crossing; full details are in
docs/go2_recent_qualified_anchor_maze01_result_2026-09-10.md.

## Benchmark complete; original waiter owns full fits

Benchmark15290 closed0 with result2e74b02b76038f92a8b74256c083cf8ecfbe28d5a5c1bc07c5cbafda214b054f.
All six serial/parallel models and ledgers match exactly. Serial886.0113726989366s,
parallel337.4036113829352s,speedup2.6259688480137835;selected workers3. Independent
source/output/decision check43508 closed0. See
docs/go2_all_phase_fit_benchmark_result_2026-09-10.md.

The single automatic waiter6946/PID2635725 is live,launch
4494e8370968c2643eec51f417d05be658bda747286f3a8b072bd67d9e6e3acd,1146 sources.
It authenticated the original completed benchmark and launched full-fit
parent2636352. Full-fit launchde9fb37f6cd51c609c21fb6ba03b2470167dedad2ecb9166907874bd351d1542,
same1144 sources,workers2636575/2636576/2636577. Its first direct fit reached700
updates and supervised/JEPA600 at the latest log observation. The earlier manual
launch next action is satisfied:do not start another fitting phase. Preserve
this waiter and parent; full admission remains pending.

Contact native/waiter completed,now30 audited episodes and zero round trips.
Contact result90116b248154b9d4731501d5c6cc6a317345101b9f39c6ed5ce712c9baa32986,
waiter resultdc57fe4cd5fb8083ec6ac43ae11d85e6d3e048db62d9b03f06361eb621378536.
Original14895 closed0. The isolated recent-reference direct-flow maze3 native
is live25801/PID2636286,worker2637549,launch
25c34387e44b035f05a4f372ecf70cd337e875de508ba3a9b037ccf741da0eb7,1843 sources.
No additional native scene may overlap it. Detailed contact result:
docs/go2_supervised_commitment_contact_maze01_result_2026-09-10.md.

A separate correction estimator/runner/admission is prepared for the expanded
models. It accounts for all36 motionless contexts rather than rejecting them.
Seven focused tests and an actual4010-context,three-schedule synthetic-prediction
accounting probe passed; no new fitted correction exists yet. See
docs/go2_all_phase_training_translation_bias_preparation_2026-09-10.md.
