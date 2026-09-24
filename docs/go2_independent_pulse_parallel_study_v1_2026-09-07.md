# Prospective parallel execution revision of the independent-layout study

Status: implemented and validated; not launched. The reviewed definition digest
is recorded separately after this protocol is finalized. The original collection
continues unchanged. Do not run both sequential and parallel studies.

This revision changes the scheduling of independent fits, not the scientific
question or its sample size. It inherits the complete original study definition
`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.
The sequential runner and its sources remain unchanged reference evidence.

## Fixed science

All original 12 layouts, roles, exclusions, prefix matching and coverage gates
remain. All 36 fits use the original three seeds, four information treatments,
three objectives, 1,200 updates, batch six, latent width 32, optimizer, EMA,
regularization and actual-horizon targets. The final snapshot alone is scored,
with inference batch six. No partial cohort, best seed, checkpoint search,
different GPU arithmetic, retried fit or favorable subset is introduced.

Single-thread deterministic CPU execution is retained **within** each fit.
Four spawned processes run independent fits concurrently. Each process owns its
own stream, trainer, optimizer and explicitly named fit artifacts. The parent
publishes one fixed schedule per seed, authenticates complete worker artifacts
and aggregates in the original seed/treatment/objective order. Completion order
does not affect schedules, initialization, baseline fitting or comparisons.

The new launcher is `scripts/run_go2_independent_pulse_parallel_study_v1.py`.
It requires the actual original all-twelve-layout terminal digest and its own
reviewed prospective definition digest. It rejects any existing original or
parallel study attempt. The exclusive output is
`go2_independent_pulse_parallel_study_v1_attempt_001` in the existing owned
development artifact root. No legacy or sealed inputs are added. These entry
checks are not a system-wide lock against an unrelated operator launching the
unchanged original executable later; the controlled workflow launches only one.

## Resource and failure behavior

The maximum is four concurrent submitted fits. Spawn, not fork, avoids inheriting
live library thread state. The parent requires at least five allowed logical
CPUs and 32 GiB available RAM, and records actual CPU affinity, RAM and free
space before launch. These are capacity observations, **not enforced process
memory limits or proof that the real workload fits**. Measure actual utilization
and resident memory when the study runs. Existing collection must finish first.

Each fit has a static 192 MiB artifact allowance, including 1 MiB reserved for
failure evidence. Parent artifacts have 1 GiB: 36 * 192 MiB + 1 GiB = 7.75 GiB,
within the original 8 GiB budget. Exclusive, disjoint filenames prevent workers
from overwriting each other. All original snapshot and per-file bounds remain.
The original whole-allowance preflight and 40 GiB free-space reserve remain;
these checks cannot reserve space against unrelated external writers.

On a worker or parent-receipt failure, stop dispatching new fits; drain already
running fits and retain their artifacts. No resubmission, changed seed, recovery
checkpoint or automatic fallback. A killed process may leave a partial ledger
without a complete failure record: retain it, do not infer completed updates
beyond durable evidence. The parent must not publish complete scientific scores
from a failed/partial factorial. Timing fields and artifact identity differ from
the original; numerical parity must compare tensors, draws and metrics, not
expect byte-identical launch metadata or wall-clock measurements.

## Validation and remaining adoption gates

Synthetic integration compares the actual four-process fit implementation with
the original sequential runner: every per-update loss/gradient/model hash,
initialization, training draw, final prediction array and aggregate score, for
all twelve treatment/objective combinations in one seed. This establishes only
the exercised synthetic case, not real-cohort runtime, convergence or science.
An additional test now covers all 36 fits (three actual seeds, production width
32 and batch six) for six updates each, compared against a separately executed
original sequential reference. Final model hashes, prediction arrays and all
nine aggregate score reports match exactly. This is 216 synthetic updates per
execution mode, not the actual 43,200-update experiment or a throughput proof.
Fault tests cover rejected jobs, duplicate attempts, failed fits, parent rejection,
artifact ownership/budget, invalid receipts and retained partial ledgers.

Full-length synthetic parity also passed: four distinct fits (full direct, full
supervised rollout, full JEPA, no-RGB JEPA), each with the actual 1,200 updates,
batch six, latent width 32 and seed 2026091101. Compared serial and spawned
execution for all per-update records, complete model/optimizer snapshots and
all final prediction arrays: exact equality. Measured serial wall time was
329.821481 seconds, parallel 128.188335 seconds (2.572945 times faster), including
spawn and artifact I/O. All four workers were used; their summed fit CPU time
was 397.536859 seconds, or 3.101194 CPUs averaged over parallel wall time.
Each process reported lifetime peak RSS 1,414,549,504 bytes (about 1.32 GiB),
not a per-fit allocation measurement or enforced memory bound. Original live
collection and some adjacent synthetic tests were concurrent background work;
this was not a controlled hardware benchmark or a GPU comparison. Actual cohort
image/metadata loading was not exercised, so real-study speedup remains unknown.

The parent additionally checks the exact snapshot configuration and binding,
reads every optimizer ledger row against the published schedule/treatment, and
requires the last ledger model hash to match the saved final model. Rebinding a
corrupt ledger does not bypass those accounting checks. This does not recompute
training numerics or independently reconstruct raw scientific predictions.

Initial run: 25 passed, one failed because a test incorrectly required all four
processes to receive these very short fits; three did. All preceding per-fit
numerical comparisons had matched. The corrected assertion requires actual
multiple-process isolation within the four-process bound. Expanded run:
27 passed in 78.51 seconds, JUnit
`.generated/navigation-development-staging.m6MDz1/independent_pulse_parallel_study_expanded_v1.xml`.
This suite includes complete real spawned execution and sequential comparison,
but no real dataset training or native simulation.

Final adjacent run: 107 passed, one long test deselected, 158.54 seconds;
separate full-length run: one passed, 460.96 seconds. See
`.generated/navigation-development-staging.m6MDz1/independent_pulse_parallel_final_adjacent_v1.xml`
and `independent_pulse_parallel_full_length_v1.xml` in the same directory.
The initial 64-update timing test also failed only on its incorrect exact-four-
workers assertion after numerical checks matched; the full-length test now
records actual worker use within the concurrency bound. The final reader tests
reuse a complete synthetic factorial for ten isolated corruptions instead of
retraining it for each mutation.

The distinct read-only scientific reader is implemented and tested:
`scripts/read_go2_independent_pulse_parallel_science_v1.py`. It authenticates the
exact 36-fit terminal, all 416 preterminal artifact bindings, job/schedule/ledger
accounting, original collection receipt and source/config definition before
aggregating the existing 27 scientific contrasts. No checkpoint deserialization,
raw collection loading, inference or scientific promotion occurs in the reader.

Before downstream native tracking: explicitly adapt its entry gate to this
distinct root/definition and revalidate the affected source-bound review. That
pending integration is not permission to launch either study on partial data.
The learning launch still requires all twelve original receipts, hardware/free-
space reassessment and the reviewed definition. Do not impersonate the
original sequential result. A negative JEPA comparison remains valid and does
not block work on actual local execution and navigation.
