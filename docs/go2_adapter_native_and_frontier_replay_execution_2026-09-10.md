# Active adapter native batch and separate frontier replay

Latest snapshot: 2026-09-10 11:04 UTC. Goal remains active and incomplete.
The hold-reorientation raw replay completed initial input admission and is now
reconstructing the actual paired controllers. Launch
`f3cdd9b0d97cb56328adc6f60edac2f5a62161a81bc91bf882117f7d1d5207f2`
binds 1,943 sources; a bounded inspection saw 41 complete rows (0–40), still
matching all recorded comparisons with no changed request. Child PID 2672447
remains live; full completion through the changed request at 405 is pending.

A separate unchanged-controller profiling process, PID 2673728 (created
1789037950.04), tool session 27421, is performing full original input admission.
It will profile fixed windows 3–12 and 395–404 while reconstructing all original
decisions through 404. Ten tests passed and source preflight verified 1,963
bindings. It launches no native scene and changes no policy. See
`go2_adapter_controller_profile_execution_2026-09-10.md` for exact scope and
process identities. All previously scheduled experiments retain their order.

The preceding 10:50 audited first-case findings remain current:
The first corrected JEPA worker has completed and passed its raw reconstruction,
model/command replay and command audits. It remains a navigation failure: no
arrival or round trip, two declared open edges crossed, zero native contact
samples, and one auxiliary strict-visibility failure at observation 1173.
All 406 planned hold-intervention prefix observations passed strict checks.
Independent verification is recorded in
`go2_all_phase_adapter_full_jepa_maze02_result_2026-09-10.md`. The completed
audited count is now **38**, with **zero verified round trips**.

Both automatic handoffs advanced: full supervised-rollout worker PID 2672443
(created 1789037034.85) is collecting the second case, with 262 observations at
inspection; raw replay child PID 2672447 (created 1789037036.8) is performing
original full input admission using worker terminal
`617056f19ba4928aa9ff7738616947e6e63a387cc6046353e30617ce50afa57e`.
It is owned by the existing CPU waiter, so do not launch it again manually.
The old first worker PID 2662101 ended normally. The existing frontier and
subsequent hold-native waiters retain their order. Fixed-snapshot timing showed
median observation-plus-control latency 1.809 s and p95 2.358 s, far beyond the
100-ms interval; no real-time claim is supported.

The preceding 10:39 launcher registration remains current:
The single-case hold-reorientation native launcher is implemented and its
automatic handoff is registered behind both the raw-replay waiter and existing
frontier waiter. New native waiter PID 2671835, tool session 88131, launch
`e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6`,
binds 1,960 sources and owns the subsequent native launch. No new native scene
has started. Twenty-two launcher/input-chain tests and five waiter tests passed;
final source-only preflight passed 1,957 bindings. See
`go2_hold_reorientation_native_execution_2026-09-10.md` for the exact process
identities, preserved execution order, validation limits and next steps.

The prior 10:19 raw-replay handoff remains live and waiting:
The full hold-reorientation raw replay is implemented and automatically queued
behind the completed audit of the exact first adapter JEPA worker. New CPU-only
waiter PID 2669739, tool session 25488, launch
`969c9153d9bebbf7cc7ba24e56ed76a912edf33f0a13aeaacc243cdcd2862e2c`,
owns this launch; do not start it again manually. The waiter binds 1,946 sources;
the raw replay output does not yet exist. Thirty-two new focused tests passed
across comparison, synthetic replay wiring and owner-wait checks. See
`go2_hold_reorientation_raw_prefix_execution_2026-09-10.md` for exact identities,
validation scope and next steps. The existing frontier waiter retains the next
native experiment; this new handoff launches no native scene.

The prior 10:06 collection and saved-prefix findings remain current:
The first adapter JEPA collection has stopped after 1,529 observations and
1,528 completed commands, with 77,150 physics samples and ten terminal zero
commands. Its terminal is
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`; no physical or
acquisition stop was recorded. Its worker is live performing the remaining
audit/finalization, and no worker audit/terminal has yet been admitted. The
completed audited count remains 37, with zero verified round trips.

The first-1,000-row hold diagnosis and independent reconstruction completed:
595 discretionary holds, all highest-utility among raw-eligible actions;
566 admit only left turn while higher-utility right turn is path-vetoed.
See `go2_adapter_hold_prefix_result_2026-09-10.md`.
A separate hold-reorientation controller passed 22 focused tests. Its immutable
saved-selection check and independent reconstruction locate the first changed
request at 405, hold to left turn, with all prior selections and boundary
forecasts/scores/vetoes identical. See
`go2_hold_reorientation_saved_prefix_result_2026-09-10.md` for exact identities
and required next raw replay. This is not a raw-controller or navigation result.
No observation after that changed request was consumed by the candidate check.

All six relevant active/completed source maps were reverified unchanged. The
frontier waiter still owns the next native launch; the hold candidate has no
native launch queued. Independent layouts remain unexecuted. The preceding
09:23 snapshot and execution details follow for chronology. The preceding
status turn verified terminal completion of the original six-model cohort and
waiter, changing the next action from waiting to fresh prospective execution.
There are 37 completed audited development episodes and zero verified round trips.

## Adapter native batch

New source: `scripts/run_go2_all_phase_adapter_maze02_matched_native_v1.py`.
Protocol: `go2_all_phase_adapter_maze02_matched_native_v1_2026-09-10.md`.
Output owner: external artifact root,
`go2_all_phase_adapter_maze02_matched_native_v1_attempt_001`.

Process PID 2659758, tool session 53094, completed input admission and launched
the first fixed model worker (PID 2662101; resource tracker PID 2662100).
Launch SHA-256:
`97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a`.
Parent and worker are live. No worker completion or new navigation result has
been admitted yet. Do not restart the original process because a verifier is quiet.
Subsequent live-stream inspection in this turn found 97 complete observations
(0–96), no terminal failure, and exact equality of the first four complete native
decisions with adapter V2 replay. The latest request was the learned left arc.
This confirms runtime progress past the original observation-3 wrapper failure;
full paired physics/public startup, command execution and raw audits remain pending.
The latest inspection reached 646 complete observations (0–645), no terminal
failure and no observed goal arrival. Its observed goal distance was 3.354 m;
this is an observed estimate, not independently audited native progress.
The command uses the existing Genesis Python environment, deterministic hash
seed, PYTHONPATH `.:lewm_genesis:lewm_worlds`, one OMP/MKL/OpenBLAS thread,
`--correction-wait-result-sha256
4bf13d2e00fb318fa836d02bad93784fbb1a9c5ccef8792bd19dcfd503f657a0` and
`--native-result-sha256
330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723`.

Source/resource preflight completed without creating output: 1,908 bound sources,
about 612 GiB artifact space and 76 GiB available RAM at preflight, original
owners ended. The launcher separately authenticates original failed cohort,
completed waiter and adapter V2 raw replay; reexecutes complete original input
admission; loads six fresh exact assigned adapted states; admits one native
worker at a time. Current sources are frozen by the running process.

Validation: tool session 98061 exited zero, 71 tests passed in 2.95 seconds across
new native evidence/startup checks, the actual model-interface adapter and the
existing study/startup/launcher suites. New checks require exact complete first
four prospective decisions, all forecasts, original physical/public startup and
the first changed command actually completed. The underlying controller,
collector and raw audit source hashes were independently checked unchanged.

## Separate reached-frontier replay

New source: `scripts/replay_go2_reached_frontier_maze03_prefix_v1.py`.
Protocol: `go2_reached_frontier_maze03_prefix_v1_2026-09-10.md`.
Output owner: external artifact root,
`go2_reached_frontier_maze03_prefix_v1_attempt_001`.

Process PID 2660458 has ended; tool session 57528 exited zero with a complete
raw controller replay and final input verification. Launch SHA-256:
`28380c18617abc59c7019a90786204ae6b3281842b70ca9dcc0f8de0b93b755f`.
An initial bounded stream inspection found 53 complete rows (0–52), 50 exact
forecast-bank comparisons, no reached frontier and no changed command yet.
The subsequent inspection reached 83 complete rows (0–82), still with exact
normalized original decisions and no frontier-triggered command change.
Final result: 135 observations, first reached frontier and changed request at
134, 132 exact raw forecast comparisons, right turn to left turn, no later
recorded observation consumed. Result SHA-256:
`00579b00e70179bd1687a44ade0ed7282d8f6d3a742bbb7727a1a18733d14d25`.
Independent reconstruction and all 135 raw packet fingerprints passed in
session 71625. See `go2_reached_frontier_maze03_prefix_result_2026-09-10.md`.
It uses the same deterministic CPU environment, no arguments, and creates no
native scene. Do not restart a quiet original verifier.

Preflight completed: 1,873 bound sources, all completed maze3 artifact bindings
verified, no models loaded or output created during preflight. Tests in tool
session 45321 exited zero: 29 passed in 4.12 seconds, covering the original
frontier transition and new exact observed-state/forecast/boundary comparison.

The replay reconstructs the original RecentQualifiedDirectFlowController and
the frontier candidate on identical raw packets and fresh old assigned JEPA
models (state `4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`).
It checks unchanged observed evidence, contact memory, accumulated cells,
executed residual history and raw forecasts. It stops at the first changed
request or terminal and consumes no later counterfactual packet. The expanded
adapter and frontier intervention are not mixed.

## Prepared frontier native verification and I/O

Added `scripts/reached_frontier_native_prefix_development.py` to reconstruct
every saved original/prospective comparison, packet fingerprint, counter and
first-divergence boundary. Its physical comparator requires identical raw physics
through the preintervention observation, identical actual prior commands and
public packets, complete candidate decisions equal to prospective replay, and
an actually completed boundary command. A nonterminal hold remains eligible as
an intervention; the checker does not select only movement outcomes.

New isolated collector and audit:
`scripts/reached_frontier_maze03_episode_development.py` and
`scripts/reached_frontier_maze03_audit_development.py`. They retain the original
complete collection/raw audit calculations and substitute only the reached-frontier
controller plus intervention/status receipts. The original sources were checked
against their completed launch bindings before these narrow derivations.

Tests: session 10793 exited zero, 18 prefix/boundary checks passed in 2.30 seconds.
Three source-structure checks passed in 0.13 seconds, proving original collection,
artifact enumeration and full raw audit calculations unchanged after normalizing
only controller identity and the declared receipts. These synthetic/source tests
do not establish physical frontier success. The complete frontier native launcher
and queue admission are now implemented and source-preflighted (1,926 paths).
The combined native suite passed 42 tests in 2.54 seconds (session 79347).
The queued handoff passed six tests in 2.15 seconds (session 61414). Both existing
source closures (1,908 and 1,873 paths) were independently verified unchanged
after preparing these new files.

The active handoff is PID 2663938, tool session 23814, queue launch
`68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74`, output
`go2_reached_frontier_maze03_native_wait_v1_attempt_001`. All 1,929 queued sources
are frozen and independently verified unchanged. It has observed the completed
prefix and is waiting on original adapter batch PID 2659758. It owns the one
future frontier native launch; do not start a duplicate or change queued sources.

## Next actions

Poll adapter batch session 53094 and frontier handoff session 23814 and inspect
their own output roots. Native startup, audit and parent completion are still
required. Preserve any failure and do not count pending runs as completed
navigation episodes. The original frontier replay session 57528 is complete;
its separately queued physical intervention must wait for the adapter batch.

The eight-layout/four-arm independent comparison is still unexecuted. Its earlier
factory remains preserved with the incompatible expanded wrapper. Separate adapter
factory/collector/audit versions are now implemented and passed actual four-arm
startup compatibility on old development packets, result
`cd0af02bdbe978e714e55c1c0253f1fa01df88c4554eeea887deae84e40ae6db`.
See `go2_independent_adapter_factory_startup_result_2026-09-10.md` for scope,
identities and validation. No independent layout was rendered or observed.
The final population launcher and development-result admission remain pending.
Reliable outbound/return navigation,
independent-layout evidence, matched planning/memory comparisons, real-time timing
and bounded hardware evidence remain unproven. No further cleanup is needed for
these jobs; workspace free space is about 20 GiB and artifacts use the external
volume. No sealed material, existing failures or model files were modified.
