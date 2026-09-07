# Independent tracking challenge V1 — draft execution protocol

Status: the narrowly scoped recording/containment resource review is issued;
the final challenge source/configuration is **not frozen or launched**. Native
workload fit remains unproved. This document does not replace the review. No new
observations, tracking qualification, learned-policy result, or navigation
success is claimed. The live twelve-layout learning collector and its planned
36-fit matched study retain scheduling priority.

## Scientific question and fixed population

Does the frozen temporal-anchor continuity candidate preserve reliable measured
motion through turns and brief reference loss on different scene/start
observations, without accepting contradictions, losing current-sensor validity,
or silently continuing after terminal failure? Its earlier 7,433/7,433 versus
7,400/7,433 result used old recordings, including old-controller stopping. It
does not establish the outcome of continuing a physical turn under the candidate.

The population is the exact eight specifications in
`lewm/independent_tracking_challenge_development.py`: two scenes (`offset_niche`,
`unequal_baffles`), two support conditions (friction 1.0 and 0.15), and both turn
directions. These are two scene clusters with four treatments each, not eight
independent mazes. Geometry, starts, appearance, seeds and command schedule are
fixed before collection; missing coverage cannot be repaired by selecting a
different seed, extending a turn, or repeating a trial.

Each trial settles for 750 physics samples, then requests 442 command intervals
of 100 ms, with at most 443 RGB-D/body/gyro observations and 22,850 physics
samples. The tape includes initial hold, translation, braking, outward turn,
translated view, reverse turn and final hold. Requested yaw integrals are not
proof of actual motion. The existing coverage audit requires measured turns in
both directions of at least 150 degrees, two translations of at least 0.20 m,
and the final one-second stopping criterion. All failures remain in denominators.

Acquisition uses the existing CPU Genesis scene, hidden-robot 640×480 ideal
RGB-D, ideal body sensing at 50 Hz and gyro sensing at 500 Hz. Native rendering
near-plane 5 mm and public depth range 0.2–5 m remain distinct. These sensors
are not calibrated hardware or deployment-valid latency/noise models. The
checkpoint gait is unchanged. High-level commands are fixed, validated against
causal packet clocks; no observer, native pose, learned predictor or friction
value selects them. Native state is used for setup, stop-only supervision and
later evaluation, not online command selection.

## Required ordering and evidence

1. Authenticate the original completed twelve-layout supervisor result and the
   completed parallel-execution 36-fit study (exact 43,200 optimizer updates and
   definition `8d8c3456054a284aa83031ea417d8c433beddbcc04a8b47d3164f120bc0ae5d8`).
   This preserves the original scientific factorial but requires the distinct
   `go2_independent_pulse_parallel_study_v1_attempt_001` output. The complete
   reader authenticates all 416 preterminal artifacts, original collection
   receipt, schedules, fit/snapshot bindings and optimizer ledgers. It does not
   deserialize checkpoints, repeat training or aggregate scores at this gate.
   Partial fits, a failure file, changed definitions or promoted evaluation
   claims cannot release the scheduling gate. Check actual owned process
   command metadata; exclude both original and parallel learning launchers and
   the live collector. Do not use an old lock file as a live-process witness.
2. Require a genuine, source-bound native recording/memory review, exact source
   definition hash, interpreter, environment, available memory and free space.
   Both challenge and outside-supervisor roots must not exist. All checks
   precede their creation. The outside keeper starts one exact, fresh user
   service; direct unbounded parent or worker entry is rejected by kernel-scope
   checks. A source/metadata-only preflight runs outside; all native recording,
   sensor replay and raw scoring run inside the limited process tree.
3. Run each trial in one fresh worker process, in declared order. Bind its
   request to the launch, specification, protocol and preceding admitted trial
   receipt. Workers recheck source identity and completed learning receipts.
   The parent records actual exit status, authenticates the worker report and
   admits raw artifacts before requesting another trial. No retry or resume.
4. After all eight raw tapes are admitted, replay both frozen observers on
   private copies of the same complete causal sensor population. Retain rows
   after observer failure; never restart an observer to improve availability.
5. Replay all eleven fixed scenarios per trial, with fresh paired observers:
   nominal, retained-reference absence for 1/10/11 frames, current RGB/depth/
   gyro unavailability, repeated RGB, depth drift, shared gyro bias and an
   anchor/increment contradiction. Fixed onset is frame 84. Authenticate and
   reconstruct all eight base and 88 stress streams before native scoring.
   Distinguish a scheduled intervention from one actually reached and applied.
6. Run raw sensor/contact/geometry/command reconstruction once per trial, then
   score base and stress poses, availability, absolute/incremental/bridge errors
   and actual motion coverage. Reauthenticate the complete sensor phase.
7. Compare actual geometry, starts and value-only sensor prefixes against the
   exact six already-exposed predecessor trials; preserve all 48 comparisons.
   Save the comparison report separately and bind it in the final report.
8. Independently verify the complete result before any adoption decision. The
   launcher itself leaves `independent_result_verification_complete`,
   `full_challenge_pass`, `navigation_qualified`, `real_time_qualified` and
   `goal_achieved` false, even after successful recording and evaluation.

Synthetic reference faults are mechanism tests, not physical occlusion. Bias
magnitudes are not calibrated sensor distributions. A missing initial prefix
is unavailable evidence, not novelty. Different RGB hashes alone are
insufficient: the old matching trajectories sometimes differed by one pixel
despite identical geometry, native motion and depth/body/gyro prefixes.
Box/start/prefix nonidentity is not statistical independence, topology novelty,
or rejection of all globally transformed or repartitioned duplicates.

## Revised prospective resources

The earlier component documents describe historical 3 GiB episodes, 512 MiB
deferred recording and a 36 GiB combined allowance. Those are superseded for
this **unfrozen** challenge by the source preparation here, not retroactively
changed in old experiments.

- Episode ceiling: 5 GiB. Reservations: 443 × 6 MiB frame operations, 128 MiB
  meshes, 8 MiB setup, and 2 GiB deferred recording. Their sum is 5,077,204,992
  bytes, below 5,368,709,120 bytes. This is allocation arithmetic, not proof of
  actual writer sizes or memory use.
- Eight raw episodes: 40 GiB. There are 192 bounded row streams and 48 bounded
  root metadata paths. With 443 rows × 128 KiB per stream and 32 MiB per metadata
  path, the complete envelope is 55,708,745,728 bytes (51.8828125 GiB), within
  52 GiB. The root roster has 240 paths. A successful final report binds 238
  preceding files: the failure report is absent and the final report cannot
  bind itself. Per-episode files are transitively bound through receipts.
- Preserve 40 GiB free storage; require 52 + 40 = 92 GiB available before a
  prospective launch. Recheck while writing and retain partial failures.
- The separate outside-supervisor root has exactly three files: `request.json`,
  `unit.log`, and `terminal.json`, each capped at 32 MiB. Its extra 96 MiB brings
  the complete two-root envelope to 55,809,409,024 bytes (51.9765625 GiB), still
  within 52 GiB, with 24 MiB arithmetic headroom. The inner 240-path roster and
  238-binding successful result are unchanged. Each outside write preserves
  the 40 GiB reserve plus its own file allowance, not all three allowances again.
- Require 16 GiB available memory before work. The prospective service limits
  the entire challenge parent and its native children to 8 GiB charged cgroup
  memory, zero swap, group OOM handling, 512 tasks, and a 48-hour service runtime.
  These are finite attempt limits, **not proof that the native workload fits**
  and not a strict instantaneous RSS guarantee. Exceeding a limit does not
  authorize fewer trials, shorter tapes, a retry, or an unlimited fallback.
  The earlier draft's unsupported retained-worker-memory proof requirement is
  replaced by an explicit recording-ceiling and scoped-failure-evidence review.
- The outside keeper persists the bound request before launch. It drains the
  exact `systemd-run --wait --pipe` handle, retaining at most the first and last
  16 MiB of diagnostics and the actual return code. Any omitted diagnostic bytes
  are counted and disqualify supervised completion; scientific recordings are
  never truncated to satisfy that log cap. A nonzero code is not automatically
  called OOM. A keeper observation error is explicitly child-state-unverified,
  not permission to restart or assume that the service terminated. Its receipt
  cannot survive the keeper itself being killed or storage failing; the root
  and available prefix remain, and exact unit state must then be inspected.
- A zero command exit must additionally authenticate the complete inner result,
  output bindings, original learning receipt and same supervision request. It
  still does not establish independent scientific qualification or navigation.

Native contact evidence also now requires the actual reviewed CPU/float32/int32,
single-environment, nondifferentiable collision configuration and effective
capacity. The recorder checks native error state before each accepted physics
sample and at termination. Changed capacity or overflow makes the tape a
recording failure, without repairing/truncating data or erasing a genuine
physical-stop row. Seven explicit installed Genesis source bindings accompany
the new definition. These checks apply only to this unfrozen successor.

The source investigation is in
[the recording-resource analysis](go2_independent_tracking_recording_resource_analysis_2026-09-07.md).
The required file
`docs/go2_independent_tracking_native_resource_review_v1_2026-09-07.json` is now
issued for the conditional writer ceilings and bounded-attempt/failure-evidence
claims described above. It does not certify that native recording completes or
that a native workload fits. Admission requires its exact current arithmetic,
ten reviewed implementation hashes, seven native-source bindings, four evidence
document hashes and both completed same-source tiny probe receipts. It also
reauthenticates the tiny probes' preregistered definitions and full source sets.
Missing or changed proof cannot be substituted by an evidence string alone.
No final challenge definition SHA or executable launch command is authorized
by this draft. The outside supervisor CLI requires actual reviewed definition
and completed-study result hashes, not placeholders. End-to-end isolated
parent/leaf service fit and group-OOM tests of the new durable keeper completed;
see [the bounded probe result](go2_tracking_keeper_memory_probe_v1_result_2026-09-07.md).
The outside evidence survived the intended tiny OOM. This is not a scaled native
test, peak-memory calibration, full challenge execution or navigation evidence.

## Next steps toward navigation

Freeze and check the complete source/configuration, and independently
verify the new challenge after the original learning study completes. If the
evidence supports adoption, test a separately frozen fresh closed-loop turn and
return. Tracking alone will not fix lower-friction dynamics or the full-loop
100 ms deadline. The latest actual room-return outcome remains 0/3.

The learning track must demonstrate predictive benefit beyond matched empirical,
direct and supervised-rollout controls, then a separate online-rollout benefit
and memory/backtracking contribution on independent maze layouts. Only after
those steps should bounded real-platform experiments assess calibrated sensing,
latency, actuation and safety. This observation challenge is preparation for that
objective, not a replacement endpoint.
