# Independent tracking collection component and partial-evidence accounting

The next part of the independent tracking challenge is implemented: a callable
single-episode collector, an explicit bounded artifact writer, an exclusive
snapshot writer and a separate native-command accounting helper. **No native
challenge collection, new observer replay, JEPA fitting or navigation adoption
has been launched.** These components are not yet a frozen cohort launcher.

Final verification:196 focused/adjacent tests passed, including35 new collection/
persistence/audit tests,50 challenge tests and111 existing adjacent tests.

The [challenge preparation](go2_independent_tracking_challenge_preparation_2026-09-07.md)
still defines the two-scene/eight-trial population and442-interval fixed tapes.
The unchanged original learning-data collector retains resource ownership.

## What now works in source and synthetic execution

`collect_episode` accepts only an exact challenge specification paired with its
fresh trial directory and an explicit protocol hash. It constructs the existing
CPU union-wall/core-ordered sensor scene, installs physical contact identities,
uses the existing checkpoint gait gains and startup checks, then acquires raw
RGB/body/depth/gyro while issuing the fixed causal-packet-valid command schedule.
It does not instantiate an observer, consult a predicted or native pose to select
commands, fit a model or evaluate native motion accuracy. Source/setup/native
terminal measurements are stop-only construction/evaluation evidence.

Acquisition, packet selection and command-dispatch-plus-physics wall times are
recorded separately. The recorded100ms miss flag includes acquisition/selection
only; it explicitly excludes observer and planning computation. Construction,
settling, setup, persistence and teardown timings are also separate. These are
paused-physics development timings, not wall-clock hardware/control qualification.

The artifact writer accepts only an enumerated static/frame roster under one
owned, fresh challenge trial. It rejects overwrites, protected/unlisted paths and
symlinks; serializes JSON/NPZ before writing; fsyncs and binds saved bytes; and
checks hashes and byte counts again at termination. A renderer exception retains
every present file in that operation's explicit roster and disables later capture.
No recursive discovery or source export is used. Complete tape recording always
ends with `TRACKING_TAPE_REQUIRES_RAW_AUDIT`, never a success or qualification flag.

Partial initialization, settling, setup, commands and sensor persistence have
explicit outcomes. An interrupted command retains its actual last physical sample
and `completed=false`. A partial image capture remains in the artifact receipt
even when it never became a complete model/depth/gyro observation. Cleanup and
remaining tape/friction records are attempted after snapshot failure, and secondary
failures are recorded. Scene destruction and Genesis shutdown remain separate.
The new session also destroys a scene if a later recorder initializer fails after
physical construction returned; it avoids destroying it twice after an earlier
constructor failure.

The new snapshot writer preserves the existing sensor manifest schema, sample
dtypes, array members, histories and calibration metadata. Source tests compare
its decoded outputs directly with the frozen recorder methods on empty and
populated synthetic sessions. The declared empty-case difference is an explicit
empty `native_contacts.npz` instead of an absent file. It does not invent contacts
or measurements, fill missing rates, or repair incomplete histories.

The command audit independently checks saved decision/dispatch identities, the
fixed tape, physical clocks, float64 requests, native applied-command slew limits,
phases, partial intervals, missing-stop accounting and partial-path timing claims.
It has no filesystem access. It is **not** a raw sensor/geometry audit and must be
called only after the future authenticated cohort evaluator admits native arrays.

## Bounds and remaining limitations

The component uses a3GiB episode budget and40GiB free-space reserve. Before native
external writes it reserves128MiB for the two meshes,6MiB per frame operation and
8MiB for setup evidence, while keeping512MiB for deferred recording. JSON/NPZ writes
are checked against serialized sizes; final receipts report actual saved bytes.

External renderer writes are guarded by admission and post-write accounting,
**not an OS quota**. A faulty renderer could exceed its reservation before it is
detected. Deferred512MiB is currently an allocation allowance, not a proved maximum
for arbitrary native contact populations or a process-memory limit. Before any
real run, the cohort launcher must validate worst-case declared image, mesh,
history/contact and serialized metadata sizes, enforce the24GiB cohort budget
including launch/receipts/failures, and check storage/resource scheduling. Do not
claim strict external-writer or memory containment from these source tests.

The synthetic complete-tape test uses mock scene/sensors and an injected fixed
selector to exercise all442 command intervals without Genesis. The separate
challenge tests exercise the real sensor-only selector with causal packets.
Together they test composition/interfaces, not actual geometry, dynamics,
observation independence or tracking accuracy.

## Next executable steps

1. Finish the cohort launcher and explicit receipt-admission rules, authenticating
   all eight trial specifications, output populations and frozen sources before
   construction. Stop the cohort on infrastructure/storage failures without retry;
   retain physical stops as negative population members. No partial cohort may
   claim the full challenge passed. Do not run a competing native job while the
   original12-layout collector is live or displace its pending matched study.
2. Integrate paired original/frozen temporal-anchor replay over the same complete
   recorded population. Persist/bind all sensor-only outputs before admitting
   native coordinates. Add exact, fixed stress arms before exposure; longer anchor
   absence and missing current RGB-D must remain different failure mechanisms.
3. Integrate the existing near-field raw sensor/contact reconstruction, actual
   startup/native geometry checks, core raster readbacks, unchanged strict depth
   checks, and independent prefix witnesses. The old short-pulse `audit_condition`
   cannot be reused unchanged: it hardcodes≤2,400 physics samples, action/history
   fields, pulse labels and the old schedule. Reuse appropriate lower-level
   checks without weakening those frozen experiments.
4. After the authenticated sensor phase, combine measured-coverage and command
   audits with per-frame pose errors, all missing/failed cases and independently
   verified actual scene/start/prefix differences. Handle zero/partial acquisition
   explicitly rather than forcing it through a nonempty-sensor audit. Only then
   consider a separately frozen fresh closed-loop test.

The learned contribution remains a separate open scientific question. Complete
the live twelve-layout cohort and the reviewed36-fit JEPA/supervised/ablation
experiment, then establish matched online-rollout and memory/backtracking effects
in independent-maze execution. Room-return success is still0/3 in the latest
physical simulation assay; deployment-valid sensors and hardware remain unproven.

## Evidence from this preparation turn

- Initial component tests31161:69 pass in22.14s. Command-audit integration93802:
  77 pass in23.77s. Snapshot-equivalence and cleanup/storage cases73674:83 pass in
  24.16s. All are synthetic/source-only, with no Genesis execution.
- Eight explicit test files49742:194 pass in57.39s. Internal-write/flush failure
  accounting45159:195 pass in57.52s. Final external-sync accounting33277 terminates
  exit0: **196 pass in57.28s**. The final JUnit has zero failures/errors/skips:
  `.generated/navigation-development-staging.m6MDz1/independent_tracking_collection_verified_v1.xml`,
  SHA256`883d3cc684a85ab58a86d7e25e7495d7fba99b852afba3b516a58f510a82f8f5`.
  This is an adjacent regression, not a new full-repository run or physical result.
- Internal write/flush errors now bind any actual present bytes, not the intended
  complete payload; external synchronization failures account for every readable
  present file before surfacing the error. Neither permits an overwrite/retry or
  a completed scientific audit. Directory synchronization is included for the
  bounded writer's own files. Native external writes are still not OS-quota
  constrained or a claim of crash-proof acquisition.
- In-memory mesh serialization52422 terminates exit0: offset-niche two-mesh PLY
  total6,357,662 bytes; unequal-baffles total6,414,722 bytes. Both fit the134,217,728
  byte mesh reservation. No meshes were saved, no native scene or RGB was acquired.
  Actual native writer identities/sizes must still be checked during construction;
  this does not prove the deferred sensor/contact budget.
- Read-only source guard3083 terminates exit0: all701 completed replay source
  bindings and the786-source learning-study definition remain unchanged. The
  study definition is still
  `3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.
  New collection files and the hardened new challenge session are outside both
  frozen source sets. Subsequent edits concern only those new files.
- `ps`69d768 confirms original supervisor2063013 and l04 child2090031 running;
  child CPU104%, elapsed49m10s. Read-only b0056c finds104/120 l04 prechecks and no
  final audit. The first four layouts remain complete:480 collection trials, not
  navigation successes. No new collection process or matched fit was launched.
