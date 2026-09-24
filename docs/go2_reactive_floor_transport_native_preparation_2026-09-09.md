# Reactive native collector, audit and launcher prepared

Prepared scripts/run_go2_reactive_floor_transport_maze_pilot_v1.py and the
protocol docs/go2_reactive_floor_transport_maze_pilot_v1_2026-09-09.md.
New collector/audit modules preserve the current learned native physics,
sensing, renderer witnesses, geometry, gait/gains/friction,3000sharedticks,
terminal drain, raw checks and physical/strict-visibility outcomes. They
construct the already frozen ReactiveFloorTransportController and perform
complete fresh reactive replay without a high-level world model. The existing
reactive command audit preserves original slew/physical checks with only the
reactive role label. No running/frozen source was edited.

The launcher requires the completed current learned native result, including
raw audits and its prospective prefix, plus reactive prefix
71a5ecd8486d6d8354762c5dc249307cc7d1bf7dc57fe6d1f2df76372d5aaba9.
The saved four decisions must agree with the prefix summary. After the fresh
reactive episode, compare all900physical samples and four paired observations
before the changed command directly against the current learned native episode.
Both native definitions use the same current mission wording; require exact
shared-state equality without normalization. All four reactive decisions must
equal the prospective replay. The new command may physically fail partway
through execution; its actual completion flag and subsequent outcome are
retained rather than borrowed from the old trajectory.

Collection artifacts and the completed raw audit are retained before later
prefix verification. A failed comparison leaves these bindings in the worker
terminal record. Negative scientific outcomes can complete the pilot; an
infrastructure/prefix failure cannot. One CPU native scene/thread,32GiB RAM
admission and10GiBcollection+1GiBpersistence above40GiBreserve. Admission is
refreshed after potentially lengthy input checking; resources are monitored.

Tests7080 CLOSED:25passed2.59s. Cases exercise saved-prefix admission, all900
physical samples, paired public packets and every saved/native decision;
altered prior physics/commands/shared state fail, whereas differing later
physics and a partially completed new command remain actual new outcomes.
Source-scope checks preserve full original collector/audit calculations outside
the explicitly declared controller/model/label changes. Worker tests retain raw
files and completed audit bindings when collection, audit or prefix stages fail.
No synthetic test created a native scene or loaded a high-level world model.

CLI16307 CLOSEDpass. Source/admission preparation80135 CLOSED:1675paths,
all1654live native and1666completed reactive-prefix bindings unchanged. Actual
completed four-decision prefix admitted and its hashes rechecked. No new native
output, preflight or execution. Six new implementation/protocol identities:

- scripts/reactive_floor_transport_maze_episode_development.py:
  5c31aaf9b62b3d3f8b00fe5dccb5c02ccf688c8a5b16019e995131131a337eb2
- scripts/reactive_floor_transport_maze_audit_development.py:
  732cd1cd5035fab2701b3221d2d52678da44c3c30af770327cf49c4bbe904acc
- scripts/reactive_floor_transport_native_prefix_development.py:
  e0ce51b5e29abe696fe0a522f942ceed1e75f6d9cd20f48f45f3ad47509b37f6
- scripts/run_go2_reactive_floor_transport_maze_pilot_v1.py:
  3b6f8a13554cf99bebbcf80cd04fa819334e5408a13f060142163aebc0765c68
- lewm/tests/test_reactive_floor_transport_native_development.py:
  a418ed9fc8f3505632dc64e9ceb6b011a6c316d9f00d8b2a8c52b2d6f1ea1c1f
- docs/go2_reactive_floor_transport_maze_pilot_v1_2026-09-09.md:
  6174d813ba2db3c4add127698e77a9624e5699a944558875a0e1cdba810fc43e

The1675path check additionally binds the unchanged reactive command audit,
completed prefix result report and existing independent-study predecessor
admission helper. It is preparation, not a native preflight or result.

Execution order remains: finish learned native19976and its dedicated readout;
preflight/execute the fixed learned layouts1,2,3 cohort; then inspect hardware,
absence of competing scenes and preflight the reactive pilot using the actual
--learned-result-sha256. Execute only after admission. No reactive readout or
matched result comparison is claimed complete by this preparation.

Live learned worker2447506was confirmed Rl/98.5%CPU at34m04s elapsed/33m35s
CPU,RSS7,397,228KiB;1613completed timingrows through1612. No result/failure
file; same19976handle, no restart. Eleven completed maze episodes, zero verified
round trips. Goal active; this turn made progress by completing native baseline
preparation while the existing experiment ran.
