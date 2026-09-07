# Continuous startup state and clearance provenance implemented

The new continuous owner carries one depth/gyro observer and one fused memory
through startup, the measured stopping tail and a navigation-consumer handoff.
Recorded-data replay succeeds. This is new software/replay evidence, not a new
physical continuation, full mission, learned policy or JEPA contribution result.
The original startup trial and full-mission **0/2** result remain unchanged.

## What changed

`lewm/continuous_startup_handoff_development.py` owns the existing startup
controller, relative observer and memory from the first admitted frame. It uses
the frozen startup controller until its successful terminal, then updates the
same memory from every subsequent 100-ms stopping-tail observation. The terminal
startup controller is never called again. No pose reset, new initial velocity,
dropped tail interval or replacement memory is introduced.

The handoff checks exact epoch/episode/cadence, actual command acknowledgements,
unchanged sensor and fusion contracts, observed conflicts, tail-region validity
through the stopping horizon and inferred speed. Three tail observations are
required. The last must have rank-3 depth, inferred speed <=0.05 m/s and all 50
new fast-gyro samples in its final 100-ms interval <=0.1 rad/s. Failures latch
zero output and prevent stale-state export. This does not substitute for the
physical wrapper's native contact, speed and stopping audit.

`READY_FOR_NAVIGATION_CONSUMER` exposes a copied state snapshot, not an approved
action. Subsequent observations keep the same owner; there is no downstream
navigation planner in this class. The snapshot distinguishes supplied initial
non-floor clearance from observed current-posture evidence, reports setup expiry,
retains the initial-velocity contribution and explicitly denies ground-support,
future-sweep, navigation-action and real-time qualification.

`lewm/setup_clearance_partition_development.py` partitions an expanded box into
at most one region-covered intersection and six residual boxes requiring sensor
evidence. It does not fill a bounding hull, call residuals clear or grant contact.
The original whole query remains subject to observed vetoes. Expiry through a
prospective horizon, wrong identity or observed conflict cannot preserve the
supplied clearance. A numerical guard shrinks the prior region, never expands it.

The current-body adapter binds this partition to the owner's measured joints,
fused pose and existing position/orientation error scales. It adds 4-cm geometry
padding and an orientation-displacement allowance derived from the all-posture
radius. This remains conditional on the uncalibrated scales and current posture;
neither joint history nor current posture is promoted to future gait evidence.

## Actual recorded-data result

The new owner consumed all 15 observations from the existing startup trial,
reproduced all 12 startup decisions and all 15 relative-observer records exactly,
and integrated 700 gyro intervals from 1.5 to 2.9 s. The same observer and memory
objects remained in use. State transitions were successful startup terminal at
2.6 s, two intermediate tail observations at 2.7/2.8 s, and handoff readiness at
2.9 s. No physical commands were executed by this diagnostic.

The final combined position scale is **0.0325800851 m**. Its decomposition still
contains **0.0119999977 m** from the initial-velocity prior and 0.0205800875 m from
the inherited proxy. Full-rank depth removed the prior's velocity contribution,
not its already accumulated position contribution. The initial epoch remains
1.5 s; retained views remain anchored at 1.5/1.7/1.9/2.1 s with a separate latest
view at 2.9 s.

With a 0.0726636451-m current-body expansion, all **27** current primitives lie
conditionally inside the supplied starting non-floor region. Sensor-observed
current-posture clearance and contact-candidate sets remain empty. This resolves
the *representation* of initial blind space: it is a labelled setup condition,
not a discovered map or measured support. It does not resolve future movement
outside that region. The original region expiry remains 3.5 s.

The diagnostic verifies the original 385-source/inherited-input/74-artifact and
16-native bindings plus three exact result identity witnesses, and separately
binds its five new development sources before and after replay. It does not
alter the original trial, its controller, result, audit or protocol.

## Verification

- 44614: 14 handoff tests passed in 6.14 s.
- 85716: 32 combined focused tests passed in 7.24 s.
- Review added expired-region continuation and in-region observed-veto tests.
- 30025: **1,726 tests across 140 explicit files**, passed in 105.60 s.
- 36946: preliminary same-owner recorded replay completed, exit 0.
- 95120: final recorded replay plus current-body partition completed, exit 0.

No source was edited concurrently with its tests or diagnostic. The final replay
ran alongside the expanded regression; no new timing or real-time claim is made.
All listed process handles are terminal. No new physics or training was launched.

Tests include dropped/duplicate/wrong-epoch/wrong-episode frames, privileged
packet fields, depth/gyro faults, wrong command acknowledgements, a hidden
between-slow-sample gyro spike, incomplete tails, startup failure, stale/faulted
exports, mutation isolation, exact box-volume/random-point coverage, degenerate
boxes, expiry, numerical boundaries and observed-conflict vetoes. A continuing
observer after region expiry remains usable but supplies zero region clearance.

## Next implementation, toward the complete task

Connect residual boxes to observation-bound clearance queries, retaining a
whole-query veto even inside supplied initial clearance. Keep ground support as
a separate explicit model/evidence channel. Then provide action-conditioned
body/foot motion and braking predictions rather than reusing current posture or
the oversized startup sphere as the corridor footprint. Test predicted-versus-
executed motion errors and failure cases on fresh bounded development data;
empirical prediction coverage is not a universal safety certificate.

Wire that action/evidence interface and this continuous owner into the existing
complete discovery/marker/return controller in a fresh declared mission. Include
its actual stop tail and source-bound raw replay. Do not launch more copies of
the successful startup assay in place of a complete mission. Preserve the
[remaining matched JEPA, multistep, memory, independent-layout, latency and
hardware requirements](go2_post_startup_continuous_navigation_plan_2026-09-06.md).
The full scientific objective is still active and unachieved.
