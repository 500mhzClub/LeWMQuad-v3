# Matched-model active-view native comparison V1

The six-model executed-transition diagnostic found a large full-RGB JEPA hold
error on the existing active-view trajectory. This new prospective experiment
tests closed-loop consequences for all six previously fitted models; none is
selected as the winner before these runs. The original failures stay frozen.

Use the final `seed_2026091001` full/no-RGB × direct/supervised-rollout/JEPA fits,
each on both known family layouts 052 and 039. Each of the twelve assignments
starts a new process, native scene, observer, map and controller. Shuffle the
complete assignment list once with seed 2026091101 before launch. This is an
order randomization, not twelve independent mazes or independent model seeds.
No original active-view case is resumed or retried. Re-running the full-JEPA
arm here is an explicitly declared matched repeat in this new study.

Change only the fitted snapshot, its trained prediction head and input variant.
Preserve the original six action plans, five-tick commitment, scan targets and
scan utility, waypoint/map rules, persistent articulated-surface vetoes, observer,
public sensing, 1.2-m goal, 240-tick navigation budget, quiet-arrival criterion,
zero drain, physical guards and evaluator-only native goal audit. Keep full
public sensor acquisition in the no-RGB arm: its world-model tensor treatment
removes RGB, while the shared visual observer/map still use it. Consequently
this ablates RGB in the fitted predictor, not RGB from the entire robot system.

Require the existing complete six-fit ledger/schedule/input/raw-score admission
at parent preflight. Bind each snapshot and the full fit artifact map in the
launch. Workers verify launch/source/native/input bindings and all fit artifacts,
reload only their explicitly assigned final snapshot, and replay all raw
sensor/observer/map/model/command decisions with a fresh evaluation-only reload.
Parent admission is inherited through the exact launch identity; no worker may
discover, choose, train, resume or substitute a checkpoint.

All twelve cases run serially, with single-threaded CPU kernels and a fresh
worker per case. This retains the established uncontended timing measurement;
independent scenes could run concurrently, but their contention would change the
timing comparison with the original controller. Existing native workload timing
supports this execution choice; the two-thread diagnostic inference benchmark
is not a native-throughput benchmark. Inspect live hardware/resources at
preflight and every approximately fifteen seconds. Require 32 GiB available RAM,
plan 16 GiB output plus the existing 40 GiB storage reserve, and retain the
runtime storage stop. These are capacity checks, not OS quotas. Physics pauses
during computation; no real-time or hardware qualification is claimed.

Continue after scientific failures, including goal failure, contact or visual
tracking failure, provided raw audits complete. An infrastructure or raw-audit
failure terminates the cohort and preserves completed cases plus the explicit
unlaunched list. No automatic replacement, resume or parameter adjustment.

Primary descriptive results are native verified goals per model (two cases),
with contact, visibility/depth checks, terminal/minimum goal distances, observer
failure and complete-loop timing. Intermediate scanning or waypoint use is not
goal-reaching. No significance or independent-maze generalization claim is
supported. Full objective requirements, including physical backtracking,
matched non-predictive planning/memory baselines, unseen mazes, realistic timing
and bounded real-platform evidence, remain unchanged.

Nine focused tests cover all six trained head/input assignments, rejection of
undeclared treatments, exact unchanged scan/waypoint selection arithmetic apart
from head/variant, sensor-failure latching, and identical native-goal/actuator
audits. This source is a new named development implementation; frozen sources
and all sealed material remain untouched.
