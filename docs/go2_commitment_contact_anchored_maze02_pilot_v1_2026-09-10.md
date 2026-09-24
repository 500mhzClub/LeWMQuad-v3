# Prospective supervised commitment-contact anchored maze-02 trial

Execute one fresh reused development maze-02 case
`full_supervised_commitment_contact_anchored_maze_02`, using the unchanged
assigned full-input supervised-rollout expanded-data model
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
The native candidate is `CommitmentContactAnchoredController`. It applies the
already checked 100-ms contact-cost scorer only to ordinary intermediate
waypoint selections after the original anchored selection/recovery chain.
Active residual recoveries, final goals, views and nominal-clearance reentry
retain their original behavior. Coefficient 1.2, raw forecasts, all eight
800-ms geometry checks, robot, sensors, mapper, tracker and mission are unchanged.
The original 800-ms contact score remains evidence but its cost is not charged
in ordinary choices. Neither contact score is a calibrated probability.

The earlier old-model contact-horizon trial ended on missing current visual
evidence at observation 140 without a goal or edge crossing. The new expanded
supervised baseline collection exhausted its budget with no forward command;
its audit is pending during this preparation. Preserve both negative results.
There is no assumption that the new candidate will retain visual tracking,
clear obstacles, reach a goal or return.

Runner: `scripts/run_go2_commitment_contact_anchored_maze02_pilot_v1.py`.
Exclusive output: `go2_commitment_contact_anchored_maze02_pilot_v1_attempt_001`
under the existing recovery-storage navigation artifact root. No automatic
retry, resume, replacement, model selection or outcome-dependent layout change.
Source-only preflight creates no runtime output and performs no model inference
or native input admission. Full preflight can run only once both required
completed result identities are available.

Require completion of both exact existing waiters:

- Raw contact-score replay waiter, launch
  `ca3027a13c67b92e2e181d92a55377ae102fcfd0bd007544ba87cf5dea61c880`.
- Previously scheduled hold-reorientation native waiter, launch
  `e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6`.

Pass their exact result SHA-256 values as `--raw-prefix-wait-result-sha256`
and `--prior-native-wait-result-sha256`. Reconstruct both handoffs, authenticate
the completed prior native trial and reexecute its full input verifier. Require
the completed four-observation fresh-model contact replay, reconstruct every
comparison and public packet, and repeat its original supervised worker input
admission. Link that original worker to the exact same completed six-case
adapter batch and correction/model assignment. Bind all five direct predecessor
artifact groups and their source identities. Recheck complete admission after
the new native trial. Scientific success of the predecessors is not required;
completed original raw/physical audits are required.

Thus the original native order remains: six-case adapter batch, reached-frontier
pilot, hold-reorientation pilot, then this candidate if its raw replay passes.
Require no live native runner/worker immediately before output creation. Use one
fresh spawned native worker, one numerical/OpenCV thread and one native scene.
Check at least 32 GiB available RAM and 51 GiB artifact free space (40-GiB
reserve, 10-GiB collection, 1-GiB persistence headroom) before admission and
again before launch. Monitor resources during the run. CPU-only replay and
read-only diagnostics may coexist; available cores alone do not justify another
native scene.

The explicit collector and auditor sources differ from the frozen original
anchored collector/auditor only in controller import, case status strings and
the declared `ordinary_waypoint_commitment_contact_enabled` receipt/assertion.
Whole-module AST checks verify that scope. Keep the same maze specification,
public mission, physics/appearance seeds, renderers, body/gait, URDF, friction,
front and auxiliary cameras, 100-ms command interval, 50-by-2-ms physics samples,
three zero warmups, 3,000 shared navigation ticks and ten terminal zero commands.
Retain complete raw sensors, model decisions, physics, command tapes, renderer
witnesses, logs and every failure. A second fresh model reconstructs the entire
raw controller sequence after collection.

Before claiming the intervention executed, compare all 900 original/candidate
physics samples through index 899 and all four public packets 0–3. Require
three identical prior commands and every complete candidate decision to equal
the prospective raw replay. Require the changed frame-3 forward command
`[0.2,0,0]` to complete through physical sample 949; the original command is
left turn `[0,0,0.45]`. Stop paired outcome equality before the changed physical
future. The new trajectory thereafter is evaluated on its own actual evidence.

Retain the original full raw-sensor/model/command, model-state, strict-visibility,
contact, observed/native goal-and-home arrival, settling, route reversal and
terminal quietness checks. A raw-audited negative episode is retained as a
completed scientific failure. Invalid source, input, collection, audit or
physical-prefix evidence remains an explicit failed attempt. Simulation pauses
physics during computation; report actual wall timing without claiming real-time
operation. This single reused layout cannot establish independent-maze
reliability, JEPA/planning/memory advantage, calibrated safety or real-platform
readiness. No hardware action or deployment qualification is implied.
