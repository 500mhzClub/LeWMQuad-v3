# Multi-seed navigation: completed 22-run development comparison

All 22 fixed assignments are complete and evaluated: **17 verified round
trips, five failures, zero recorded contacts**. JEPA, direct prediction and
supervised rollout each completed 5/6 assignments. The fitted-motion control
completed 2/2; the reactive control completed 0/2. This experiment establishes
no JEPA success-rate advantage and no learned-motion advantage over the fitted
control. The broad navigation goal remains incomplete.

| Method | Goal and round trip | Assignments | Contacts |
| --- | ---: | ---: | ---: |
| JEPA | 5 | 6 | 0 |
| Direct prediction | 5 | 6 | 0 |
| Supervised rollout | 5 | 6 | 0 |
| Fitted motion | 2 | 2 | 0 |
| Reactive | 0 | 2 | 0 |

## Scope and actual treatment

Three training seeds (2026091001, 2026091401, 2026091402), three neural training
methods and two development mazes supply 18 learned-model assignments. Each
fitted/reactive control runs once per maze. The training-seed repetitions are
not additional independent mazes. Both layouts are distinct from the explicit
76-layout development registry, within the same maze family. This is not sealed
or final evaluation.

All nine neural models and their matched motion corrections were fixed before
navigation. The learned arms used model-derived corrected XY and learned yaw;
the fitted control used fitted pose/command XY and command yaw. Reactive
selection used no neural model or candidate future outcomes. Its predictive
clearance and some recovery rules differ from the predictive arms, so this
comparison does not isolate predictive action ranking alone. Contact scoring
was disabled in the predictive arms. Persistent observation-based routing
memory was retained throughout; this experiment contains no memory ablation.

All shared settings and 144 common runtime source hashes match within each
maze. The tracker was BatchedConsensusMotion with unchanged reference banks,
floor registration and reacquisition. No failed assignment was replaced.

Sensing used paired RGB and depth, a fixed 200-mm to 5-m depth range, 2-mm
synthetic depth noise and ideal public gyro. Physics ran at 2 ms, gait requests
at 20 ms, cameras at 100 ms and planning at 400 ms, with a 300-ms planning delay.
The navigation budget was 4,800 ticks. Observed arrivals required 20 mm and
quiet dwell; independent physical checks required 40 mm, one second of zero
requests and bounded measured motion. Native geometry and poses were used
only for rendering/evaluation, not supplied to planning.

Timing used the measured-simulation clock. These are not hard real-time or
hardware-calibrated results; some runs accumulated substantial host lag.

## Per-assignment results

Times below are simulated seconds for completed round trips. Failed runs stay
in the table rather than being excluded from a successful-run timing average.

| Method | Training seed | Maze 0 | Maze 1 |
| --- | --- | --- | --- |
| JEPA | 2026091001 | Round trip, 224.92 s | Round trip, 207.68 s |
| JEPA | 2026091401 | Round trip, 252.60 s | Tracking queue failure |
| JEPA | 2026091402 | Round trip, 281.92 s | Round trip, 148.42 s |
| Direct prediction | 2026091001 | Round trip, 250.78 s | Round trip, 181.56 s |
| Direct prediction | 2026091401 | Round trip, 219.92 s | Round trip, 142.40 s |
| Direct prediction | 2026091402 | Round trip, 228.42 s | Tracking queue failure |
| Supervised rollout | 2026091001 | Round trip, 206.46 s | Round trip, 161.14 s |
| Supervised rollout | 2026091401 | Round trip, 214.82 s | Round trip, 136.00 s |
| Supervised rollout | 2026091402 | Round trip, 239.94 s | Budget exhausted; no goal |
| Fitted motion | — | Round trip, 209.70 s | Round trip, 178.44 s |
| Reactive | — | Budget exhausted; no goal | Budget exhausted; no goal |

## The five failures

- **JEPA seed 1401, maze 1:** tracking queue overflow; 494 captured frames,
  460 registered poses, maximum position error 3.386 mm, no arrival. Late
  tracking service median was 117 ms against the 100-ms camera period.
- **Direct seed 1402, maze 1:** tracking queue overflow; 460 captured frames,
  426 registered poses, maximum position error 2.869 mm, no arrival. Exact
  replay and CPU profiling identified expensive repeated reference attempts,
  paired-floor fitting and evidence copying. Neither failure is replaced.
- **Supervised seed 1402, maze 1:** forward depth became entirely nearer than
  the 200-mm sensor minimum. Floor/planning reacquisition occurred 58 times,
  but independent obstacle evidence remained unavailable and movement did
  not resume. No goal within 480.72 simulated seconds.
- **Reactive, maze 1:** the same near-range blindness mechanism. Fifty floor/
  planning resumptions produced no subsequent nonzero command. No goal within
  480.80 simulated seconds.
- **Reactive, maze 0:** no missing poses or floor-reacquisition holds. After
  right-arc commands, all 531 plans from frame 2680 onward selected hold due
  to stored clearance below the 45-cm nominal radius. At first hold, stored
  clearance was 438.9 mm versus physical centre-wall distance 475.2 mm; the
  final values were about 402.1/438.7 mm. The source of the map/pose discrepancy
  and the approach dynamics are not fully isolated. Maximum pose error was
  32.329 mm. No goal within 480.86 simulated seconds.

The final reactive run exited 0 after 11:02.81, with 4,805 poses, 1,193/1,200
plans on time and zero contacts; clean process exit does not change its failed
navigation outcome. The final direct-seed-1401 run exited 0 in 3:44.99, with
1,422 poses, goal/home frames 948/1420, 331/348 on-time plans, maximum pose
error 4.585 mm and physical quiet-dwell maxima 15.924/9.431 mm.

## Evidence and retention

Artifact base:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

- Aggregate: `go2_multiseed_navigation_complete_comparison_v1_attempt_001/result.json`.
- Per-run roots: `go2_multiseed_navigation_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001`.
  Each contains physical arrival, actual controller treatment and summary
  evaluations; learned/fitted arms also retain saved motion/yaw analyses.
- Six five-panel comparisons, all PNG/SVG generated and visually inspected:
  `go2_multiseed_navigation_seed{seed}_layout{index:02d}_all_controls_comparison_v1_attempt_001/`.
  The same two control runs are repeated across seed panels, explicitly labelled.
- Layout inventory: `docs/go2_multiseed_navigation_layout_inventory_2026-09-15.json`.
- Fixed model/fit registry: `docs/go2_multiseed_navigation_models_2026-09-15.json`.
- Earlier detailed progress: `docs/go2_multiseed_navigation_progress_notes_2026-09-15.md`.

After complete evaluation and plotting, depth was retired from eleven redundant
successes: 45,522 depth files and 14,148,472,832 allocated bytes. Preserve full
seed-1001 maze-0 references for all three neural methods, the first supervised
success (seed 1401 maze 1), both fitted controls and every failed recording.
All outcomes, RGB, poses, gyro/body, commands, physics, models and fits remain.
Retired recordings no longer support exact historical depth replay. Inventory:
`.generated/depth_retirement_completed_multiseed_successes_2026-09-15/`.
About 19.4 GB was free after retirement, before the next experiment.

Later retention update: the redundant seed-1401 supervised layout-1 and fitted
layout-1 success depth was retired before the shared-recovery transfer's direct
assignment. Seed-1001 layout-0 neural references, fitted layout 0 and every
failure remain full; all original outcomes and non-depth records remain.
See `docs/go2_development_artifact_retention_2026-09-14.md` for the completed
inventory. This supersedes only those two full-depth pins above.

After the complete five-controller shared-recovery layout-0 comparison, the
remaining four older successful references (seed-1001 layout-0 JEPA/direct/
supervised and fitted layout 0) also retired depth. The five new controller
references remain full, along with every original multiseed failure. All
original outcomes and non-depth records remain. The completed inventory is
`.generated/depth_retirement_superseded_multiseed_references_2026-09-15/`.

## Next scientific work

Both fixed auxiliary-only turn recovery follow-ups are complete. Reactive
exercised 40 degraded turn requests but resumed no translation and exhausted
its budget. Supervised seed 1402 failed with tracking queue overflow before
any degraded-turn exposure. Neither reached the goal; both had zero contacts
and remain separate failures. See `docs/go2_auxiliary_turn_recovery_2026-09-15.md`.

Tracking throughput still needs a material improvement in native execution. Two isolated
460-frame replays preserved every raw/registered pose, paired-floor constraint
and revisit receipt. Whole-pair caching reused only 96/1,687 fits and did not
improve overall replay time. Deferring one duplicate registration copy reduced
late-frame median motion time by about 5 ms (to 96–97 ms), but many calls
still exceeded 100 ms and total replay time was essentially unchanged
(59.512 versus 59.608 seconds). Neither variant is used by navigation or the
camera-recovery follow-ups. Results are in the retained direct-seed-1402
failure under `gyro_coherent_floor_{cached_pair_floor|deferred_registration_copy}_replay_v1/`.

Further work must establish reliable sensing/timing, repeatability on additional
unseen mazes, cleaner isolation of predictive-planning contributions, and
bounded real-platform evidence. The available simulation evidence does not
establish deployment readiness or a JEPA-specific benefit.

A third isolated optimization reuses individual floor-cloud statistics. It
preserved all 460 replay estimates/receipts exactly and reduced late motion
medians by about 7 ms against a fresh baseline, with essentially unchanged
overall replay time. The direct-seed-1402 layout-1 native follow-up completed
a verified round trip in 162.12 simulated seconds, with zero contacts and no
pipeline faults. This tracker change was the only intended intervention;
native trajectory and scheduling also differed. The original failure remains
in the 22-run comparison. A separate exact 494-frame JEPA-failure replay and
native follow-up are complete: the JEPA follow-up also achieved a verified
round trip, in 163.22 simulated seconds, with zero contacts. Both native runs
had lighter host load than their original failures. Both fixed original-tracker
serial controls also achieved verified contact-free round trips (157.64 s
direct; 146.50 s JEPA). Thus the cache improves isolated estimator cost but
has no demonstrated native success or speed advantage in these controls. See
`docs/go2_cached_floor_moments_tracking_2026-09-15.md`.

Close-wall candidate selection now has a stronger replay result: conditioning
registration candidates on its exact transported reference normal restored all
820 poses in a recorded supervised-failure prefix (581 originally), preserved
every shared pose and restored 243 frames with usable pose/obstacle evidence.
The native supervised test stopped at another tracking overflow before recovery
exposure; the reactive test completed with all poses accepted and no floor holds,
but still failed to reach the goal. It exhibited repeated right-arc vetoes
followed by leftward view recovery and a final stored-clearance hold. Both
attempts remain failures, with zero contacts. See
`docs/go2_transport_conditioned_floor_recovery_2026-09-15.md`.
