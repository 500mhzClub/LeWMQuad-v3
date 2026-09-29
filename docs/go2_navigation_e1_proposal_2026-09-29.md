# E1 benchmark proposal, 29 September 2026

**This is a proposal, not an authorisation.** E1 execution needs Andrew's approval. It follows the [capability qualification result](go2_navigation_capability_qualification_result_2026-09-29.md) on the frozen harness `v4_completed_support`, and every number below is measured in this programme.

## 1. Decisions needed before E1

1. **The C3 intervention.** C3 did not qualify. The qualification report proposes one targeted intervention, a recalibration of C3's motion readout. E1 should either run C3 as it stands, which measures the current JEPA stack, or wait until the intervention has been approved, developed and checked. The check must use development data and the V4.2 branch panel, never validation or sealed data.
2. **Shared-harness traps.** Three harness mechanisms caused failures across controllers:
   - the view-requirement deadlock, where no movement is eligible;
   - the terminal-heading limit cycle at the goal;
   - the latched recovery turn blocked by clearance.

   Fixing any of them would be a new harness version. The six-version cap is reached, so this is Andrew's call. It would also need a fresh C0 gate and a 20-episode C1 screen, about 8 h of machine time. If the traps stay, E1 measures each controller together with those traps, and the report must say so.
3. **The post-intervention check set.** The validation set has now been used once, for qualification. Re-checking an intervened controller on it would be iteration on validation, which is forbidden. So a post-intervention capability check needs either a newly generated held-out set (needs approval) or going straight to the sealed E1 set.
4. **Design size.** Section 3 below.

## 2. Design

- **Controllers:**
  - C1 command history;
  - C2 reactive;
  - C3 JEPA;
  - C4 supervised predictor;
  - C0 on a 10-maze subset, only as a check that the harness still behaves on new mazes.
- **Sets:** the 60 sealed test mazes, 120 registered episodes, paired across controllers, with the maze as the unit.
- **Seeds:** three independent training seeds each for C3's learned parts (predictor and readout) and for C4. The frozen V-JEPA 2.1 encoder is shared and not retrained. The C1 fit is deterministic, so it gets one instance.
- **Pre-registered outcome,** fixed before any run. It is the paired per-maze difference in round-trip success, averaged over seeds within each maze, with a maze-cluster bootstrap. The comparisons are:
  - H2: C3 against C1;
  - H4: C3 against C4;
  - C3 against C2;
  - SPL, time and stall as secondary outcomes;
  - safety as a hard constraint.
- **Mechanism accounting:** the failure-mechanism rules already fixed in `scripts/diagnose_go2_capability_validation_timeouts_development.py`, frozen before E1.
- **Labels:** E1 results are paper results only if the pre-registration says so. The training-render provenance caveat still applies to C3 and C4.

## 3. Measured cost and schedule

**Missions.** Owner wall time per mission, measured during qualification with up to five concurrent owners on one R9700 GPU and a 16-core CPU:

| Controller | Median simulated s | Median owner wall s | Resource |
|---|---:|---:|---|
| C1 | 161 | 413 | CPU |
| C2 | 331 | 817 | CPU |
| C4 | 159 | 1,469 | GPU (V-JEPA encoder) |
| C3 | 397 | 5,729 (2 concurrent) | GPU (encoder + predictor) |
| C0 | 202 | 1,388 | CPU (physics branches) |

C3 and C4 share one GPU, which ran at 100% throughout, so the GPU sets the schedule. Qualification needed about 15 h of wall time for 20 C3 and 20 C4 missions. C1 and C2 run on the otherwise idle CPU in parallel.

**Training per seed.** Per seed:
- C3 predictor: about 0.5 GPU-h (1,760 updates per arm, including encoding).
- C3 readout: about 3.5 h of feature extraction, cacheable across seeds, plus minutes of updates.
- C4: about 1.35 GPU-h.

That's about 8 GPU-h for three seeds of both.

| Design | Missions (C3 + C4 per seed × seeds) | GPU schedule | Storage (hash-only, about 120 MB per mission) |
|---|---|---:|---:|
| 60 mazes × 1 episode × 1 seed (pilot) | 60 + 60 | about 2 days | about 20 GB |
| **60 × 1 × 3 seeds (recommended)** | 180 + 180 | **about 5–6 days** | about 60 GB |
| 60 × 2 × 3 seeds | 360 + 360 | about 11–12 days | about 115 GB, which exceeds the roughly 90 GiB free |

**Recommendation.** Run 60 mazes × 1 episode × 3 seeds, reserving the second episodes. Start with the 1-seed pilot and re-project before seeds 2 and 3. Videos come from verified replay: 8–25 minutes each, and the replays have already reproduced concurrent runs bit for bit.

## 4. Real-time latency mode

- **Current decision latency.** Measured in isolation (median / p95, 297 replay-verified decisions each): C1 0.20 / 0.22 s, C2 0.19 / 0.21 s, **C4 0.83 / 0.85 s**, **C3 2.66 / 2.71 s**, C0 1.73 / 1.76 s. Under qualification concurrency, C3 rose to 8.1 s and C4 to 3.0 s.
- **The comparison.** The earlier real-time planning deadline was 300 ms. C3 needs about nine times less compute per decision to meet it, and C4 about three times less. C1 and C2 already meet it.
- **Recommendation.** Do **not** add a real-time mode to E1. With the current models it would simply fail C3 and C4 for compute reasons, and that would mask the representation question.
- **Instead:**
  1. keep physics-paused E1 as the primary condition;
  2. report decision latency as a cost;
  3. run a separate feasibility study before E4 (a smaller or distilled encoder, and a lighter predictor), with its own gate.

## 5. Risks

- C3 at its current capability would make E1 mostly a measurement of the no-eligible-movement deadlock, unless the intervention and harness decisions come first.
- One GPU makes C3 the schedule's long pole. A second GPU would roughly halve E1.
- The workspace disk is close to its reserve (about 1.5 GiB above 4 GiB). All E1 outputs must stay on RecoveryStorage.
- Sensor idealisation must be declared in any E1 write-up: noise-free RGB, and an ideal gyro and accelerometer, the accelerometer being used for initial gravity alignment. Real-hardware claims need E4.
