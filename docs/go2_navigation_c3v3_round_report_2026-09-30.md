# C3-v3 on-policy round: report, 30 September 2026

## Outcome

**C3-v3 did not pass the pre-declared offline acceptance, so C3-v2 and C4-v2 enter E1.**
- It passed the primary closed-loop criteria by a wide margin, but failed both no-regression criteria (N1 and N2).
- Under §7 of the [pre-declaration](go2_navigation_c3v3_onpolicy_round_predeclaration_2026-09-30.md) (commit ec2e34c9; Amendment 1, commit 45eeca14):
  - the safety check does not run;
  - C3-v3 and C4-v3 do not enter E1.
- This was the second and final C3 intervention before E1.
- C3-v1's validation result (13/20) remains the pre-registered capability result.
- **E1 launches only after Andrew's confirmation.**

**The trade-off.** On-policy C1 data fixed C3's closed-loop cruising deficit, but it cost accuracy on the slow and turning offline windows.

| C3 criterion (C3-v3 against C3-v2) | Result | Pass? |
|---|---|---|
| P1: closed-loop moving, median 800-ms predicted/true ratio in [0.75, 1.25] (1,348 decisions) | **0.917** (C3-v2 0.329) | yes |
| P2: closed-loop moving, median XY error ≤ 50% of C3-v2's | **21.0 mm** against 89.0 mm | yes |
| R1: offline rest starts, median ratio in [0.75, 1.25] (72 windows) | 1.092 (C3-v2 0.799) | yes |
| N1: offline held-out groups, XY/yaw RMSE ≤ 1.05× C3-v2 | 11 of 12 measures above 1.05×; in-place turns worst | **no** |
| N2: transfer population, XY/yaw RMSE ≤ 1.05× C3-v2 | all 4 measures above 1.05× (1.21–1.66×) | **no** |
| N3: in-place-turn spurious translation ≤ 10 mm | 4.8 mm | yes |

**Recommendation.**
- Launch E1, when Andrew confirms, with **C3-v2 and C4-v2** as pre-declared.
- The E1 report must state C3-v2's measured closed-loop deficit. On C1's held-out cruising states it predicts 0.33 of true forward travel (89 mm XY error at 800 ms); on its own fresh-check states the figure is 0.21. E1's C3 comparisons therefore largely measure a readout that under-predicts travel while cruising, and not the representation alone.
- **The round shows the deficit can be fixed from the same frozen features.** C3-v3 reaches 0.92 on those states. So the representation is not what limits C3 here. A follow-up intervention (after E1, which is out of scope now) would need a mix that keeps the slow and turning windows, and would need its own pre-declaration.

## 1. What was run

**Sets.** 32 new mazes (`c3v3_sets_v1/registry.json`, `b1645898…`), excluding every earlier maze: 16 on-policy fit, 6 on-policy held-out, 10 safety check.

**On-policy data from C1** (`cohorts/c3v3_onpolicy_c1`).
- 22/22 round trips with zero contacts and zero hard violations, in 54 min.
- All 22 missions were replayed to regenerate their frames, and every replay matched the log bit for bit.
- 25,469 fit contexts, and 2,243 held-out decisions.

**Matched fits** on identical data: 36,651 contexts, with each batch 32 old + 16 C3-v2 maze pool + 16 on-policy.
- C3-v3 readout `85ab19ec…`: 440 updates, recipe identical to C3-v1 and C3-v2, 2.77 h.
- C4-v3 `992c22fb…`: 1,760 updates, C4-v1's recipe; C4's GPU total is 8.10 of 12 h.
- Neither fit used held-out data. The records are in the [model-version record](go2_navigation_c3v3_c4v3_model_versions_2026-09-30.md).

**Amendment 1 to criterion P** (approved, committed before any held-out output existed).
- Under the original exact-tape definition only 8 held-out decisions qualified. C1 cruises continuously, so its forward candidate tape `FFFFFFF0` rarely equals the executed `FFFFFFFF`.
- P is now scored on the executed tape: 1,348 moving decisions with at least four forward steps.

## 2. Closed-loop held-out results (P and the reporting additions)

All numbers use the deployed computation. Each cell is the median predicted/true 800-ms translation, then the median XY error at 800 ms.

| Decisions | n | True | C3-v1 | C3-v2 | **C3-v3** | C4-v1 | C4-v2 | **C4-v3** |
|---|---:|---:|---|---|---|---|---|---|
| **Moving, pooled (P)** | 1,348 | 138 mm | 0.13 · 118 | 0.33 · 89 | **0.92 · 21** | 0.79 · 29 | 0.90 · 16 | **0.98 · 8** |
| Steady cruise | 378 | | 0.16 · 114 | 0.37 · 87 | 0.94 · 19 | 0.81 · 27 | 0.91 · 13 | 0.98 · 7 |
| Command switch | 970 | | 0.12 · 119 | 0.31 · 91 | 0.91 · 22 | 0.78 · 30 | 0.90 · 16 | 0.99 · 9 |
| From rest (closed loop) | 1 | 45 mm | 0.29 · 32 | 0.28 · 32 | 1.41 · 21 | 0.98 · 3 | 0.86 · 9 | 1.45 · 22 |
| Original exact-match definition | 8 | 141 mm | 0.08 · 130 | 0.17 · 116 | 0.71 · 34 | 0.76 · 35 | 0.84 · 22 | 0.83 · 18 |

On the 8 exact-match decisions, C1's logged forecast gives **0.94 · 9 mm**. Only there is C1's forecast for the executed tape available.

**Per mission, moving decisions** (C3-v2 → C3-v3 | C4-v2 → C4-v3; ratio · XY error in mm):

| Mission | n | C3-v2 | C3-v3 | C4-v2 | C4-v3 |
|---|---:|---|---|---|---|
| 16 | 229 | 0.35 · 87 | 0.92 · 20 | 0.89 · 18 | 0.97 · 9 |
| 17 | 224 | 0.34 · 88 | 0.92 · 22 | 0.91 · 15 | 0.98 · 9 |
| 18 | 250 | 0.24 · 102 | 0.90 · 22 | 0.91 · 16 | 0.98 · 8 |
| 19 | 173 | 0.38 · 82 | 0.91 · 21 | 0.89 · 17 | 0.98 · 8 |
| 20 | 207 | 0.41 · 77 | 0.88 · 25 | 0.90 · 16 | 0.99 · 8 |
| 21 | 265 | 0.30 · 94 | 0.95 · 19 | 0.91 · 14 | 0.99 · 7 |

**The closed-loop gain is consistent:** in all 6 missions and in both tape strata, C3-v3 lies within 0.88–0.95.

## 3. Offline regressions (N1 and N2)

These are RMSEs on the existing C3-v2 held-out recordings and on the transfer population, as C3-v3/C3-v2 ratios. C4-v3/C4-v2 is shown for reference.

| Windows | Measure | C3-v2 | C3-v3 | C3 ratio | C4-v3/C4-v2 |
|---|---|---:|---:|---:|---:|
| In-place turns (672) | XY 500 / 800 ms | 7.2 / 9.1 mm | 17.6 / 21.4 mm | **2.46 / 2.35** | 1.99 / 2.03 |
| | Yaw 500 / 800 ms | 1.52 / 2.36° | 1.97 / 3.52° | 1.29 / 1.49 | 1.26 / 1.46 |
| Rest starts (72) | XY 500 / 800 ms | 17.3 / 31.0 mm | 29.8 / 28.0 mm | **1.73** / 0.91 | 0.72 / 0.83 |
| | Yaw 500 / 800 ms | 1.74 / 2.04° | 1.97 / 2.87° | 1.13 / 1.41 | 1.00 / 0.85 |
| Other (640) | XY 500 / 800 ms | 14.9 / 19.5 mm | 18.3 / 21.0 mm | 1.22 / 1.07 | 1.13 / 1.27 |
| | Yaw 500 / 800 ms | 1.12 / 1.42° | 1.31 / 1.78° | 1.17 / 1.25 | 1.01 / 1.04 |
| Transfer (240) | XY 500 / 700 ms | 11.9 / 15.9 mm | 19.0 / 19.9 mm | **1.60** / 1.25 | 1.19 / 1.42 |
| | Yaw 500 / 700 ms | 1.62 / 1.69° | 1.96 / 2.82° | 1.21 / **1.66** | 1.13 / 1.33 |

**Reading.**
- C3-v3 now **over-predicts translation in slow and turning states.** Its median 800-ms translation ratio is 1.62 on in-place turns, against 0.90 for C3-v2, and 1.11 on the other windows, against 0.80. Its yaw error also grew.
- **C4-v3 moved the same way on turns and transfer** (turn XY about 2×, transfer 1.19–1.42×). So the regression comes from the data mix: a fixed 16-of-64 on-policy share drawn from a large cruising-dominated pool. It is not specific to the JEPA path.

## 4. What was not run

**The safety check did not run:** under §6 it runs only if C3-v3 passes. Its 10 registered mazes (layouts 22–31) remain unused.

## 5. Budget and storage

- **Programme active time:** 108.4 h of 160 h.
- **E1's own running time so far:** 0.23 h.
- **The owner's calendar stop** (2 Oct 03:47 BST) is irrelevant now, because no further missions are planned before E1.
- **RecoveryStorage:** 122.7 GiB free after the approved deletion (see [the storage log](storage_manifests/storage_log.md)). After E1's projected 64.9 GiB, 45.8 GiB would remain above the 12-GiB reserve.
