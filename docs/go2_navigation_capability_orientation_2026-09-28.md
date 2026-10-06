# Navigation capability: orientation on taking over, 28 September 2026

Written on taking over from the previous session, under [the handoff](go2_navigation_capability_handoff_2026-09-28.md). Checked against the repository records and the artifact root at 09:55 BST. **Nothing has been run.** I stopped because of the handoff's discrepancy rule: one count in the handoff disagrees with the records (section 2). No erratum has been recorded, no 02/0 re-evaluation done and no gate episode launched.

## 1. Confirmed from the records

| Item | Record | Value |
|---|---|---|
| Current harness | `go2_navigation_capability_harness_v4_completed_support_final_2026-09-27.json`, frozen at `7da82b23` | `v4_completed_support`, SHA-256 `82b7b6049a93c34cc00599db93a82fee0e7a882683af45ccaae47b5443c84871`. All 150 implementation bindings match the working tree today. |
| Protocol | `go2_navigation_capability_completed_support_v4_2026-09-27.json` | SHA-256 `c757503b…0720f`. Gate owner: `scripts/run_go2_capability_completed_support_v4_cohort_development.py` (`332b26da…`). |
| Versions charged | Per-version freeze records (`outcome_driven_versions_consumed`) | **5**: `v0_startup_c2` (containment failed), `v1_paired_floor`, `v2_exhausted_view`, `v3_live_turn`, `v4_completed_support`. Not charged: `v0_task_c1` and `v0_grid_c3` (correctness), `v3c1_live_turn` (implementation erratum: wrong deployed memory class bound; zero decisions run). See section 2.2 on what the cap counts. |
| Wall hours | `wall_budget_origin.json`, conservative calendar accounting from 25 Sep 11:49 BST | **70.10 of 160 h** used, idle time included. The last recorded projection was 102.98 h, made at 62.24 h elapsed. A rough update, using the measured V4 gate C0 owner time (mean 971 s, against 676 s in the pilot), gives about 112 h with 15% contingency. The formal re-projection is due after the next cohort. |
| Storage | `df`, now | RecoveryStorage: 103.1 GiB free, so **91.1 GiB usable** above the 12-GiB reserve. The last record said 91.9 GiB, and 46.9 GB was projected as still needed. The handoff's "about 100 GiB" is approximately right. Workspace: **5.47 GiB free**, only 1.47 GiB above its 4-GiB reserve. |
| C3 head | Protocol `controllers.C3` | **Maze-data head**: `go2_maze_view_readout_v1_attempt_003/maze_data_final.pt`, `aa853c6f…`. Justification: lower V4.2 regret point estimates in both exposure strata. Encoder `vjepa2_1_vitl_dist_vitG_384.pt` (`7ea9b7cb…`); predictor `go2_horizon_dense_predictor_v1_attempt_001/action_final.pt` (`5d39753f…`). |
| C4 source | Protocol `controllers.C4`; `C4_final_binding` | **A new direct supervised fit.** The inventory found no compatible existing predictor. It is trained on the same 5,966 original/heading contexts and 2,448 maze contexts as the maze-data readout, has about 17.4M parameters, one seed and 1,760 updates. Checkpoint `c4_fit_attempt002/direct_final.pt`, `d491584b…`. Training-render provenance is unverified. |
| Validation episode | Protocol `fixed_validation`; pilot report | **Episode 0 of each maze: 10/0–29/0.** C0 runs on 10/0–19/0. These were fixed before any validation outcome. |
| C0 gate progress | `cohorts/v4_completed_support_C0_gate/`; stop record | **Complete and qualified:** 00/0, 00/1, 01/0, 01/1, all round trips with 0 contacts, 0 hard violations and 0 unresolved samples. **Fifth attempt 02/0: unqualified** (below). **Unstarted:** 02/1–09/1 (15). |
| Fifth attempt's ending | 02/0 `mission.json`, `result.json` | It **reached a terminal mission outcome**: `OBSERVED_ROUND_TRIP_CANDIDATE` at 160.0 s simulator time (158.5 s of mission). Outbound arrival was observed at 94.4 s. The closeout prefix check then raised `oracle executed-prefix fidelity check failed`. The native physical reader (arrival, contacts, clearance) was never run, so this is not yet a verified success or a verified-safe run. All native arrays, requests, planning records and hashes are preserved. |
| First C0 gate replay | `completed_support_v4_C0_sensor_replay/result.json` (`f6d8bd6a…`) | **PASS.** 1,128 frame pairs; 2,256 RGB and 2,256 depth packets matched bitwise; native trace exact. Retention was full frames for 00/0 and hashes only for 00/1, 01/0, 01/1 and 02/0, each bound to this receipt. |
| Safety history | Every screen result record, the V3 report and the pilots | No disallowed contact or hard or operating-margin violation in any evaluated run. 02/0 has not been evaluated. |
| V4's change; did a V3 exist? | Freeze records | **V3 existed** (`v3_live_turn`, `8c825807`, with its V3c1 binding erratum `4f503906`). It released the turn-memory latch when its direction becomes ineligible, and the screen gave 8/10. **V4 is not a turn-selection change.** Existing recovery now uses the tracker's actual selected-feature count, including sparse corner completion, instead of only the original strong-corner subset. Its screens gave 10/10 on the first episodes and 10/10 on the second. |
| V4 record names | `docs/` | `go2_navigation_capability_completed_support_v4_2026-09-27.json` (protocol), `…_harness_v4_completed_support_final_2026-09-27.json`, `…_completed_support_v4_budget_2026-09-27.json`, `…_completed_support_v4_screen_result_2026-09-28.{md,json}`, `…_completed_support_v4_second_screen_result_2026-09-28.{md,json}`, `…_v3_support_diagnosis_2026-09-27.md`. |
| Pre-registration | `docs/` | `go2_navigation_capability_preregistration_v1_2026-09-25.{md,json,sha256}`; JSON `b6a84db2…`, committed at `bb9ab232`. |
| Progress report | Filesystem | `/home/andrewknowles/Documents/LeWMQuad_JEPA_World_Model_Progress_Report_2026-09-23.md`. It is outside the repository. |

Other handoff facts check out: the 160-h cap (`13ebc8e2` erratum); 13,278 C1–C4 sensor packets; the §7 history for task_c1, startup_c2, grid_c3, paired_floor_v1 and exhausted_view_v2; the ±8/±7.9-m bound; and 2,322 exact comparable prefixes. Two throughput figures are approximate, not contradicted: the V4 C1 screens took 1.43 h and 1.22 h, against the "2–3 h" in the handoff.

## 2. Discrepancies

### 2.1 Count of unmatched decisions in 02/0

**Handoff §3:** "for six decisions there was no executed prefix", at "decisions where C0 had selected a forward candidate".

**Records:** there was one decision, not six. At the 119.9-s source boundary, all six candidate branches had zero matched prefix, which gives six comparison rows (`oracle_prefix_check.json`; stop record).

- All six branch tapes, including hold, begin with `[0.2, 0, 0]`. That is the 300-ms committed prefix carried over from the 119.5-s selection, so it is common to every candidate. It is not a forward candidate chosen at 119.9 s.
- The dispatch at the boundary tick applied `[0, 0, 0]`, because `CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE` found no current observation. The next 14 ticks were `COMMAND_WINDOW_VETO_LATCHED`.
- The 20 later requests labelled with the 119.9-s origin are all `COMMITTED_PREFIX_NOT_EXECUTED` holds, from 120.20 to 120.58 s. The shared harness refused to execute that plan. **The unqualified prediction therefore produced no motion.**
- The same veto sequence occurs in C1. In the V4 second screen, 02/1, 05/1 and 08/1 each have one or two stale-observation vetoes and 20 `COMMITTED_PREFIX_NOT_EXECUTED` holds. C0 00/1 had a stale veto at 129.8 s, 300 ms after the 129.5-s boundary. Its hold branch matched the dispatched commands for 700 ms and the other five matched for 300 ms, all exactly, so it passed. This is a shared-harness mechanism, not something C0 or restoration caused.

**Effect on the disposition, as I read it: none.** The approved erratum would count 02/0 as having one decision with no matching branch, caused by a veto. The other 2,322 comparisons match exactly. The attempt reached a terminal outcome, so it would be re-evaluated from its preserved records and not rerun. I'm reporting this as the handoff requires, before running anything.

### 2.2 What the six-version cap counts

This one only matters if the gate fails.

- The pre-registration fixes "at most six versions **including V0**", with version IDs 0–5.
- The later records count only charged changes against six. They say "five of six outcome versions consumed", which leaves one change available.
- Under the pre-registration's wording, V0 plus five charged changes makes V4 the sixth and final version. A failed gate would then mean stop and report, not make another change.
- Handoff §4 assumes another change is possible.

I need a ruling on which count applies before any post-gate iteration.

## 3. Ready to run once confirmed

1. **Record the erratum.** For each decision, compare only the branches whose command tape matches the dispatched commands over a prefix of at least one tick. Any mismatch on such a prefix still stops. A decision where no branch matches is counted and reported with its cause (veto or override), and neither qualifies nor disqualifies the episode. Also report veto counts per episode. The existing checker (`lewm/navigation_capability_oracle_development.py:verify_executed`) is unchanged. The erratum check runs evaluator-side over the preserved records.
2. **Resolve 02/0 from its records, without rerunning it.** Run the erratum check over the preserved branches and requests. Then run the unchanged native physical reader for arrival, contacts and clearance. Its navigation outcome stands, whatever it is.
3. **Resume the gate.** Run 02/1 through 09/1 on the same frozen V4 harness and owner, with hash retention, recheck the reserves first, and apply the erratum at closeout. Veto counts so far per C0 episode (stale / latched / not executed): 00/0 0/0/0; 00/1 1/64/0; 01/0 0/0/0; 01/1 0/0/0; 02/0 1/14/20.
