# Paired-floor V1 failures by mission leg — 27 September 2026

**Three return timeouts (01, 05, 09), one outbound timeout (08), and one return pose loss (07).** All five are diagnosed from the completed recordings; no new physics, models, historical frame access or sealed material was used. The 480-second mission budget is unchanged.

## Time and distance accounting

Times are simulated seconds from the first consumed packet, excluding the original 1.5-second settling. Translation, turn-only and hold are mutually exclusive classifications of the logged applied 20-ms commands and sum to the leg duration. Nonzero command is activity, not proof of net progress: in-place turns and gait drift can consume time without advancing. Holds include arrival settling, dispatch vetoes and selected holds.

Visual recovery is an **overlapping** quantity: duration of the latest logged planning result with `LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW`, held until the next planning result. The complete camera-cadence recovery latch was not retained, so these are planning-state durations, not exact per-frame latch durations. No recovery time is added to the other three columns.

Shortest round-trip distances are the registered home–beacon path plus its reverse, using the predeclared 0.46-m inflation plus 5-mm clearance on the 20-mm true-geometry grid.

| Episode / outcome | Beacon arrival (s) | Shortest round trip (m) | Leg | Translate (s) | Turn only (s) | Hold (s) | Visual recovery (s, overlaps) |
|---|---:|---:|---|---:|---:|---:|---:|
| 01/0 / timeout | 144.0 | 20.940 | OUTBOUND | 63.4 | 61.8 | 18.8 | 0.8 |
| 01/0 / timeout | 144.0 | 20.940 | RETURN | 0.0 | 47.6 | 288.4 | 20.4 |
| 05/0 / timeout | 103.6 | 14.183 | OUTBOUND | 59.0 | 38.7 | 5.9 | 0.0 |
| 05/0 / timeout | 103.6 | 14.183 | RETURN | 5.3 | 64.7 | 306.4 | 270.0 |
| 07/0 / pose_loss | 211.70000000000002 | 12.964 | OUTBOUND | 48.5 | 76.7 | 86.5 | 78.8 |
| 07/0 / pose_loss | 211.70000000000002 | 12.964 | RETURN | 0.0 | 32.4 | 4.9 | 14.6 |
| 08/0 / timeout | not reached | 21.338 | OUTBOUND | 6.4 | 154.8 | 318.8 | 320.0 |
| 09/0 / timeout | 458.09999999999997 | 18.774 | OUTBOUND | 71.8 | 352.2 | 34.1 | 98.8 |
| 09/0 / timeout | 458.09999999999997 | 18.774 | RETURN | 16.1 | 4.6 | 1.2 | 0.0 |

08 never enters RETURN: return duration and all return activity are zero, not a failed attempted return.

## Beacon turn-around and failure mechanisms

| Episode | First return recovery | First return translation | Terminal distance to active target | Finding |
|---|---:|---:|---:|---|
| 01 | 148.0 s, 4.0 s after beacon | none | 10.527 m home | Right turn interrupted, repeated reversals, then stale turn-direction memory holds indefinitely |
| 05 | 106.0 s, 2.4 s after beacon | 465.9 s | 6.411 m home | Attained visual reference repeatedly holds; route exists throughout |
| 07 | 215.2 s, 3.5 s after beacon | none | 6.569 m home | Repeated turn/recovery reversals, pose loss at 249.0 s |
| 08 | not applicable | not applicable | 9.856 m beacon | Outbound attained-view recovery stall near the first explored dead end |
| 09 | none on return | 463.9 s | 6.609 m home | Normal turn-around and steady return progress, but only 21.9 s remains |

At beacon arrival, the base centre is about 0.60 m from the nearest wall in all four return failures. The evaluator-only primary-camera optical centre ray meets a wall at 0.294 m (05), 0.317 m (07), and 0.294 m (09). In 01 it initially sees down the corridor (1.674 m), but the first interrupted turn faces a wall at 0.271 m. At first return recovery the ray is 0.333 m in 05 and 0.307 m in 07. These are geometric view diagnostics, not contact or articulated-clearance measurements. A nearby beacon wall is not sufficient to predict failure: 09 turns away successfully.

**01 — return routing/recovery-memory deadlock.** No commanded translation occurs after beacon arrival. After turn/recovery oscillation, from 197.6 through 479.6 s, 705 decisions retain an active right-turn override even though that right turn fails its unchanged forecast-clearance rule. The left turn remains admitted; the original selector prefers it. Both directions have historical weak-view interruptions recorded. The active override changes the selection to hold. This specifically resolves the old hold reader’s 705 `insufficient_evidence` rows as an explicit route-turn-memory override; the old result is preserved. From 204 s onward, every 30-second window is zero-command holding. This is not lack of a known route or steady progress.

**05 — return attained-view stall.** The known home route is repeatedly replaced by a weak-support recovery heading. Of the recovery holds, 569 occur within the existing 0.1-rad heading tolerance, with at least one turn still clearance-eligible. The heading selector chooses hold because further turning worsens the already-attained view objective; release also requires the strong-corner count to recover. Repeated recovery retriggers and dispatch-prefix cancellation prolong the standstill. Translation resumes only at 465.9 s. Return time is 306.4 s holding, 64.7 s turning, 5.3 s translating.

**07 — return turn-induced pose loss.** Return never commands translation: 32.4 s turning and 4.9 s holding. It loses pose 37.3 s after beacon arrival. This entire recording is identical to C3, so the already completed bitwise replay and camera diagnosis remain applicable and are not rerun. At failure the optical wall ray is 0.188 m, inside the 0.20-m depth near limit. The primary view is severely restricted; the existing auxiliary fallback rejects its combined consensus/grid/displacement gate. The retained message does not identify which individual subcriterion failed. Both cameras already participate; adding the same fallback again would not address this. The native articulated robot remains safely separated.

**08 — outbound attained-view stall.** Only 6.4 s of translation occurs over 480 s. The robot enters one neighbouring maze cell and then alternates recovery turns and holding. There are 578 recovery holds with heading error at most 0.1 rad and a clearance-eligible turn. It spends 320.0 s under the logged recovery objective and 318.8 s at zero command. This shares 05’s demonstrated attained-reference failure despite occurring outbound.

**09 — budget pressure at the end, substantial earlier turn stall.** Flagged for the user: the return itself makes steady progress with little holding (16.1 s translation, 4.6 s turning, 1.2 s holding), reduces the home path from approximately 9.39 to 6.61 m, and times out with no return recovery or selected holds. However, the whole episode does not satisfy the “steady progress with little stalling” explanation: from 90 to 390 s there is zero commanded translation, remaining beacon distance stays around 6.9 m, and the robot repeatedly turns between route/frontier and recovery objectives. Outbound consumes 352.2 s turning versus 71.8 s translating. Thus this is a late-arrival budget-exhaustion case, with a major recoverable navigation inefficiency; it is not evidence to increase the budget. No budget change is made.

## Does return reuse the outbound route?

The deployed stack does not store and play back a breadcrumb waypoint tape. It retains the observed floor/obstacle map, changes the active objective to home, and replans through that persistent map. The mission transition resets target-specific scan state but not mapping or tracking. On these returns, every scored plan is either an observed-floor route to the home goal or measured-view recovery. There are **no return frontier/re-exploration plans**.

Floor cells retained at the first return plan are 9,457 (01), 8,466 (05), 8,358 (07), and 7,879 (09), exactly matching the last outbound planning receipt. No map reset is observed.

Native trajectories, mapped to the generator’s 1.3-m cells, show:
- 01 and 07 never leave the beacon cell; holding/turning prevents any substantive retrace.
- 05 eventually moves one cell back along the outbound route.
- 09 immediately retraces two cell transitions along the reversed outbound route.
- No return enters a cell absent from its outbound recording. Loop-erased return sequences are prefixes of the reverse loop-erased outbound sequences. This is a coarse topological check, not proof of identical continuous trajectories or an explicit breadcrumb implementation.

The return failures are therefore failures to execute known-route motion, involving recovery/turn memory, rather than demonstrated loss of the outbound map or return re-exploration.

## Selected next change — V2 exhausted measured-view reference

**One shared recovery change:** retire a measured-view reference after accepted visual poses remain within the existing 0.1-rad heading tolerance for one full second but the existing strong-support release has not occurred. Consume that old reference so it cannot immediately retrigger; only a later observation satisfying the unchanged strong-support criterion can establish another reference. Ordinary routing then resumes under every existing eligibility, stopping, dispatch, and pose-admission gate.

The one-second dwell is fixed before runs: ten continuous 100-ms observation intervals (eleven boundary observations), spanning more than two 400-ms planning periods, allow the attained view to be observed again rather than treating arrival at its heading as immediate recovery. A heading departure or a missing observation restarts the dwell. No thresholds are fitted from new runs. This is reference exhaustion, not a claim that weak feature support is safe or that tracking has recovered.

Rationale: it directly targets the largest demonstrated avoidable holding mechanism in two timeouts (05 and 08), also seen outbound in 07. It may reduce oscillation, but it does not directly repair 01’s separate stale route-turn override, prove resolution of 09’s repeated turns, or guarantee prevention of 07’s pose loss. Those mechanisms remain recorded; combining their repairs would violate one change per version.

Tracker estimation, original/weak corner selection, both sensors, candidates, models, map, route cost, clearance thresholds, mission budget and arrival rules remain unchanged. The implementation changes recovery behaviour only, identically for C0–C4. It consumes the third of six outcome-driven versions. The same 10-episode C1 screen and original safety disqualification rules apply; no C0 gate before 9/10.

## Evidence

- Detailed leg/window accounting: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/paired_floor_leg_diagnosis_attempt001/result.json`.
- Per-episode native geometry and override witnesses: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/paired_floor_leg_diagnosis_attempt001/mechanism_witnesses.json`.
- Read-only analysis: `scripts/analyse_go2_capability_paired_floor_legs_development.py`.
- Completed V1 result: `docs/go2_navigation_capability_paired_floor_v1_result_2026-09-27.md`.
- Unchanged 07 replay/pose-loss evidence: the C3 failure diagnosis and retained `grid_c3_pose_loss_diagnosis_attempt001`.
- Every source recording and failed outcome is preserved. Training-render provenance remains unverified; this diagnosis makes no representation attribution.
