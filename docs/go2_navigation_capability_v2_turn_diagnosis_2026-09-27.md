# V2 remaining turn failures and V3 change

Read-only diagnosis of completed 01/0 and 09/0 confirms different dominant mechanisms. No new physics or model calls were used.

| Episode/window | Translation | Turning | Holding | Direction reversals | Absolute/net yaw travel |
|---|---:|---:|---:|---:|---:|
| 01 return, 144–480 s | 0 s | 47.6 s | 288.4 s | 36 | 21.119 / 0.997 rad |
| 09 outbound, 90–390 s | 0 s | 274.4 s | 25.6 s | 218 | 111.663 / 0.079 rad |

Times classify applied commands; yaw is evaluator-only native motion sampled every 100 ms. The small net displacement (0.172 m and 0.118 m) is turning drift, not forward route progress.

## How turns are selected, latched and cleared

The ordinary selector scores predicted progress/alignment. The clearance-turn layer can latch a clear alternative direction when the preferred direction is blocked, including a long-way turn. A measured weak-view recovery can replace the route objective and clear that clearance latch. A separate visual interruption memory remembers route turns interrupted by weak support and may latch the other direction. It clears on mission-generation change, a changed local target/location, measured heading completion, successful translating selection, or a new visual-recovery interruption.

That last visual-memory layer does **not** clear merely because its latched direction becomes ineligible. In 01 it retains a blocked right turn and substitutes hold for the ordinary selector's eligible left turn. From 197.6 s onward, all 705 such holds record a blocked latched right turn, a clear left turn, and both directions in the previous visual-interruption history. Its visual memory is active in 721/839 return selections; only five return selections have an active clearance-turn latch. V2 did not retire any attained reference here.

09 is predominantly a different oscillator: 447/750 selections in the 90–390-s window have an active clearance-turn latch, with 104 new clear-alternative latches. The route/frontier objective and weak-view objective repeatedly interrupt/reverse one another. There are 218 executed turn-direction reversals and nearly zero net yaw, so these are repeated oscillations, not sustained exploration or productive forward motion. Only 12 selections use the visual turn-memory latch and only two hold under it. This is not the same persistent blocked-latch failure as 01. The reference-exhaustion rule does not trigger because this case keeps turning rather than remaining aligned for its one-second dwell.

The return in 09 still makes progress after beacon arrival at 458.1 s; the preceding five-minute oscillator is why so little mission time remains. The 480-second budget stays fixed.

## Single V3 intervention

Release a visual route-turn-memory latch when its selected direction becomes ineligible under the unchanged current forecast-clearance rule. Then run the existing selection/memory logic normally. Keep historical interruption records, pose admission, tracker estimation, both sensors, candidates, models, routing, reference exhaustion, clearance thresholds and dispatch guards unchanged.

This directly addresses 01. It may affect two holds in 09, but no claim is made that it fixes 09's separate clearance/view oscillator. Combining another recovery-policy repair would violate the one-change rule. One additional success is sufficient to qualify for the newly approved second-episode check.

Focused checks reproduce the old blocked-latch hold despite an eligible opposite turn, verify the new selection preserves every clearance mask, retain an eligible latch, retain hold when no turn is eligible, and preserve visual-interruption/mission-reset behaviour. Four tests pass before execution.

V3 is the fourth of six outcome-driven versions. Run the same ten first-episode C1 assignments once. If at least 9/10 pass safely, run the ten second-episode C1 assignments on this same harness, requiring 9/10. Only then run C0's unchanged 19/20 gate and strict first-episode RGB-D replay. If the second-episode check fails, future versions screen all 20 with an aggregate 18/20 threshold. No validation runs before the oracle gate passes.

Evidence: `v2_turn_diagnosis_attempt001/result.json` under the programme artifact root; source `scripts/diagnose_go2_capability_v2_turns_development.py`. All prior failures and source versions remain preserved.
