# Measured-view arc recovery: exposed failure experiment

The completed fifteen-run comparison left supervised maze 1 holding during
return-view recovery. Both pure turns failed additional reserve checks while
existing arcs passed reserve-recovery and planned-stopping checks. This trial
adds a fallback only for that conflict: a nominally clear hold may become an
already-clear arc turning toward the saved measured heading. Its predicted
heading improvement must exceed hold and zero, and all forecast positions must
remain within the existing 0.20-m local-reference radius. Existing footprint,
reserve, stopping, observation-age and actual dispatch checks remain unchanged.

Nine focused tests passed. Applying the rule to saved predictions and measured
poses identifies 643 eligible plans among 653 recovery holds, first at frame
2232. This is eligibility evidence only; it establishes neither physical safety
nor navigation success for unexecuted actions. The saved diagnostic is
`measured_view_arc_saved_eligibility_v1.json` in the original supervised maze-1
failure directory. That complete failure recording is preserved.

The new experiment uses the same supervised checkpoint, six candidates,
0.8-second prediction horizon, 4800 navigation ticks, layout/physics/appearance
seeds, 2-mm synthetic depth noise, ideal gyro, CPU allocation and extra 20-ms
planning delay. It is one new full closed-loop mission on an exposed layout,
outside the completed prospective cohort. It may diverge before the original
stall; any success alone would not isolate the intervention's causal effect.

Launcher: `scripts/run_go2_measured_view_arc_recovery_development.py`.
Plan: `docs/go2_measured_view_arc_recovery_plan_2026-09-17.json`.
Output: `go2_measured_view_arc_recovery_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
under the existing navigation development artifact base.

Report independent goal/home arrivals, contacts, tracking, planning timing,
fallback selections and dispatches, and view-recovery release. Preserve all
failures. This does not test JEPA superiority, fresh-layout transfer, new
environment families, real sensors or hardware.

Launched in session 37928, owner PID 4082054, on CPUs 8–15,24–31.
The native launch was acknowledged; the owner subsequently exited 0 after
archival, and the full-mission evaluator completed successfully.

## Completed outcome

**Goal reached, no return; budget failure at 480.88 simulated seconds.** The
frame-4792 goal arrival passed physical distance and quiet-motion checks: maximum
native distance during the one-second dwell was 14.16 mm, maximum 100-ms speed
18.76 mm/s, with all requested dwell commands zero. There were zero disallowed
contacts and no tracking failure. Of 1197 plans, 1134 were on time (94.7%).
All raw depth and failure records are retained.

**The new arc fallback was selected zero times.** There were 51 visual-recovery
plans and 14 recovery publications, but this trajectory did not reproduce the
targeted hold conflict. The live efficacy of the fallback remains untested;
the different outcome cannot be attributed to the fallback. Keep it experimental
rather than adopting it as a demonstrated repair.

The apparent stall was mainly alternating turns during frontier exploration,
not sustained hold or weak-view recovery. Frames 800–3796 contain 750 plans:
729 pure turns, 15 holds and six translations; 731/750 were on time. Registered
endpoint displacement over this 300-second interval was only 0.310 m. The
interval contains 181 turn-direction reversals (counted between successive turn
plans, ignoring intervening nonturn plans). Only 18 plans had a translating
candidate passing both forecast clearance and stopping checks.

The existing early heading-release rule activated 88 times, with matching
commands actually applied for 86. Of these releases, 83 were followed by a new
alternative-turn latch within two seconds; median time to the next latch was
1.6 seconds. Thus the release/relatch cycle is exercised in this recording,
unlike the previous four-run heading-release comparison where no eligible
release occurred. These observations motivate a targeted ablation, but do not
prove that disabling release would complete navigation.

Near frame 3800, ordinary reserve-recovery arcs and their stopping checks became
clear, and forward travel resumed. No measured-view arc fallback caused that
transition. The robot reached the goal too late to return.

Detailed evidence: `turn_cycle_diagnosis_v1.json` in this trial directory;
reproducible reader: `scripts/read_go2_measured_view_arc_recovery_development.py`.
The next intervention comparison should isolate the existing heading-release
rule on this now-exposed layout, preserving the original rule and failure as the
reference. Do not combine that intervention with additional model, candidate,
clearance or visual-recovery changes. A new run must report whether the relevant
recovery states are actually exercised; success alone is insufficient attribution.
