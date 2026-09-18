# Executed motion and stopping forecast diagnosis

The authenticated two-case diagnostic compared only forecasts whose command
prefix actually executed. It found 36 JEPA and 33 direct forecast prefixes.
Mean first-100-ms XY errors were 7.066860 and 7.643677 mm; maxima were 26.488048
and 12.199379 mm. No accepted first step had a sampled native centre crossing
of 0.45 m against its forecast's nearest observed cell. This checks one known
cell per forecast and does not certify all obstacles or continuous motion.

At JEPA's terminal tick 38, the hold forecast's first-step XY error was
11.792777 mm and its 800-ms error was 17.540777 mm. At direct's terminal tick 35,
the corresponding errors were 11.639142 and 14.472118 mm. Both complete
eight-command hold prefixes matched the actual zero-command drain. JEPA's
preceding tick-37 hold also matched all eight executed commands, with errors
26.488048 mm at 100 ms and 39.784351 mm at 800 ms.

The diagnostic's recorded first-step native minimum clearances at the terminal
holds were 0.456104106 m (JEPA) and 0.452181093 m (direct), although both model
forecasts were rejected. A follow-up check of all 401 retained native samples
over the same 800-ms hold, mapped with the recorded initial anchor and map
rotation, found minima 0.454917686 m and 0.445081311 m against cell [11, -2].
Direct first crossed the nominal threshold after 158 ms while requesting zero.
These are nominal centre-clearance outcomes; the original physical-contact
audit passed. Zero-command waiting is not a clearance guarantee.

Next test a bounded active zero-command reobservation state after valid forecast
infeasibility. Keep updating actual primary/auxiliary sensing, map and learned
forecasts, and allow movement only if the original constraints admit an action.
Allow at most ten consecutive wait commands within the existing mission budget.
Preserve sensor/model failures, view exhaustion, arrival, hard mission limits and
the original physical stops. Recorded drain replay can test proposed recovery
and locate its first different command; it cannot establish native recovery.

Root: `go2_auxiliary_depth_executed_motion_diagnosis_v1_attempt_001` under the
guarded development base, with 1,247 bound sources. Two focused target-prefix
tests passed. No model was fitted, changed or called by this diagnostic.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 4079f6ff1008d4ac6d4ecec5740f3609c3ce73c6da467392ac5f0d9b2d5946e4 |
| motion.json | f138bb28f7ce968a313e026fee701412cc333c9bb8bd91ea1a3e84b7f58b0196 |
| result.json | 8e04661518182c57f897fb60ef39f5bcefc52712ed9eede6c5e0555145436d59 |
