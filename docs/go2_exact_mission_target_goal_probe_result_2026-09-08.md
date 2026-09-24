# Exact mission target: native integration result

Both fixed corrected models completed collection and raw audits. **0/2 fully
verified goals.** This reused integration layout is not an independent maze.

| Condition | Native minimum distance | Native terminal distance | Terminal |
| --- | ---: | ---: | --- |
| Full JEPA, seed 2026091001 | 0.03971029335649642 m | 0.042645782907013687 m | Mission tick budget exhausted at 243 |
| Full direct, seed 2026091001 | 1.1011647990168776 m | 1.1011647990168776 m | No feasible phase candidate at 46 |

JEPA executed 253 commands and acquired 254 paired frames over 13,400 physics
samples, including the original ten terminal zero commands. Its observed
distance first entered the controller's 4-cm radius at tick 242; tick 243 had
only one qualifying quiet interval when the unchanged budget terminated the
mission. Native distances at subsequent frame boundaries exceeded 4 cm from
tick 244 and settled at 4.2646 cm. Merely extending dwell would not establish
the unchanged controller condition. The evaluator-only 6-cm one-second quiet
test passed, but the required controller arrival terminal did not. Do not
count this as a verified arrival or change the tolerance retrospectively.

Direct executed 56 commands, acquired 57 paired frames and recorded 3,550
physics samples. Its ten active infeasible waits were ticks 36–45. No physical
or acquisition stop occurred in either case. Raw sensor reconstruction,
strict visibility and exact fresh model-command replay passed. Both model
states stayed unchanged. All 311 auxiliary images passed visibility with
zero robot-occluded pixels; maximum observed XY errors were 2.8750 mm for
JEPA and 1.3957 mm for direct. These are measured integration results, not
hardware calibration or prospective error bounds.

Relative to the observed-floor-contact predecessor, JEPA's first command
change was at tick 176 and first terminal change at 226. The common 177-frame
prefix had exact native/public histories, RGB, observation memory and learned
forecasts. Direct's complete 57-frame comparison was unchanged. The new model
pair first changed commands at 26; its 27-frame causal prefix had exact raw,
public, RGB, auxiliary and map evidence, with different learned forecasts.
The same-model readout did not separately compare auxiliary archive hashes.

The JEPA median complete command iteration was 815.689325 ms (maximum
1181.387205); direct was 804.500270 ms (maximum 1028.905348). Every command
iteration exceeded 100 ms. Acquisition medians alone were 207.758776 and
204.109170 ms. Physics was paused during computation; real-time execution
remains unqualified. Collection plus audits took 426.5111534530297 s under
the fixed two-process CPU arrangement.

Read-only exploratory rescoring retained the full 800-ms contact penalty and
all original feasibility checks while evaluating final-target pose progress
at the actually executed 100-ms horizon. On recorded JEPA ticks 200, 210, 225,
235 and 241 this favored left_arc instead of the old hold/right_turn. At 171
it favored hold instead of forward, and at 176/190 it favored left_turn instead
of hold. This is not executed recovery evidence. It motivates a separately
named prospective scorer, activated by the existing observed exact-goal
connector, with no altered goal tolerance, budget or model.

Artifacts are direct children of the approved navigation artifact base:

- `go2_exact_mission_target_goal_probe_v1_attempt_001/launch.json`:
  `90cf9716165c9564e6434284692173742fa520ae477768af7f5b24f48de6d50f`
- Native `result.json`:
  `d9e0cef6a7e66459a5cc6d21cc6cf6e638666a33a12cdcfb8b2364f1477bd125`
- `go2_exact_mission_target_goal_readout_v1_attempt_001/launch.json`:
  `624485137a83dabbeafbe5765217c8ddaad044217a0a0dbe0ec0bd5e470814ce`
- Readout `result.json`:
  `d13c20027d16f48e146a542c443af61e205e02420bcf27e7e7cfa2ace7d61a43`

Native source closure contains 1,330 paths; readout contains 1,333. Original
attempts, failures, source and input identities remain immutable. No fitting,
checkpoint selection, independent-maze evaluation or navigation qualification
was performed. Exploration, physical backtracking, matched baselines and
bounded real-platform evidence remain outstanding.
