# Supervised maze1: feasible movement loses to contact cost

The first original supervised case completed collection, raw audit, exact
physical-prefix comparison and final source/input verification. It made no
arrival or cell crossing. All3000 selected actions were1501 right turns,
1498 left turns and1 hold. Raw sensor/command/model reconstruction and strict
physical visibility pass, with unchanged assigned supervised weights and no
hard measurement failures. The mission exhausted its3000-tick shared budget.
The remaining original layouts2 and3 continue in their fixed order; this first
case cannot establish a three-layout training-objective comparison.

Original cohort root go2_supervised_rollout_mazes_v1_attempt_001; first case
full_supervised_rollout_novel_maze_01. Launch SHA
49a182da3d795f1b31910fb2b6123732db66349dfc011d768bb72fc9732a09e3;
completed first worker SHA
730b2d8a20d680066854427e6a660c58346d296f6073ee12ec3ae984574380e5;
raw audit SHA
4bd54d5454e2665cb354817be2eef447897fa5d3ab4d65541d9368fdcf37de60;
prefix comparison SHA
b585690d9af0400fab25bc6c5109ab1c0f2adce4634dfa6b1208c2d6402ef0bc.
Independent14939 authenticated all1672 sources and18119 output bindings
before/after, original terminal/progress/launch, unchanged assigned model and
saved audit/prefix outcomes. It did not reexecute the completed original worker's
transitive input verifier or raw controller. Worker wall9027.174647341017s,
maximum RSS10,206,183,424 bytes. The first intervention at frame3 preserved900
physics samples,4 public observations,3 prior commands and1 paired forecast bank;
supervised right turn replaced original JEPA left arc and actually completed.

The separate all-selection readout93282 completed at exclusive
go2_supervised_maze01_turn_scores_v1_attempt_001. Result SHA
e546494d75770c174453663d14cb5d38754a8f994bf89ba847ab425dbd3ddf8f;
launch SHA49d9ec85580bb64ec2eb345381c4ce580c1ac1b38f171c75223b9e2b92570554;
1674 bound sources,224.61413529515266s. All3014 observations,3013 completed
commands and3000 selections were included. Every complete saved score output
reconstructed exactly with the frozen original score_waypoint_execution function.
All original source/input bindings passed before/after. Independent24281 then
verified the result,1674 sources, launch and original worker/audit identities,
complete population and aggregate consistency. No raw geometry/controller/model
was rerun by this readout; no parameter was fitted or changed.

Every one of the six actions was feasible at every selection. Translation had
zero phase, measured-surface or nominal-path vetoes. On all2999 selected turns,
at least one feasible translating action predicted greater geometric potential
progress but lost after contact cost. Forward satisfies that comparison at all
3000 selections, including the single hold. The actual turn direction changed
1003 times on consecutive selections. These are saved-score relationships,
not estimates of what an unexecuted translating command would have achieved.

Mean candidate quantities over all3000 selections, in millimetres of potential
or utility except the contact score:

| Action | Predicted distance progress | Predicted alignment progress | 800ms contact score | Utility |
| --- | ---: | ---: | ---: | ---: |
| Forward | 7.0191 | 12.7042 | 0.0209701 | -5.4408 |
| Left arc | 3.5860 | 15.0250 | 0.00966717 | 7.0104 |
| Right arc | 7.4173 | 10.4060 | 0.00682825 | 9.6294 |
| Left turn | -2.0499 | 15.5229 | 0.000984331 | 12.2918 |
| Right turn | 2.1068 | 11.3484 | 0.000900467 | 12.3747 |
| Hold | 0.0460 | 13.6528 | 0.00114455 | 12.3253 |

The existing1.2m coefficient makes the forward contact term average25.1641mm,
exceeding its19.7233mm predicted100ms geometric potential improvement. These
contact outputs are not calibrated probabilities. The scorer intentionally
combines100ms pose utility with800ms contact cost: the original
go2_executed_waypoint_maze_prefix_v1_2026-09-08.md retained the latter to isolate
its pose-scoring change. Do not revise that completed experiment retrospectively.

The next prospective test should align the contact-cost horizon with the actual
100ms command while retaining all existing800ms nominal-path checks, surface
and phase vetoes, forecasts, observer, mission and memory. Reuse the exact
assigned supervised model in a fresh causal replay and stop at the first changed
command. A positive prefix would still need a later fresh physical experiment;
it would not prove better navigation, calibrated risk, physical clearance,
100ms execution or hardware deployment. Preserve the current native queue.

Confirmed total is24 completed raw-audited native episodes and zero verified
round trips. The full navigation and comparison goal remains active.
