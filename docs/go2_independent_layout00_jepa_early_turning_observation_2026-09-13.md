# Early turning in the first independent JEPA trajectory

This is a read-only observation of completed recorded decisions from the live
layout-0 full-RGB JEPA case. It is not a terminal navigation result, a policy
change, or a prediction of what another policy would physically achieve.

Across inclusive frames 200–500, the 301 commands were:

| Command | Count |
| --- | ---: |
| Right turn in place | 151 |
| Left turn in place | 50 |
| Left arc | 54 |
| Right arc | 20 |
| Forward | 17 |
| Hold | 9 |

The observed x range was 0.795642–1.220120 m and y range was
0.542186–1.137740 m. Observed distance to the mission goal was 2.240753 m at
frame 200 and 2.384343 m at frame 500. These endpoint distances alone do not
measure exploration progress: the observed map's frontier target also changed.
For example, the selected frontier target moved from [0.375, 1.325] at frame 200
to [0.475, 0.075] at frame 500. Both proposals reported an observed-floor route
to a frontier. No arrival or terminal state had been recorded at frame 500.

At frame 500, the selected command was a right turn in place. All six candidates
were recorded as phase-admissible and without sampled surface conflicts. The
waypoint was [1.075, 0.175]. The existing scoring rule combines residual-corrected
0.1-second waypoint distance/alignment progress with 0.8-second predicted contact:

| Candidate | Distance progress, m | Alignment progress, m | Full-plan contact score | Final utility, m |
| --- | ---: | ---: | ---: | ---: |
| Forward | 0.02184414 | 0.00326623 | 0.02631618 | -0.00646904 |
| Right arc | 0.01830074 | 0.00691799 | 0.02043164 | 0.00070077 |
| Right turn | -0.00046115 | 0.00708974 | 0.00386839 | 0.00198652 |

Utility is distance progress plus alignment progress minus 1.2 times the
full-plan contact score. The scores are uncalibrated model outputs, not measured
collision probabilities. In this decision the longer-horizon contact term is
large enough to favor turning over the available translating alternatives.

This is a hypothesis about one contributor to the repeated turning, not proof
that the scoring rule is wrong or that a different score would complete the
maze. Frontier selection, observed residuals and physical dynamics also matter.
The frozen run and all queued controls are unchanged. Compare the completed
JEPA, reactive, supervised-rollout and direct trajectories before deciding
whether a separate scoring-horizon experiment is warranted.

Input is the completed prefix of `context_decisions.jsonl.gz` in
`go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_jepa_v1_attempt_001/independent_00_frozen_reference_seed_2026091001_full_jepa`.
Only recorded public controller decisions were inspected. No native state was
fed to a controller and no post-change trajectory was inferred.
