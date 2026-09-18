# Goal metric: within-trajectory fit versus opposing-turn separation

Status: **COMPLETE**, diagnostic session 35718 exited 0 in 6.01 seconds.
This uses existing training images only; it fits no model and runs no simulation.

All 24,294 fitted scalar-goal pairs are within individual recordings. Select
four training layouts at appearance seed 2026090940. For each, compare the
common quiet frame 3 to left-turn frame 23 and right-turn frame 23, then compare
those opposing endpoints. The first two pairs are explicitly present in the
fitted supervision; the cross-trajectory pair is absent. Both endpoint images
are training inputs. Common initial RGB matches exactly between turn recordings.

| Training layout | Learned / physical initial-to-left cost | Learned / physical initial-to-right cost | Learned opposing-turn cost | Physical opposing-turn cost | Cross-turn ratio |
|---|---:|---:|---:|---:|---:|
| cluster 00, left opening | 1.019 | 0.990 | 201.949 | 427.812 | 0.472 |
| cluster 00, right opening | 1.040 | 1.043 | 14.058 | 427.812 | 0.033 |
| cluster 01, left opening | 0.967 | 0.985 | 191.310 | 427.812 | 0.447 |
| cluster 01, right opening | 0.987 | 1.030 | 16.887 | 427.812 | 0.039 |

Each turn is about 51–52 degrees from the common start; the endpoints differ
by 103.29 degrees. The metric fits the supervised distances to within about
4.3%, while assigning opposing turns only 3–4% of their squared physical cost
on the right-opening layouts. Mean cross-turn cost ratio across all four is
0.248. These are squared-cost ratios, not linear angular-distance ratios.

This is a measured failure on a missing relationship between known training
images, rather than solely an unseen-layout failure. It supports the hypothesis
that within-trajectory distance supervision does not adequately constrain how
different trajectories are positioned relative to each other in the learned
embedding. Distances along two paths sharing a start can be fitted while their
relative directions are wrong. The result does not prove that this is the
only failure mechanism or that adding pairs will fix navigation.

The next bounded intervention is to mix cross-trajectory and within-trajectory
pairs from the same existing training observations. Match geometry, appearance,
coordinate frame and physical configuration before comparing native poses.
Keep the original architecture, initialization seed, pair count, epochs and
update budget; do not add fresh-task images, refit the encoder/predictor, or
tune against transfer outcomes. Retain the original within-only fit as the
matched reference. Report cross-pair geometry, actual/predicted goal ranking
and prospective control separately.

Source: `scripts/diagnose_go2_goal_metric_turn_separation_development.py`.
Result with exact image identities:
`go2_goal_metric_turn_separation_2026-09-17.json`.
