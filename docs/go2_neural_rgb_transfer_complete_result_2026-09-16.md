# Completed neural-RGB navigation pilot

All 36 fixed assignments completed and were physically evaluated: **35 verified
goal-and-home round trips, one tracking failure, zero disallowed contacts**.
Full-input models completed 18/18 missions; matched no-RGB models completed
17/18. JEPA and direct prediction each completed 12/12, supervised rollout
11/12. These are descriptive results on **two independent development mazes**,
not 36 independent environments or a reliability estimate.

The study used three fixed training seeds, three training methods and two
neural-input treatments. Weights, per-model motion corrections, runtime and
assignment order stayed frozen throughout. All 18 paired reports share the
same 175 common runtime source identities. Actual model, correction and neural
input receipts passed for every assignment. No failed assignment was retried.

The no-RGB treatment removes images from the neural predictor and its matched
training treatment, including JEPA future-RGB targets; it retains camera-based
RGB-D tracking and mapping. This is not an inference-only ablation or a test
of a camera-free robot. Both arms retain predictive clearance/stopping guards,
persistent maps, learned corrected XY and learned yaw, with contact scoring
disabled. This cohort has no new fitted-motion or reactive comparator.

## Outcomes by environment

Times below are mean simulated completion seconds across the successful
training-seed runs. The failed mission stays in its success denominator and
is excluded only from successful-completion time, not counted as a fast finish.

| Method | Neural input | Maze 0 round trips | Maze 0 mean s | Maze 1 round trips | Maze 1 mean s |
|---|---|---:|---:|---:|---:|
| JEPA | Full | 3/3 | 203.08 | 3/3 | 183.56 |
| JEPA | No RGB | 3/3 | 233.08 | 3/3 | 209.57 |
| Direct | Full | 3/3 | 195.38 | 3/3 | 171.72 |
| Direct | No RGB | 3/3 | 249.70 | 3/3 | 157.13 |
| Supervised rollout | Full | 3/3 | 184.41 | 3/3 | 150.81 |
| Supervised rollout | No RGB | 2/3 | 179.57 | 3/3 | 155.15 |

Full input was faster in 12/17 pairs with two completed missions: 4/6 JEPA,
4/6 direct and 4/5 supervised-rollout pairs. The failed pair remains separate.
The third seed completed all twelve missions; full was faster in three of six
pairs. Full supervised rollout has the lowest mean completion time among the
full-input methods on each maze, but this small pilot does not establish
training-method superiority. It provides no JEPA advantage claim.

## What explains the differences

Both route behavior and final stopping matter. In several pairs, most of the
time difference occurs within the final 10 cm of a goal or home. Terminal
turning can be sustained by the interaction between position priority and
the rule that restores heading guidance unless a translation predicts entering
the arrival radius. A separate roughly 3 mm coordinate mismatch can favor
holding while the mission still observes the goal outside its 20 mm radius.
The first-28-run diagnostic found 69 recorded hold decisions in five runs whose
existing forecasts no longer satisfy the consistent mission-coordinate test.
That is post-hoc evidence, not demonstrated navigation improvement.

Other pairs differ during exploration. Third-seed direct maze 0 favored full
input by 96.46 s, with nearly equal terminal time. Its no-RGB controller incurred
96 stopping-projection changes from translation to hold, and more turning.
Third-seed supervised maze 0 instead favored no-RGB by 26.44 s; full spent
31.22 s more on outbound pure turns, with almost identical terminal time.
Guard interventions and changed trajectories do not reveal the outcomes of
unexecuted actions. These opposite effects prevent a simple general RGB claim.

The one failure was second-seed no-RGB supervised rollout on maze 0. It stopped
after 1,015 acquired frames, before either arrival, when raw visual tracking
detected an anchor/increment conflict. Exact replay reproduced all 1,013 accepted
poses and the conflict. An older anchor and recent increment disagreed by
22.250 mm; evaluator-only errors were 3.642 and 21.290 mm respectively, exposing
accumulated drift rather than a lack of all usable image evidence. The failure,
raw replay diagnosis and its depth/comparator remain retained.

## Prediction and timing limits

Per-run executed-window XY endpoint RMSE ranged from 8.746–41.837 mm raw and
5.275–9.651 mm after the frozen motion correction. These use each controller's
own executed trajectories; windows overlap and only the matched command prefix
through 700 ms is evaluated. They are not paired counterfactual candidate
accuracy or a demonstration of calibrated clearance reserves.

Command-based yaw prediction had lower endpoint RMSE than the learned yaw
head in **all 36 runs**. Per-run ranges were 1.289–3.076 degrees for command
prediction and 1.945–5.754 degrees for neural prediction. The deployed treatment
still used learned yaw throughout this cohort. Together with earlier matched
fitted-motion comparisons, this is evidence against assuming that adding the
learned predictor is already a useful navigation improvement.

16,074/16,434 selected plans were on time; 360 were late. Per-run peak simulator
lag ranged from 323.836 ms to 4,349.851 ms. Synthetic independent 2 mm depth noise,
an ideal gyro and measured simulation timing do not establish calibrated
sensing, hard real-time execution or hardware performance. All 35 successful
missions separately passed physical goal/home distance and quiet-dwell checks.

## Artifacts and next experiment

Artifacts are under the configured development-artifact base:

- `go2_neural_rgb_transfer_complete_comparison_v1_attempt_001/`: complete
  `result.json`, `rows.csv`, `pairs.csv`, `by_layout_method_input.csv`.
- `go2_neural_rgb_transfer_seed_2026091402_complete_comparison_v1_attempt_001/`:
  complete third-seed result, alongside the preserved first/second-seed reports.
- Eighteen `go2_neural_rgb_transfer_comparison_seed_*` directories: matched
  results, PNG/SVG trajectories and terminal diagnostics; additional action
  diagnostics identify the detailed cases in the study log.

The aggregation implementation is
`scripts/summarize_go2_neural_rgb_transfer_development.py`; its `collect()`
function recomputes the result without writes. The command-line entry point
preserves the existing completed output rather than overwriting it.
The detailed chronological record remains in
`docs/go2_neural_rgb_transfer_2026-09-15.md`.

The four-assignment controller-correctness follow-up in
`docs/go2_mission_coordinate_followup_2026-09-16.md` is now complete: four round
trips, zero contacts, no demonstrated speed benefit from the coordinate fix.
Observed/physical arrival tolerances were unchanged. A subsequent same-window
readout found the simple fitted XY predictor more accurate in 30/36 pilot runs;
see `docs/go2_neural_rgb_motion_controls_result_2026-09-16.md`.
The larger goal
still requires independent-environment evidence, a demonstrated contribution
from learned prediction against matched simpler controls, realistic sensing
and timing tests, and physical-platform validation. Do not scale training on
the unchanged task merely to search for a favorable seed.
