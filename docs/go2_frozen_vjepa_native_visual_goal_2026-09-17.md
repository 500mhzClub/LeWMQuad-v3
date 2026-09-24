# Decision diagnostic from the existing dense native branch forecasts

The unchanged historical predictors can improve a limited visual-goal choice
even while losing to persistence on factual feature MSE. Prediction error alone
must not be the sole gate for a decision experiment.

This CPU-only analysis reuses the complete 3x3 forecast/target distance matrices
from `go2_frozen_vjepa_native_branches_result_2026-09-17.json`; no model, encoder,
simulation or training was run. For each recorded successor image, treat that
image as a reachable visual goal and choose the candidate with minimum feature
MSE to it. Only after selection, compare the chosen recorded endpoint with the
goal endpoint using the 500-ms target motion labels. Average exact ties uniformly.
The physically optimal endpoint is the corresponding recorded branch, with zero
error, so endpoint error is also regret within this recorded candidate set.

| Model, exposed transfer role | Correct action | Position error (mm) | Heading error (degrees) |
| --- | ---: | ---: | ---: |
| Historical one-step | 11/18 | 0.5350 | 0.9786 |
| Historical rollout | 11/18 | 0.6279 | 0.8801 |
| Persistence, uniform tie expectation | 6/18 expected | 1.0999 | 1.7912 |

Both predictors improve mean position error over the uniform tie reference in
all six transfer history groups. There are only two exposed geometries, and the
18 goal tasks share histories and outcomes; these are not 18 independent trials.
The original persistence score of 0/18 counted **strict** retrieval wins: it
means all actions tied, not that uniform tie-breaking has zero expected success.

This is a retrospective goal-image diagnostic, not the current navigation
controller, which scores physical displacement/yaw forecasts against waypoints.
The goal image is deliberately supplied; it is not an unseen future observation
available to ordinary waypoint navigation. Endpoint differences are tiny, from
short command pulses. There is no demonstrated navigation advantage or isolated
JEPA-training contribution here.

Source inspection identifies two concrete integration gaps: the dense predictor
returns visual tokens at 500 ms, whereas
`lewm/paced_multirate_controller_development.py::_select_action` expects physical
motion/contact outputs at eight 100-ms offsets for six candidates. Its existing
ridge readout consumes a different model's learned decoder features
(`lewm/frozen_motion_readout_development.py`). Swapping checkpoints cannot bridge
either gap. Establish a useful decision interface before another navigation sweep;
do not silently interpolate the dense model into that eight-step contract or
reuse the unrelated motion head.

The native two-arm adaptation continues unchanged. After its completed branch
evaluation, run this same scorer on the adapted result, using a new output path.
Assess action-effect accuracy and decision quality alongside factual feature
error. A reduction in overall MSE alone is insufficient, and failing persistence
on overall MSE alone does not rule out useful action ranking.

Reproduction:

```sh
python3 scripts/score_go2_dense_visual_goal_branches_development.py \
  docs/go2_frozen_vjepa_native_branches_result_2026-09-17.json \
  --output /tmp/go2_dense_visual_goal_reproduction.json
```

Saved result: `go2_frozen_vjepa_native_visual_goal_result_2026-09-17.json`.
The script completed successfully on all 36 branches, with target horizon/role
checks. It preserves source and dataset hashes and per-group, per-goal selections.
