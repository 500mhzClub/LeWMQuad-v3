# Physical turn branches at the dense planner's stall

All three native branches and the frozen-model evaluation completed with exit
code 0. The replay reproduced the original physical poses exactly at all three
context frames (434, 439, 444), and their primary RGB pixels were bitwise equal.
Each branch replayed 2220 recorded 20-ms commands, then executed its saved
800-ms hold/left-turn/right-turn tape. All three completed without contact.
This is a post hoc diagnostic at one exposed context, not a navigation result.

The robot needed a positive viewing-angle correction. During the actual
300–700 ms commitment interval:

| Candidate | Actual yaw change | Decoded from predicted future | Decoded from actual future |
|---|---:|---:|---:|
| Hold | +0.12° | +0.45° | -0.11° |
| Left turn | +9.17° | -0.76° | -1.38° |
| Right turn | -9.24° | -1.10° | -2.61° |

Positive yaw is leftward. The frozen physical readout gets the left-turn sign
wrong even with the actual future images. The error also exists at its trained
500-ms endpoint: actual left yaw was +4.19°, decoded predicted-future yaw -0.32°,
and decoded actual-future yaw -1.43°. It is not solely an unsupported-horizon
effect. Actual future features are unavailable to online control.

In contrast, for each actual branch image, the correct-action latent forecast
had strictly lower MSE than both wrong-action forecasts at all three evaluated
post-branch horizons:

| Horizon | Correct-action forecast wins | Action-blind strict wins |
|---|---:|---:|
| 500 ms | 3 / 3 | 0 / 3 (ties) |
| 700 ms | 3 / 3 | 0 / 3 (ties) |
| 800 ms | 3 / 3 | 0 / 3 (ties) |

The 300-ms futures precede branching and are identical; retrieval is inapplicable
there. The action-blind arm produces identical forecasts for all candidates,
so its ties are expected. Current-image persistence is substantially better on
the stationary hold branch; successful action discrimination does not establish
accurate absolute features or physical predictions for every action.

This narrows the diagnosis: action information survives in the predictor's
latent futures at this failure context, while the current motion readout fails
even on true future features. It does not establish that the predictor is
accurate enough for navigation, that other decoders will succeed, or that JEPA
training caused the discrimination. The next model intervention should address
physical-readout transfer, with encoder/predictor identities retained as the
baseline and the exposed failure context kept out of training.

Artifacts:
`.generated/navigation_development_artifacts_v1/go2_dense_stall_turn_branches_v1_attempt_001/`.
The collection plan, per-action results and `evaluation/result.json` retain the
raw motion comparisons and full latent error matrices. Native collection took
about 41 seconds per branch before persistence; CPU evaluation took 26.2 seconds.
RGB, physics, commands and depth hashes are retained; raw depth arrays are not.
No checkpoint, online controller or prospective maze was changed.
