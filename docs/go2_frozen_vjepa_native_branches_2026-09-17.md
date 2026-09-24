# Existing dense V-JEPA predictors on current native-camera branches

Completed 2026-09-17. Development-only; no training, simulation or navigation.

The surviving August encoder screen, dense predictors, factorial and direct
counterfactual results are relevant evidence. Subsequent planning interfaces did
not establish an end-to-end benefit: the August 20 true-future route scorer
failed, and the August 26 latent-goal cost experiment had an acknowledged
floor-only rendering limitation. These do not invalidate the earlier predictor
results or establish that all their training images were floor-only. See
`go2_active_interface_specification_2026-09-04.md` for the source distinction.

This experiment tests whether the existing predictor works unchanged on the
current native RGB/action population before connecting it to the planner.

## Executed comparison

- Official frozen V-JEPA 2.1 ViT-L image encoder, final dense 768x1024 tokens.
- Existing factorial RGB one-step and RGB rollout predictors, first registered
  seed 2026080901, fixed epoch 21; strict checkpoint loading, no selection based
  on native outcomes. Existing training-only control normalization reused.
- All 36 previously collected short-pulse departures; 18 training-role and 18
  exposed geometry-transfer contexts. Each role has two layouts and six matched
  history groups with three alternative actions. No new independent maze test.
- Context frames 3, 8, 13; target frame 18: 500-ms spacing and forecast horizon.
  Full native 4:3 RGB resized to 512x384, ImageNet normalization, no historical
  square-frame crop. Float32 encoder and predictor, no autocast.
- The predictor consumes deterministic post-limiter candidate command tapes,
  reconstructed with the platform limiter and the causal last applied command.
  Completed tapes were subsequently checked against recorded applied commands.
- All predictions completed before future RGB loading. Same-history equality
  checked across each action group. Score all tokens in the same frozen feature
  space; historical scene-dependent changed-token masks were not transferred.

## Results: exposed geometry-transfer role

| Predictor | 500-ms feature MSE | MSE / persistence | Correct action beats both alternatives |
| --- | ---: | ---: | ---: |
| Frozen one-step | 0.622875 | 1.6061 | 11/18 |
| Frozen rollout | 0.653284 | 1.6845 | 11/18 |
| Current-feature persistence | 0.387830 | 1.0000 | 0/18 (action-independent ties) |

Persistence beats both models in all six matched transfer groups. Centered
action-effect errors are also slightly worse than predicting zero action effect:
ratios 1.0260 and 1.0246. Action discrimination alone therefore does not establish
accurate action effects. The training-role results have the same ordering:
one-step 0.622908, rollout 0.653456, persistence 0.393201.

This is a zero-shot domain/timing-interface transfer diagnostic. It neither
retests the historical encoder geometry ranking nor falsifies the earlier
positive predictor results. Absolute scores are not comparable with historical
changed-token metrics. No planning benefit, geometry retention, or general
JEPA advantage is established by this experiment.

## Execution and retained failures

One GPU process on Radeon AI PRO R9700, four CPU threads, encoder batch one
with identical-image reuse, predictor batch three. The successful evaluation
encoded 64 distinct frames in 24.63 seconds, peak torch GPU allocation 1.55 GiB.
No dense token caches were retained.

Attempt 001 failed at the first predictor call because the new control adapter
preserved float64 instead of casting to float32. Attempt 002 then caught the
requested/applied command mismatch during target-side validation: requested
yaw 0.45 became applied yaw 0.35 for the pulse. Neither failure produced a
completed scientific result. Both failure records and execution logs remain.
Attempt 003 corrected the causal command conversion and completed with exit 0.

Results: `go2_frozen_vjepa_native_branches_result_2026-09-17.json`.
Artifacts: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_vjepa_native_branches_v1_attempt_003/`.
Runner: `scripts/evaluate_go2_frozen_vjepa_native_branches_development.py`.

## Next scientific step

Keep the evidenced frozen V-JEPA encoder. Adapt the existing dense predictor to
available native training trajectories, preserving correct 500-ms image/action
alignment and a matched action-independent control. Reuse existing RGB rather
than collect or duplicate bulk data; cache only what is needed in RAM. Compare
adapted forecasts with these unchanged checkpoints and persistence on the fixed
exposed branches. Then test decision utility in the current navigation interface
if the adapted predictor earns that comparison. This does not resume or alter
any historical frozen training attempt.
