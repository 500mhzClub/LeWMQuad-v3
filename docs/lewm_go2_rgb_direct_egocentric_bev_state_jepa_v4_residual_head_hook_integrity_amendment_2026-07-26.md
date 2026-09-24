# Direct BEV V4 residual-head hook integrity amendment

Date: 2026-07-26

Status: preregistered for source implementation, independent review, recursive
source closure, and CPU-only synthetic tests. Execution is not authorized.

## Frozen V3 disposition

V3 is permanently closed by
`docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_coordinate_aware_film_unet_predictor_terminal_audit_2026-07-26.json`:

- commit: `2496bfac12c3841c2ead46cb582bc1a25a9ce2b2`
- raw SHA-256: `c298a56fe3f4c7ab9d7a02447f6dfdd16ad28c0909b6cd67d6c2b0900bd1f324`
- content SHA-256: `ee7caf34174bdab3fbaf8765950140cc09eb618dda59bfc21dc285062e64d203`
- byte count: `7684`
- status:
  `PASS_VALID_UPDATE_ZERO_INTEGRITY_INSTRUMENTATION_FAILURE_CLOSES_V3_NO_RETRY`
- classification:
  `VALID_UPDATE_ZERO_INTEGRITY_HARNESS_FAILURE_OUTER_PREDICTOR_FORWARD_HOOK_INCOMPATIBLE_WITH_DIRECT_ALL_ACTIONS_METHOD_ZERO_SCIENTIFIC_WORK_V3_PERMANENTLY_CLOSED`

V3 performed zero updates, presentations, training objective evaluations,
backward calls, optimizer updates, and EMA updates. Its initial and terminal
model-state SHA-256 were both
`84748bc66f0639b9dae1c81880f5c0fa756f4c4d9e75d0ffddac1310c7d05d0a`.
No V3 checkpoint or training trace may be opened or reused.

## Exact failure and one correction

The update-zero gradient probe observed call counts
`online_state_stack=3`, `predictor=0`, and `target_state_stack=3`. Every
substantive gradient and isolation fact passed: online encoder,
decoder/state, and predictor gradients were finite and nonzero; predictor
absolute gradient sum was `5.07173465937376`; targets were gradient-free;
the next-RGB gradient equalled the G-next-only gradient; fixed-negative RGB
had no optimizer gradient; and the observation-only output was detached.

The count of zero is correct for the inherited hook target but not for the
transition call. V3 deliberately calls
`predictor.predict_all_actions(current_state_logits)` directly so that it can
encode state and coordinates once. That direct method does not traverse
`nn.Module.__call__`, so a forward hook on the outer predictor cannot fire.
The exact V3 `predictor.residual_head` is traversed once by the single batched
nine-action decode.

V4 changes only that call-count witness. During the inherited update-zero
gradient probe, a narrow model view delegates every attribute to the real V3
model except `predictor`, which resolves to
`real_model.predictor.residual_head`. The frozen probe is otherwise called
unchanged. This makes its existing `predictor` forward-hook counter observe
the exact once-per-all-actions residual-head call. The view must be discarded
after the probe and may not enter training, observation, checkpointing, or
model state.

The adapter must fail closed unless all of the following are exact:

- the frozen probe uses `model.predictor` only as its hook-registration
  witness;
- the real predictor is the frozen coordinate-aware FiLM U-Net;
- `residual_head` is the registered `Conv2d(16,3,3,padding=1,bias=True)`;
- an outer-predictor hook observes zero and a residual-head hook observes one
  for `predict_all_actions`;
- every non-`predictor` model-view attribute is object-identical to the real
  model attribute; and
- the adapter changes no output, gradient, parameter, buffer, RNG, state
  dictionary, or call to the scientific objective.

## Frozen science

V4 is a separate integrity successor, not a V3 retry. It reuses the exact
frozen V3 model source; there is no V4 model or model-test source. It must
freshly construct the model and reproduce the V3 initialized-state SHA above.
It preserves exactly the V3 RGB encoder, BEV decoder, three-logit state head,
coordinate-aware FiLM U-Net predictor, optimized all-actions path, N320
initialization, data and mappings, seed and draw order, G + J + C losses,
optimizer and clipping, EMA, schedule, observations, gates, thresholds,
accounting, receipt semantics, and caps.

The single experiment ID is
`go2_rgb_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity`.
Its fresh output root is
`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_direct_egocentric_bev_state_jepa_probe_v4_residual_head_hook_integrity`.
Its preflight environment key is
`LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V4_RESIDUAL_HEAD_HOOK_INTEGRITY_PREFLIGHT_JSON`.

There is exactly one fresh attempt, no retry or resume, and the unchanged hard
caps are 1,000 updates, 16,000 presentations, and 60 GPU-active minutes. V3
terminal gate control strings remain unchanged because V4 tests the exact V3
scientific mechanism. Passing update zero permits only the same V4 attempt to
continue to the unchanged update-100, update-400, and update-1000 gates.

## Exact additive source surface

V4 adds no model source. It adds exactly these eight source files over the 83
frozen V3 sources:

1. `lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity.py`
2. `scripts/run_go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity.py`
3. `scripts/launch_go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity.py`
4. `scripts/check_go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity_source_closure.py`
5. `lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity_contract.py`
6. `lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity_runner.py`
7. `lewm/tests/test_launch_go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity.py`
8. `lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity_source_closure.py`

The recursive V4 source closure must therefore contain exactly 91 sources.
V4 source, review, authorization, and execution may consume only the committed
V3 terminal audit as runtime-result evidence; they may not reopen any V3
runtime output.

## Present authority

Only the eight-file implementation, independent source review, recursive
closure, and CPU-only synthetic tests are presently authorized. Synthetic
tests may instantiate the frozen V3 model and synthetic tensors but may not
open generated inputs, datasets, RGB, labels, checkpoints, traces, runtime
outputs, GPU state, navigation, G2, held-out, or sealed material. Reservation,
training, GPU use, retry, checkpoint qualification, navigation, held-out
access, production, promotion, and deployment require a separately frozen
source closure, independent review, and exact one-shot authorization.
