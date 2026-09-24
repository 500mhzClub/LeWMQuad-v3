# V18 object-space height-volume command-integrity replacement V3 preregistration

Date: 2026-07-30

Status: preregistered science-identical command-integrity replacement only.
No V3 reservation, runtime-input access, GPU work, training, checkpoint,
qualification, calibration, G2, navigation, or held-out access has occurred.

## Trigger and terminal predecessors

- The original V18 attempt and integrity replacements V1 and V2 are terminal.
  None may be retried or resumed.
- Original V18 and V1 failed before any completed training presentation because
  of separately frozen adapter-surface omissions. Those omissions are closed
  by the V2 source and its independently verified 26-of-26 training surface.
- V2's complete command-failure audit is frozen in commit
  `432b356f545a539f2cfcbb2b3d50895a03af1c73` at
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v2_terminal_failure_result_2026-07-30.json`,
  file SHA-256
  `350e3e3c73e0c5ed5d9b7c8d9045661aba1d471c58439c576f15a63ee9b6e114`,
  byte count `7072`, content SHA-256
  `8435d611d11bcda2deb42b27725ab14269fd93a839cceb5ae98809b118a92271`.
- V2 reserved successfully and revalidated its certified source and authorized
  runtime inputs, then failed the exactly-one-visible-GPU guard during runtime
  composition. The invocation defined blank `ROCR_VISIBLE_DEVICES` and
  `CUDA_VISIBLE_DEVICES` alongside `HIP_VISIBLE_DEVICES=0`.
- V2 constructed no V18 model and completed zero observations, updates,
  presentations, microbatches, predictor training forwards, backward calls,
  optimizer steps, EMA steps, or produced checkpoints. No scientific gate was
  reached. Its failure is not evidence for or against the V18 hypothesis.

V3 is a distinct science-identical one-shot root, not a retry or resume of a
consumed predecessor.

## Sole operational correction

- Preserve the V2 model, V18 runner, inherited tensor/training mechanisms, and
  hardware-detection code byte-for-byte.
- Launch with Python isolated and bytecode-disabled modes (`-I -B`).
- Set only `HIP_VISIBLE_DEVICES=0` for the authorized R9700.
- Explicitly remove, rather than define as empty, all conflicting compatibility
  selectors: `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
  `GPU_DEVICE_ORDINAL`, `HSA_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`,
  `NVIDIA_VISIBLE_DEVICES`, `ONEAPI_DEVICE_SELECTOR`, and `ZE_AFFINITY_MASK`.
- Update the source-only isolated-launcher regression to exercise that exact
  environment rule. This changes no executable scientific mechanism.
- Change only V3 preregistration/evidence bindings, schema/evidence prefix,
  experiment-arm identity, and fresh output/source roots.

The exact launch form is:

```text
/usr/bin/env -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL -u HSA_VISIBLE_DEVICES -u HSA_OVERRIDE_GFX_VERSION -u NVIDIA_VISIBLE_DEVICES -u ONEAPI_DEVICE_SELECTOR -u ZE_AFFINITY_MASK HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B /home/andrewknowles/Workspace/LeWMQuad-v3-v18-object-space-height-volume-integrity-replacement-v3-source/scripts/launch_go2_rgb_object_space_height_volume_joint_jepa_v18.py --future-authority /home/andrewknowles/Workspace/LeWMQuad-v3-v18-object-space-height-volume-integrity-replacement-v3-source/docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v3_execution_authorization_2026-07-30.json
```

No model code, tensor operation, architecture, data field, loss, coefficient,
seed, schedule element, optimizer setting, EMA operation, observation, metric,
control, threshold, stopping rule, accounting multiplier, or runtime hardware
guard may change.

## Frozen scientific identity

Preserve original V18 preregistration commit
`5522b226e845907b091ff98ebac3b6f6315a4ca7`, V1 preregistration commit
`402f61522d59943e0def9df0b90ebf785867d366`, and V2 preregistration commit
`baad8efaf524bb3f88f2d4516db7ef368f15684e` exactly, including:

- learned RGB unified ray field plus the explicit eight-height object-space
  volume as the sole JEPA perception state;
- 64-by-64 XY geometry at 0.1 m, eight fixed height centres, five voxel input
  channels, 8-channel 3D residual volume, and height-major 64-channel flatten;
- jointly trained online encoder, volume representation, semantic decoder,
  action-conditioned predictor, and stop-gradient EMA target;
- all V18 parameter counts and initialization, the same N320 initialization,
  train/checkpoint-selection roles, labels, RGB, camera metadata, and inputs;
- constructor, schedule, experiment, bootstrap, projection, and volume seeds;
- four microbatches of four, float32 AdamW, frozen parameter groups and learning
  rates, clipping, losses/weights, and one optimizer plus EMA step per update;
- observations at updates `0`, `100`, `400`, and `1000`;
- exact update-100 health checks, five-check update-400 falsification gate, and
  unchanged V14 final gate;
- maximum 1,000 updates and 16,000 presentations; and
- checkpoint publication only after a complete passing update-1000 result.

There is no model/data retry, search, architecture change, automatic resume,
extension, or gate reinterpretation.

## One-shot V3 identity

- Schema/evidence prefix:
  `lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v3`.
- Exact fresh attempt root:
  `.generated/go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v3/attempt_v1`.
- Exact clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v18-object-space-height-volume-integrity-replacement-v3-source`.
- There is exactly one V3 attempt. Both roots must initially be absent. Retry
  and resume are false.
- Predecessor outputs are closed. Only committed preregistrations and terminal
  results may enter as source-only witnesses; no predecessor authority,
  certification, runtime artifact, model state, tensor, or checkpoint may be
  consumed.
- Any V3 source, authority, reservation, custody, exception, or gate failure is
  terminal and publishes no failed checkpoint. No V4 replacement is
  preregistered.

## Authority boundary

Implementation, regression, recursive source closure, independent source
review, narrow clean-export certification, and one-shot authority must be
frozen before reservation. Until a complete passing update-1000 development
result, probability calibration, G2, navigation, held-out, sealed, production,
promotion, deployment, retry, resume, recovery, and extension remain forbidden.
