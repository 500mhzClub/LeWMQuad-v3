# RGB Swept-Progress Survival Joint-JEPA V7 Hierarchical CNN Encoder — Execution Binding

- Status: frozen before the single V7 GPU execution.
- Preregistration commit: `34c4a33e2fa25926b3127e0c893755757426cfd4`.
- Model and model-test commit: `79bbc50f57bc6a6ca20b77d85c5f86dc740e77f5`.
- Training-wrapper, executor, and focused-test commit:
  `23f1d97bd148a9554715ad4c670c41cee1bca0e7`.
- Independent exact six-file committed-source review: PASS with zero science or
  integrity blockers and zero worktree drift.
- Focused V7 model/wrapper/executor suite: 14 passed. Complete relevant
  V1--V7 swept-progress regression suite: 151 passed.

## Frozen V7 source

- Model SHA-256:
  `16de00df1a40e045a56487a2e87aaf6d8d1e203c3487fefcddf91d493a3697a6`.
- Model-test SHA-256:
  `3e119828d69762e3a6eba269033c3e01e530b65dcb640fea58c329b5bf5ce2c7`.
- Training-wrapper SHA-256:
  `b462225464b3d61b461e5dbb742e3b1333ced44e1d408190a79053390539d8ad`.
- Training-wrapper-test SHA-256:
  `81c92321ffc72d60c69b15b794954504dd978c4c4b4f2306cb7a32cb0916b7ed`.
- Executor SHA-256:
  `c83d603d786e5a2918770a6c9f68a132bed454f8cfcdccf347ae610dd1455357`.
- Executor-test SHA-256:
  `35524eec98d581eec73b3160c76e6ac67cdf0de127913593d3b02b786b33c2e2`.
- Preregistration SHA-256:
  `5158222471bddefe6dfcded3b52533c66ecf31f892f428a91b52be73ee15a3d1`.
- The sole scientific change from clean V4 is wholesale replacement of the
  online and EMA-target VisionEncoder with the preregistered 1,994,880-parameter
  hierarchical CNN. It consumes the same normalized `112x112` RGB and returns
  the same mean-CLS plus 256-spatial-token `[B,257,192]` interface.
- No N320 parameter value survives in the CNN. A schema-valid accepted N320
  state is still transiently validated and loaded by the inherited constructor
  before both inherited encoders are replaced; the receipt records this
  numerical/state independence rather than claiming zero N320 access.
- The V4 BEV lift, residual semantic decoder, predictor, survival head,
  target-BEV path, and every non-encoder initial state tensor are checked for
  exact equality against a fresh clean-V4 construction before GPU transfer.

## Frozen execution

- Model and clean-V4 reference construction occur on CPU. The online CNN is
  initialized under isolated CPU seed `20260715`; the target is its exact
  frozen copy and follows the inherited EMA. Only after initial equality
  checks is V7 transferred to the single visible GPU.
- All 1,994,880 online CNN parameters enter the inherited encoder optimizer
  and clipping group exactly once. All target parameters remain frozen and
  excluded. Per-update receipts require finite gradients for all online
  tensors, prove every tensor becomes active, and require zero target
  gradients.
- Data, labels, action order, masks, optimizer hyperparameters, clipping,
  schedule, controls, metrics, bootstrap, terminal selection, and the complete
  unchanged V4 24-check full-arm gate are inherited exactly.
- Training remains the joint V4 `S+P+U+R+O` objective with `O` coefficient
  `0.5`. V5 loss `H` and the V6 fine-RGB branch are absent. The semantic
  representation and action-conditioned predictor train jointly from update
  one.
- Cap: exactly 1,000 optimizer/EMA updates, 4,000 size-four microbatch
  graphs/backward calls, and 16,000 presentations. There is no retry, resume,
  extension, alternate seed, intermediate selection, or CNN variant.
- Prelaunch inspection found exactly one visible
  `AMD Radeon AI PRO R9700` with `34,208,743,424` bytes total VRAM. The launch
  restricts the process to that device with `HIP_VISIBLE_DEVICES=0`.
- The write-once output root
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder/attempt_v1`
  was confirmed absent immediately before this binding.
- Checkpoint and trace are written after terminal training and before
  evaluation. They remain development-only, unqualified, and non-resumable.
- Exit code `2` with a complete `result.json` is a valid scientific full-arm
  failure. It closes V7 without calibration, tuning, retry, or checkpoint use.
- A 24/24 pass stages only a separately frozen one-shot application of the
  unchanged four-parameter calibration and 2,016-tuple physical gate. It does
  not qualify the checkpoint or open G2, navigation, held-out, sealed,
  production, deployment, or promotion access.
- Exact command:
  `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder.py`.
