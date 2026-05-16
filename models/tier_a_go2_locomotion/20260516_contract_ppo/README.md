# Go2 Locomotion Contract PPO

Frozen Genesis-native Go2 PPO checkpoint trained on the LeWM trainable velocity
primitive set.

- Training run: `lewm-go2-contract-20260516T163413Z`
- Checkpoint: `model_500.pt`
- Config: `cfgs.pkl`
- Checkpoint SHA256:
  `e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff`
- Config SHA256:
  `bc3e68c18252475199e57b30c8ac49d813e3c784a3983e0e8b1a762490dde24f`
- Validation:
  `scripts/check_genesis_go2_policy_contract.sh --exp-name lewm-go2-contract-20260516T163413Z --ckpt 500`

Validated command contract:

- Command vector order: `[vx_body_mps, vy_body_mps, yaw_rate_radps]`
- Trained/validated primitives: hold, forward slow/medium/fast, backward,
  yaw left/right, arc left/right
- Nonzero lateral `vy_body_mps` is not validated and is clamped to zero in
  `config/go2_platform_manifest.yaml`.
