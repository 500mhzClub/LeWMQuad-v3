# Causal residual diagnostic V1: terminal input-adapter failure

`go2_causal_executed_residual_diagnosis_v1_attempt_001` failed before producing
any per-case result or motion artifact. Recorded JSON turns the runtime tuple
episode identity into a list; the script passed it directly to the unchanged
live `current_joint_pose` validator, which correctly rejected its encoding.
No scientific residual result, model update or native execution occurred.

- Launch SHA-256:
  `b53475869e52057e6e806cd6fd19b58efc384035262be97c1a32d51b68f5708b`
- Failure SHA-256:
  `5218c6ae580b38e2e0f4076bfc7be39110f88a5bab668756910fde65af505957`
- Failure: `SensorContractError('identity must be (environment, episode, reset)')`.

Preserve the original attempt, protocol and source. A separate integrity
replacement restores only a three-integer JSON identity list to its runtime
tuple before invoking the original complete pose/clock/witness admission.
The correction window, algorithm, evaluated models/trajectories and metrics
remain unchanged. Add JSON round-trip and failed-admission tests before launch.
