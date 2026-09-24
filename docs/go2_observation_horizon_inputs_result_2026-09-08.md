# Observation-horizon causal input result

The complete causal input check passed. All 408 training contexts and 420
transfer contexts retain their original past tensors and exact command
prefixes. All 24 six-suffix branch groups have identical history tensors.
The complete 912-slot index and all three unchanged mixed schedules are saved.

Training materialized 3,072 actual short-horizon future observations and
motion targets, plus 3,162 contact targets including 90 positives. Transfer
inference materialized zero future observations or training-target tensors.
Transfer labels remain segregated target-side metadata for later evaluation.
All 84 unavailable contexts remain explicit.

The private untrained width-32 model passed finite output, exact clock/mask
and loss checks for all six input/objective combinations. It performed no
optimizer steps, retained no gradients and kept state SHA-256
`7ee6e3f1966a786aaa707232310543fdae393e3e1f33294c8dad86faa0ff72a8`.
Its weights were not saved and cannot be used as scientific weights.

The check took 99.357 seconds after launch and binds 1,115 source paths.
Three target-derivation tests, seven model/learning/ablation tests and three
causal-stream tests passed. Exact artifact identities under
`go2_observation_horizon_inputs_v1_attempt_001` in the established navigation
development artifact root:

- `launch.json`: `d90efbf41fea4820b1444043b66a2c7f4e191e78a55f76d7cc9772310dec5152`.
- `tensor_index.json`: `b6933c9c1f3c44b305fc24d372ef291840e92800794619c3648e6c3f81015159`.
- `training_schedules.json`: `85f6d9d91558a660ec3988b0cbaf7620a93210ce7f63b5add5201f8ff3fdb0d1`.
- `resource_monitor.jsonl`: `dd31cb1e56e31f6c61ae43b4466d4d273a76bf27cad742a5cd674869284f5781`.
- `result.json`: `73e11e168f933633dbbd3b82668a5b021c076e18bbfc7704987008b4d290c1b5`.

No native execution, predictive-performance comparison, navigation or hardware
qualification occurred. The next step is the separately frozen fitting
benchmark and then the complete prescribed fresh matched study.
