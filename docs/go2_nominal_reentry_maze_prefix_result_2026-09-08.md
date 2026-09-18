# Nominal reentry recorded-prefix result

The new controller reproduced every original decision field except its identity
through decision 406. At observation 407 it first requested a left turn
`[0,0,0.45]` instead of the original zero wait. All **408** causal observations,
map and mission receipts, raw model forecasts, original surface/nominal checks
and phase allowances remained exact. Replay stopped before any outcome of the
new request could be read. No terminal-policy difference occurred before this
intervention, and no native recovery was executed.

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_nominal_reentry_maze_prefix_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `launch.json` | `46b5d5db5ed8ab30dd72565f7f14023f38ff3703bfe5a15b5922b920b3f128ab` |
| `context_decisions.jsonl.gz` | `82a96c730b6153b203bf9bae908d995993dedb39fcdc901bc13f3e2844d5de97` |
| `result.json` | `8ece4c85f9226265b5c6a425ff589071024738c815513cb0f0d34063e02bb77b` |

The replay bound 1,410 sources, used one fresh unchanged JEPA model and took
252.324 seconds after launch. Corrected model state remained
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`, with no parameter
gradients. Output bindings were rehashed after completion. Available RAM/free
artifact space afterward were 81.700 GB and 56.703 GB.

At the intervention, current observed clearance was 0.44447629913788655 m to
cell `[13,-15]`; the original 0.45 m path veto remained false. The left-turn
forecast's all-eight-segment minimum was exactly that starting clearance, and
its first endpoint clearance was 0.44489041360718357 m, a predicted gain of
0.414114 mm. It was the only eligible recovery. Other actions either violated
the phase allowance or forecast later motion closer to occupied cells; a larger
first-step gain by itself did not qualify the right turn.

The very small predicted gain is not a model-error bound or physical safety
certificate. The source rule is an explicit nominal-policy exception. The old
native maze failure stays unsuccessful. A fresh native recovery intervention
must independently verify its unchanged physical/public prefix and subsequent
commands, sensing, constraints and both arrival/path gates. The reused maze does
not become a new independent layout, and no matched-baseline advantage follows.
