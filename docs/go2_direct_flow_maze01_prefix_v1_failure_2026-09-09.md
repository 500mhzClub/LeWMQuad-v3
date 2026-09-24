# Direct-flow maze1 prefix V1: terminal validation failure

Session 40411 exited 1. Preserve the entire original attempt and frozen sources.
Its decision stream contains 215 observations. All 214 preceding complete
decisions matched exactly. At observation 214 the controller advanced to tick
214, returned no failure or terminal, and selected left-turn [0,0,0.45]. The
fallback receipt selected auxiliary camera, reference 213, ANCHOR_MEASUREMENT,
accepted=True with original geometric and temporal limits unchanged.

The runner's extra pose check then failed because it was given the JSON
round-tripped decision. The live sensor identity contract requires a tuple;
JSON necessarily turns that tuple into a list. The exact exception was
`SensorContractError('identity must be (environment, episode, reset)')`.
The controller itself calls the same raw-pose contract before floor registration,
mapping and action selection. Nevertheless this attempt did not complete the
runner's final model/input checks, so it is not a completed verified prefix.
The candidate row and terminal failure were retained. No native run occurred.

| Artifact | SHA-256 |
| --- | --- |
| `launch.json` | `4d9900f40d39eb40d7aa1740b99838756b7dc48cb595fafacae5c8b01fc72d2f` |
| `context_decisions.jsonl.gz` | `d4d90b933d07f18ce86d3dac86763d002a6b0f1e3fccdc4bd79801fadc656521` |
| `failure.json` | `3eae1b519549f7762a3d353cf430ee9856863b9dabf6a233c186aea6a66c87e5` |

All are under `go2_direct_flow_maze01_prefix_v1_attempt_001` in the navigation
development artifact root. The separate V2 runner keeps the live decision for
live contracts and verifies that its serialized form is exactly the recorded
decision. It uses the same observer, controller, model, input population,
comparator, stopping boundary and unchanged rules. This is an explicit runner
correction with a new protocol/output identity, not an automatic retry or a
change to the failed attempt. Re-execute the fixed prefix because V1 did not
reach its final weight and input authentication. Preserve negative outcomes.
