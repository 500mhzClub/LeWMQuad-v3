# Online local-choice adapter: replay qualification result

The strict adapter is implemented and its fixed qualification passed all 72
replay choices. Each of the six actual bound checkpoints reproduced every saved
per-seed, five-candidate rollout prediction **exactly**: maximum absolute
difference 0.0, below the predeclared 1e−5 tolerance. Independent ensemble
probability/cost reconstruction and action selection matched as well.

The replay exercised eight development-validation canonical contexts, three
intents and three methods. It used the actual strict packet reader and the
new runtime factory; no fitting, weights change, physical execution or fresh
generalization occurred. The three-seed ensemble is a fixed new policy, not
selection of the best individual seed.

The learned adapter measures about 8.8 ms median for packet validation, tensor
conversion, all five candidates through all three members, output hashes and
selection on one CPU thread; all-stop is about 2.0 ms. These measurements exclude
RGB capture and are not a claim of physical real-time sensing. In this replay,
supervised rollout chose movement in 18/24 intents and stop in 6/24; JEPA chose
reverse in 16/24 and stop in 8/24. These observed choices do not themselves prove
realized progress or safety.

Seventeen new synthetic tests cover contract rejection, complete/current
histories, episode/reset/one-shot semantics, command slew and release timing,
mean-probability ensemble arithmetic, ties, stop parity and model-byte guards.
The old model/loader/training sources are unchanged and remain hash-bound.

Output: `.generated/go2_online_local_choice_replay_development_v1_attempt_001`.
Session 60888 terminated with exit 0.
Result SHA-256: `9f709223ab1a6f38c0c8f0bbde41122567d0716878f9199d7842d5669c7bfe25`.

The separate [fresh online conditional-choice pilot](go2_online_choice_maze_pilot_development_v1_2026-09-05.md)
has been launched after this prerequisite. It evaluates the actual current
packet-to-command path on eight disjoint procedural layouts, retains all 72
fixed trials, and requires raw physics plus policy-replay auditing. Results
must remain unclaimed until that experiment and its audit finish. Neither
package completes receding-horizon control, online memory, maze exploration,
beacon discovery/return or hardware transfer.
