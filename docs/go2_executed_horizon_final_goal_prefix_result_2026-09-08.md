# Executed-horizon final-goal prefix result

Both fixed corrected models passed two exact fresh replays. Every causal
observation/map receipt, forecast and surface/nominal check matched the
recorded exact-target controller. Each selection exactly equalled the new
pure scoring transformation. Model states were unchanged; no gradients or
controller failures occurred. Twelve source/native/readout tests passed in
1.98 s before this launch.

JEPA first changed its requested command at tick 171: hold [0,0,0] instead of
forward. The replay included 172 observations and stopped before executing
that changed command or consuming a later observation. Only tick 171 activated
the new score within this prefix. No terminal change occurred.

Direct's complete 57-frame recorded trajectory stayed unchanged in both
passes. It never selected the exact final goal. Its active infeasible waits
remained ticks 36–45 and its original no-feasible-candidate terminal remained.
There is no claimed direct-model recovery.

This is deterministic causal-prefix compatibility, not prospective navigation
or proof that the shorter score improves control. An earlier hold may stall;
the separate unchanged-budget native experiment must establish its outcome.

Exclusive `go2_executed_horizon_final_goal_prefix_v1_attempt_001`, under the
approved artifact base, contains:

| File | SHA-256 |
| --- | --- |
| `launch.json` | `bd6126bbefb67b27c56d03eb784a339fd1ed5849babbeef82340a1330c6e61dd` |
| `seed_2026091001_full_jepa_decisions.json` | `ac89084faddc8b894113081389b1fc99c81f7ade0fc1f09968724fb1dffb3e8e` |
| `seed_2026091001_full_direct_decisions.json` | `d6076302ddb957d7b3a5bf5b48811ded454ac39c7883681e609b20d7139d42db` |
| `result.json` | `290e19f2b537aace3b9c67c9bb11ba12814d60eb65ee7e093d751e40a1d5236e` |

The 1,338-path source closure and all original native/readout/model inputs
were reverified. Replay and verification took 265.01232366799377 s on one CPU
thread. Terminal available RAM was 81,164,472,320 B, artifact free space
62,282,469,376 B, GPU utilization zero. No fitting, native execution, timing
qualification, independent-maze evaluation or full-goal completion occurred.
