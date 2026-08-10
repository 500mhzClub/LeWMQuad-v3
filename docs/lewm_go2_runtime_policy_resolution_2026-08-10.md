# Runtime policy path resolved; snapshot schema versioned to v1.1

Date: 2026-08-10
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.**

Frozen baseline `cd9ecee` and all listed digests preserved. No world-model
checkpoint was loaded, no scorer-fit data generated, no scorer trained, no frozen
scientific result altered.

---

# 1. Exact deployed policy call graph

The deployed callable is this project's own adapter, **not** an upstream example or
a training-time actor path: `lewm_genesis/lewm_genesis/rollout.py::GenesisGo2PPOPolicy`
— the same rollout stack that generated the existing locomotion corpus.

| stage | location |
|---|---|
| checkpoint load | `rollout.py:599 _load_policy` → `OnPolicyRunner.load(model_500.pt)` → `runner.get_inference_policy(device).eval()` |
| observation assembly | `rollout.py:740 _build_policy_observation` |
| policy invocation | `rollout.py:707–715` (deduplicated unique-row path) and `rollout.py:726–732` (batched path) |
| action post-processing | `rollout.py:736–738` |
| actuator application | `rollout.py:739` — `target_policy_order[:, self._rollout_from_policy]` |
| buffer update | `rollout.py:738` — `self._last_actions = actions.copy()` |
| reset hook | `rollout.py:634–643` — clears `_last_actions` on episode reset |

# 2. Resolved 45-D observation layout

| slice | semantic | scale |
|---|---|---|
| [0:3] | body angular velocity | `ang_vel` 0.25 |
| [3:6] | projected gravity (body frame) | — |
| [6:9] | commanded (vx, vy, yaw_rate) | `command_scale` |
| [9:21] | joint position − default | `dof_pos` 1.0 |
| [21:33] | joint velocity | `dof_vel` 0.05 |
| **[33:45]** | **previous policy action** (`self._last_actions`) | — |

The composition I inferred last pass from the dimension count is **confirmed by the
source**: the final 12 dimensions are the previous policy action.

# 3. Inference mode

`self._policy(obs_dict, stochastic_output=False)` at **both and only** call sites,
on a module that has been `.eval()`-ed. **Deterministic mean; zero sampling call
sites.**

Therefore **policy RNG is IRRELEVANT**, and `distribution.std_param` is a
compatibility-bound parameter rather than runtime state. Note the wrapper was *not*
modified to force determinism to simplify replay — it is already deterministic as
deployed, which is the outcome the instruction wanted preserved.

# 4. The four previously unresolved fields — all resolved

| field | class | resolution |
|---|---|---|
| **previous policy action** | CONTROLLER_SERIALISED | `self._last_actions`, `(n_envs, 12)` float32 in policy joint order. It is the **raw actor output**, before `action_scale` and the default-pose offset. Written *after* the executed action is chosen, so the observation at tick *k* reads the action chosen at *k−1*. Exactly one buffer — no `last_last_actions` |
| **action latency** | CONTROLLER_SERIALISED (same buffer) | `simulate_action_latency: bool = True` — **on by default**. `exec_actions = self._last_actions if self.simulate_action_latency else actions`. No separate delay queue, actuator filter or smoothing exists |
| **policy RNG** | IRRELEVANT | deterministic mean at every call site |
| **gait phase** | IRRELEVANT | no phase/gait/clock/cadence/oscillator variable in the executed control path; the only "cadence" hits are documentation of the 10 Hz emission rate |

**The single most important finding:** `_last_actions` serves *two* roles at once —
it is both observation dimensions [33:45] **and**, because latency is on by default,
the action actually applied to the actuators. Omitting it from a snapshot would
corrupt both the policy input and the applied torque, while a solver-only replay
check would pass. That is precisely the silent-divergence failure this audit
existed to find.

# 5. State inventory v2 — `9b08939adabff5650b570c7f7b806524c0a0a38332946cd6d715313823e3595a`

Supersedes `82c034d7…`. **`unresolved_fields` is empty; the gate is lifted.**

# 6. Snapshot schema v1.1 — `b586fb9c39a7e05a69fa3ae66eaf790d8987505045464b2db983a91902abc2eb`

v1 (`b18c49ea…`) is preserved unchanged. It had a real defect: its canonical digest
bound only the solver bytes and harness JSON — **the controller/RNG payload was not
bound**, so a snapshot with a corrupted `_last_actions` would have verified clean.

v1.1 binds all three layers with **domain-separated, length-prefixed hashing**
(`len(tag) || tag || len(payload) || payload` over `GENESIS_SOLVER_V1`,
`HARNESS_STATE_V1`, `CONTROLLER_RNG_V1`), removing the concatenation ambiguity. The
compatibility digest additionally binds Genesis version and backend, scene geometry
and sim options, policy weight digest, the 45-D observation layout, the inference
mode, command/action scaling, controller decimation, and executed source digests.

---

# 7. What was NOT done, and why

Parts 4 (implementation), 5, 6 and 7 — the working snapshot, omission-sensitivity
controls, deterministic replay qualification, branch-order invariance, and the
20-state oracle pilot — **were not executed**.

The blocking gate from the previous pass is lifted, and the schema is now correct,
but the implementation itself is a substantial build: snapshot capture/restore wired
into `GenesisGo2PPOPolicy` and the rollout loop, scene loading, the rendering path,
BFS-derived progress labels, oracle utility, and H=1–4 spatial-label capture. None
of it exists yet. I stopped rather than produce a partial snapshot that would pass
its own tests while omitting state — which is the exact failure mode section 4
identifies.

**No sensitivity results, replay results, pilot results, identifiability verdict or
spatial-label coverage are reported, because none were produced.** Runtime and
storage estimates are **unchanged at ≈ 40 h / ~22 GB**, since no pilot throughput
was measured.

# 8. Remaining blockers

1. **Snapshot implementation unbuilt** — schema and inventory are now complete and
   correct, so this is ordinary engineering rather than an open question.
2. Downstream and unchanged: no leakage-free predictor score exists; H=2–4 spatial
   labels are absent from the temporal corpus.

## Stopping condition

Nothing is running. No scorer-fit or evaluation corpus generated, no scorer trained,
no predictor scored or retrained, no frozen result altered.
