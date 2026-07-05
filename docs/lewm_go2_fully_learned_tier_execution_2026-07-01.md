# Go2 Fully Learned Nav — Tier Execution Report (2026-07-01, evening)

Executes the plan from `lewm_go2_fully_learned_pause_handoff_2026-07-01.md`:
replacement-safety cap (Tier 1), then the indicated Tier 2 training move.
All runs: kinematic mode, seed 1, fixed target order, 700-tick budget,
generalized runtime contract, held-out scene `medium_enclosed_maze_eeb8320a6934`.

Artifacts: `.generated/go2_memory_closed_loop/generalized_learned_local_suite_v217_replacement_cap_20260701/`
and `.generated/go2_head_dagger/`.

## Tier 1 — replacement-safety cap: implemented, works, and is now exhausted

`--body-clearance-hard-veto-replacement-cap` (commit `977f3b1`): the
body-clearance hard veto may only replace a translating primitive when the
replacement's own learned clearance blocked probability is at or below an
absolute cap. Default 1.01 preserves old behavior.

Held-out sweep (v133 explore, recovery rerank, combined retry 3, hard veto 0.78):

| config | claimed | stalls (hard) | forwards | visited cells | hard vetoes |
|---|---|---|---|---|---|
| prior clean-frozen (hard 0.82) | red | 0 (0) | 14 | 7 | 637 |
| cap 0.70 | red | 21 (15) | 96 | 16 | 1 |
| cap 0.72 | red | 21 (15) | 96 | 16 | 1 |
| cap 0.75 | red | 40 (32) | 183 | 15 | 3 |

The freeze mechanism is fixed (637 vetoes → 1). The exposed failure is not a
threshold: among guard-permitted translations the heads carry **no signal** —
outcome head AUC 0.41 (anti-predictive), clearance head 0.52 on held-out.
At stall ticks the executed forward had outcome blocked_prob ≈ 0.05 and
clearance ≈ 0.16 with true displacement 0.0 m. No veto policy can act on a
head that scores a wall-push at 0.05.

## The pause-doc framing was too narrow: the stack fails on TRAIN scenes

Identical config run on the 13 policy-train scenes (v218 bundle, below):
**3 beacon claims out of 52 possible.** Per-scene stalls 6–676, visited cells
8–38 of the maze, and in 8/13 scenes not a single beacon color is ever seen in
700 ticks. The earlier clean all-beacon demo was one heavily-tuned reference
maze. Held-out generalization is not the frontier problem; the learned local
stack does not yet work in-distribution.

Two further mechanisms isolated:

1. **SEEK guard hole.** The wall guard runs only in EXPLORE until 3 claims
   (`--wall-guard-post-claim-min-claims 3`). On train scene 000c67 the robot
   saw red through a wall, entered SEEK, and executed `forward_medium` 591
   consecutive ticks at outcome blocked_prob 0.999 — no veto is even
   consulted in that state. The designed remedy (weak-memory-seek stall
   recovery) was disabled by `--weak-memory-seek-colors green` +
   `--weak-memory-seek-stall-streak 999`.
2. **Per-edge first-contact cost.** The online map correctly prevents exact
   repeats (21 stalls / 17 unique (cell,yaw,primitive) edges) but each wall
   costs 2–3 stalls because forward/arc_left/arc_right from the same pose are
   distinct edges.

v218 bundle (weak-seek recovery for all colors, streak 4;
`--learned-local-online-map-low-progress-block-m 0.08`; cap 0.72): unjams the
SEEK hammer (000c67 591→83 stalls, 15 recoveries) but creates no routes; on
held-out it trades more motion for more stalls (21→62) at identical coverage.
Config levers are exhausted.

## Tier 2 — DAgger-for-heads: executed, decisive negative

Pipeline (new, reusable):

- `scripts/build_go2_head_dagger_frames_from_result.py` (commit `8d845f1`)
  re-renders the ego RGB frame at every closed-loop decision pose recorded in
  a result JSON and emits counterfactual-labelable rows.
- 3,353 unique decision poses across all 13 train scenes (v217+v218 runs),
  ≈23k (pose, primitive) examples with exact kinematic grid labels, blocked
  rate 0.68 — precisely the wall-adjacent distribution the original teacher
  data missed.
- Retrained both heads (v219) with an 11/2 scene train/val split
  (`.generated/go2_wallaware_learned/*_v219_dagger13_*`).

Result:

| head | train acc | unseen-scene val acc |
|---|---|---|
| clearance v219 (obstacle, after_start_min, m0.02, h192) | 0.91 | 0.58 (F1 0.49) |
| outcome v219 (configuration, swept_min, m0.03, h224) | 0.99 | 0.61 |

With on-distribution data and exact labels, the heads memorize scenes and do
not transfer. **The frozen 96-d global JEPA latent
(`go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02`)
does not encode scene-transferable local wall geometry.** Closed-loop with
v219 heads confirms the two-sided failure: held-out goes 0 stalls but
near-frozen (10 forwards, 8 cells); train scene 010092 still 43 stalls, no
claims. The head threshold only selects which failure mode occurs.

## Verdict and recommended path

1. **Stop tuning** guards, vetoes, frontier knobs, and policy rerank weights
   on the current latent. Three independent lines of evidence (cap sweep AUC,
   train-scene AUC, v219 scene-holdout gap) say the ceiling is perceptual.
2. **Branch A (stays in the runtime contract): geometry-capable perception.**
   Replace the global-latent head input with spatial/patch features or train
   an encoder with geometric auxiliary supervision (depth/occupancy targets
   are privileged-at-training-only, allowed by the contract; runtime stays
   RGB). The phase3a latent-map machinery is the in-repo starting point.
   Offline gate before any closed-loop work: unseen-scene val accuracy ≥0.85
   on the existing `.generated/go2_head_dagger/` benchmark (2-scene holdout).
   This pipeline is cheap to iterate: frames+rows exist, training is minutes
   on the R9700.
3. **Branch B (adjust the contract): ego-depth occupancy.** Depth-based
   occupancy nav already fully completed a maze
   (docs/perception depth work, v43). If the demo claim is re-scoped to
   "learned memory + learned policy over online depth occupancy," the
   perception blocker disappears immediately.
4. **After perception, coverage is the next co-equal blocker**: runs are ~72%
   yaw with 8–38 cells visited per 700 ticks and beacons usually never seen.
   Do not re-litigate exploration until a trustworthy guard exists, then
   re-measure — much of the yaw-walk is guard-conversion of translations.
5. Keep from this session regardless of branch: the replacement cap (default
   off), weak-seek recovery enabled for all colors with stall streak ~4, and
   the SEEK guard hole fix (guard or recovery must be active pre-claim).

Tier 3 (strict checker + generalization gate) was not run: no candidate run
satisfies the gate preconditions. The checker command in the pause handoff
remains the acceptance test.
