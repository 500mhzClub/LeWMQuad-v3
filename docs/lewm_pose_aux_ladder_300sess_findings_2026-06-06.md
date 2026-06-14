# Pose-Aux Geometry Ladder — 300-session screen findings (2026-06-06)

Screen execution of the Path 2C ladder from
`docs/lewm_pose_aux_experiment_design_2026-06-06.md`. All cells init weights-only
from the fixed source
`…/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/lewm_seq11_e3.pt`, trained 1 epoch on
300 sessions, identical `--torch-seed 0` / shuffle seed, pinned e3 objective
(sigreg 0.09, rollout 0.25/h10/γ0.9, warmup 0), pose-aux weights from the measured
gradient scale (encoded 0.097, predicted 0.071 ≈ 0.1× base encoder grad). Per-cell
benchmarks via `watch_finetune_evals.sh` (CPU). Outputs under
`models/checkpoints_pose_aux_ladder_20260606/{F0,C0,C0/posthoc,C1,C2}`.

C0 has no pose head (geometry off), so its decodability is measured by **C0/posthoc**
— a fresh frozen head trained on C0's drifted encoder (the standardized drift
control). F0 is the frozen-e3 head ceiling.

## Results

cols: zf@10 · fp@10 · MPC ‖ encXY encρ · predXY predρ ‖ first-ρ regret poseNav
(nav metrics are mean progress over **only 8 scenes** — directional, noisy.)

| cell | zf@10 | fp@10 | MPC | navL2 | enNav | encXY | **encρ** | predXY | **predρ** | first-ρ | regret | poseNav |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F0 frozen ceiling | 0.217 | 0.411 | 0.71 | 0.048 | 0.062 | 0.312 | +0.067 | 0.344 | +0.076 | +0.019 | 0.093 | −0.051 |
| C0/posthoc drift ctrl | 0.228 | 0.540 | 0.65 | −0.388 | 0.318 | 0.260 | +0.121 | 0.289 | +0.148 | −0.034 | 0.136 | −0.117 |
| C1 encoded | 0.221 | 0.527 | 0.68 | −0.067 | −0.138 | 0.227 | +0.150 | 0.270 | +0.087 | −0.061 | 0.094 | +0.279 |
| C2 encoded+predicted | 0.259 | 0.521 | 0.67 | 0.238*| 0.015 | 0.238 | **+0.188** | 0.259 | +0.142 | −0.012 | 0.100 | −0.092 |

(\* C2 navL2 ≈ −0.016; table reuses encXY slot — see JSON for exact navL2.)

## Findings

1. **The geometry loss injects encoded metric decodability.** encρ rises
   monotonically with geometry pressure: F0 +0.067 → C0/posthoc +0.121 → C1 +0.150
   → **C2 +0.188**, with the lowest encoded XY error at C2. Measured over the
   512-window eval cache, this is the reliable signal of the screen.

2. **The forward model is preserved.** C2 zero-free@10 = 0.259 is the *best* of all
   cells (gate ≥ 0.18); MPC-vs-zero 0.67 (gate ≥ 0.62). `fp@10 ≈ 0.52` breaches the
   ≤ 0.50 gate, but uniformly across C0/C1/C2 (control included), so it is a
   one-extra-epoch continuation artifact, not the geometry objective.

3. **The decisive deployed contract does not beat the control.** predicted→goal
   correlation: control C0/posthoc **+0.148** vs C2 **+0.142** — a tie. predicted XY
   does improve monotonically (0.344 → 0.289 → 0.270 → 0.259), but the design's
   *decisive* offline gate is the predicted→goal **correlation**, and C2 does not
   exceed the plain-continuation control on it. C1 (encoded-only, predicted-λ=0)
   even drops to +0.087, as expected.

4. **No actionable geometry yet.** First-action Spearman is ≈0/negative for *every*
   cell; poseNav swings sign (C1 +0.279, C2 −0.092) — N=8-scene noise. Decodable
   geometry improved; the latent still cannot rank one-step actions.

## Verdict vs. promotion gates

- Forward-model no-regression: **pass** (fp marginal, continuation-attributable).
- Geometry: **partial** — encoded clears both baselines; predicted→goal ties control.
- Planning (positive first-action Spearman): **fail** for all cells.

This is the design decision-tree branch: *"C1 improves encoded geometry, C2/predictor
contract fails → encoder retains geometry but rollout projection destroys it; align
predictor/projector before navigation."* Do **not** advance to physics/nav.

## Next

1. **Confirm at 1000 sessions** (the predρ gap +0.142 vs +0.148 is within plausible
   300-session/1-epoch noise). Same 5-cell ladder, separate output dir. If C2's
   predicted→goal still fails to beat a 1000-session control, the gap is real.
2. If confirmed real, the fix is **predictor/projector alignment** (jointly align
   rollout endpoint and projection outputs) or **harder action-conditioned endpoint
   pairs**, *not* more λ tuning.
3. Re-run the planning/nav diagnostics with many more scenes/seeds before trusting
   any nav number — N=8 is uninformative here.
